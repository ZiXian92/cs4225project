from svmutil import svm_train, svm_predict
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
from datetime import date, datetime, timedelta
import random

# Prepare desired columns
desiredsmartnos = ['smart_1_normalized', 'smart_5_raw', 'smart_187_normalized', 'smart_188_raw', 'smart_197_normalized', 'smart_198_raw']
desiredcolumns = ['date', 'serial_number', 'model', 'failure']+desiredsmartnos
for feature in desiredsmartnos:
    desiredcolumns.append('del_'+feature)

# Computes rate of change of smart values
# Input is a list of dictionaries
def computeChangeRates(records):
    for idx, record in enumerate(records):
        for col in desiredcolumns[4:4+len(desiredsmartnos)]:
            records[idx]['del_'+col] = 0 if idx==0 else records[idx][col]-records[idx-1][col]
    return records

# rddEntry format: [features]
# Output format: (numFailedPredictions, expectedFailedPredictions, numFalseAlarms, numGoodRecords)
def getPredictionStats(rddEntries, trainingdata, expectedLabel):
    output = [0, 0, 0, 0]
    # Options to be passed to SVM for training
    svm_options = '-s 0 -t 2 -c 10'
    # Train SVM model once per partition
    model = svm_train(trainingdata[0], trainingdata[1], svm_options)
    for entry in rddEntries:
        labels, acc, values = svm_predict([expectedLabel], [entry], model)
        if expectedLabel==1:
            output[0]+=labels[0]
            output[1]+=1
        else:
            output[2]+=labels[0]
            output[3]+=1
    return [(output[0], output[1], output[2], output[3])]

if __name__ == "__main__":
    sparkconf = SparkConf().setAppName('hddpredict')
    sparkcontext = SparkContext(conf=sparkconf)
    sparksql = SparkSession.builder.master('local').appName('hddpredict').getOrCreate()

    # Load the entire data and project wanted columns
    # Then, compute rate of change for remaining smart attributes
    drivedatadf = sparksql.read.csv('hdfs://spark-master:9000/user/zixian/project/input/*.csv', inferSchema = True, header = True)
    drivedatadf = drivedatadf.select(desiredcolumns[:4+len(desiredsmartnos)]).fillna(0)
    drivedatardd = drivedatadf.rdd.map(lambda r: r.asDict())

    # Compute rate of change for each smart attribute
    # Output format: [(key, [records])]
    drivedatardd = drivedatardd.map(lambda r: ((r['model'], r['serial_number']), r)).groupByKey().mapValues(list)
    drivedatardd = drivedatardd.map(lambda r: (r[0], sorted(r[1], None, lambda x: x['date'])))
    drivedatardd = drivedatardd.flatMap(lambda r: computeChangeRates(r[1]))
    drivedatadf = sparksql.createDataFrame(drivedatardd)
    drivedatadf.cache()

    # Set of dates to iterate over
    datesset = sorted(drivedatadf.select('date').distinct().rdd.map(lambda r: r.date).collect())

    numDaysWindow = 7
    daysDifference = timedelta(numDaysWindow)
    combinePredictionStats = (lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]))

    outputfile = open('predictions.csv', 'w')

    for day in datesset:
        # Check if we have training data and can get expected labels. If not, skip prediction for this day.
        # Get the records in the past time window, excluding today as training set
        trainningrecordsdf = drivedatadf.filter((drivedatadf.date<day) & (drivedatadf.date>=day-daysDifference))
        trainningrecordsdf.cache()
        if trainningrecordsdf.count()==0:
            trainningrecordsdf.unpersist()
            continue

        # Get future drive records, to identify correct results for test data
        verificationrecordsdf = drivedatadf.filter((drivedatadf.date>=day) & (drivedatadf.date<=day+daysDifference)).select('serial_number', 'model', 'failure')
        verificationrecordsdf.cache()
        if verificationrecordsdf.count()==0:
            verificationrecordsdf.unpersist()
            continue

        # Identify good and bad drives in training set
        trainingdrivesdf = trainningrecordsdf.select('serial_number', 'model', 'failure').groupBy('serial_number', 'model').agg({'failure': 'max'}).withColumnRenamed('max(failure)', 'failure')
        trainingdrivesdf.cache()
        traininggooddriveset = sparkcontext.broadcast(set(trainingdrivesdf.filter('failure = 0').distinct().select('serial_number', 'model').collect()))
        trainingfaileddriveset = sparkcontext.broadcast(set(trainingdrivesdf.filter('failure = 1').distinct().select('serial_number', 'model').collect()))
        trainingdrivesdf.unpersist()

        # Take very small amount of sample records from both failed and good drive records.
        # Aim is to be able to hold all training data in 1 partition, to be broadcasted later.
        traininggoodrecordsrdd = trainningrecordsdf.rdd.filter(lambda r: (r.serial_number, r.model) in traininggooddriveset.value)
        trainingfailedrecordsrdd = trainningrecordsdf.rdd.filter(lambda r: (r.serial_number, r.model) in trainingfaileddriveset.value)
        trainningrecordsdf.unpersist()
        # Tune these sampling ratios to tweak between prediction rate ad false alarm rates
        trainingfailedsamplesrdd = trainingfailedrecordsrdd.sample(False, 1.0)
        trainingfailedsamplesrdd.cache()
        numFailedSamples = trainingfailedsamplesrdd.count()
        clusteringModel = KMeans.train(traininggoodrecordsrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:])), numFailedSamples)
        traininggoodsamples = map(lambda s: s.tolist(), clusteringModel.clusterCenters)
        numGoodSamples = len(traininggoodsamples)
        traininglabels = map(lambda s: 0, traininggoodsamples)+trainingfailedsamplesrdd.map(lambda r: 1).collect()
        trainingdata = traininggoodsamples+trainingfailedsamplesrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:])).collect()
        trainingfailedsamplesrdd.unpersist()
        trainingdata = [traininglabels, trainingdata]

        # Prepare future data to facilitate finding expected labels
        verificationdrivesdf = verificationrecordsdf.groupBy('serial_number', 'model').agg({'failure': 'max'}).withColumnRenamed('max(failure)', 'failure')
        verificationrecordsdf.unpersist()
        verificationdrivesdf.cache()
        # Good and bad test drive sets for filtering of test data
        testfaileddriveset = sparkcontext.broadcast(set(verificationdrivesdf.filter('failure = 1').distinct().select('serial_number', 'model').collect()))
        testgooddriveset = sparkcontext.broadcast(set(verificationdrivesdf.filter('failure = 0').distinct().select('serial_number', 'model').collect()))
        verificationdrivesdf.unpersist()

        # Get today's drive records and process into SVM data
        todayrecordssdf = drivedatadf.filter(drivedatadf.date==day)
        todayrecordssdf.cache()
        todaygoodrecordsrdd = todayrecordssdf.rdd.filter(lambda r: (r.serial_number, r.model) in testgooddriveset.value)
        todayfailedrecordsrdd = todayrecordssdf.rdd.filter(lambda r: (r.serial_number, r.model) in testfaileddriveset.value)
        todayrecordssdf.unpersist()
        # Convert to features list
        todaygoodrdd = todaygoodrecordsrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:]))
        todayfailedrdd = todayfailedrecordsrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:]))

        # Run predictions
        trainingdata = sparkcontext.broadcast(trainingdata)
        goodpredictrdd = todaygoodrdd.mapPartitions(lambda records: getPredictionStats(records, trainingdata.value, 0))
        failedpredictrdd = todayfailedrdd.mapPartitions(lambda records: getPredictionStats(records, trainingdata.value, 1))
        predictionStats = goodpredictrdd.aggregate((0, 0, 0, 0), combinePredictionStats, combinePredictionStats)
        predictionStats = combinePredictionStats(predictionStats, failedpredictrdd.aggregate((0, 0, 0, 0), combinePredictionStats, combinePredictionStats))
        print 'Predictions for', day
        print 'Samples:', numGoodSamples, ',', numFailedSamples
        print predictionStats
        outputfile.write(str(day)+','+str(predictionStats[0])+','+str(predictionStats[1])+','+str(predictionStats[2])+','+str(predictionStats[3])+'\n')
	    outputfile.flush()

    # Clean up at the end
    drivedatadf.unpersist()
    outputfile.close()
