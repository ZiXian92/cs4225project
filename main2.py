from svmutil import svm_train, svm_predict
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from datetime import date, datetime, timedelta
import random

# Prepare desired columns
desiredsmartnos = [1, 3, 5, 7, 9, 194, 197]
desiredcolumns = ['date', 'serial_number', 'model', 'failure']
for sno in desiredsmartnos:
    desiredcolumns.append('smart_'+str(sno)+'_normalized')
    desiredcolumns.append('smart_'+str(sno)+'_raw')
for feature in desiredcolumns[4:4+2*len(desiredsmartnos)]:
    desiredcolumns.append('del_'+feature)

# Computes rate of change of smart values
# Input is a list of dictionaries
def computeChangeRates(records):
    for idx, record in enumerate(records):
        for col in desiredcolumns[4:4+2*len(desiredsmartnos)]:
            records[idx]['del_'+col] = 0 if idx==0 else records[idx][col]-records[idx-1][col]
    return records

# Defines reservoir sampling to get k items out of the given item list
def sampleK(itemList, k):
    sampleList = []
    for idx, item in enumerate(itemList, start=1):
        if idx <= k:
            sampleList.append(item)
        else:
            token = random.randint(0, idx-1)
            if token < k:
                sampleList[token] = item
    return sampleList

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
    sparkcontext.addFile('hdfs://ec2-34-204-54-226.compute-1.amazonaws.com:9000/libsvm-322', True)
    sparksql = SparkSession.builder.master('local').appName('hddpredict').getOrCreate()

    # Load the entire data and project wanted columns
    # Then, compute rate of change for remaining smart attributes
    # drivedatadf = sparksql.read.csv('/user/zixian/project/input/*.csv', inferSchema = True, header = True)
    drivedatadf = sparksql.read.csv('hdfs://ec2-34-204-54-226.compute-1.amazonaws.com:9000/data/2016-11-*.csv', inferSchema = True, header = True)
    drivedatadf = drivedatadf.select(desiredcolumns[:4+2*len(desiredsmartnos)]).fillna(0)
    drivedatardd = drivedatadf.rdd.map(lambda r: r.asDict())
    # Output format: [(key, [records])]
    drivedatardd = drivedatardd.map(lambda r: ((r['model'], r['serial_number']), r)).groupByKey().mapValues(list)
    drivedatardd = drivedatardd.map(lambda r: (r[0], sorted(r[1], None, lambda x: x['date'])))
    drivedatardd = drivedatardd.flatMap(lambda r: computeChangeRates(r[1]))
    drivedatadf = sparksql.createDataFrame(drivedatardd)
    drivedatadf.cache()

    # Set of dates to iterate over
    datesset = drivedatadf.filter(drivedatadf.failure==1).select('date').take(5)
    datesset = [datesset[len(datesset)/2].date]
    # datesset = set(drivedatadf.select('date').distinct().rdd.map(lambda r: r.date).collect())

    numDaysWindow = 10
    daysDifference = timedelta(10)
    combinePredictionStats = (lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]))

    # for day in sorted(datesset)[15:16]:
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
        # numGoodSample = traininggoodrecordsrdd.countApprox()
        traininggoodrecordsrdd = traininggoodrecordsrdd.sample(False, 0.0001)
        trainingfailedrecordsrdd = trainingfailedrecordsrdd.sample(False, 1.0)
        traininggoodrecordsrdd.cache()
        trainingfailedrecordsrdd.cache()
        numGoodSamples = traininggoodrecordsrdd.count()
        numFailedSamples = trainingfailedrecordsrdd.count()
        traininglabels = traininggoodrecordsrdd.map(lambda r: 0).collect()+trainingfailedrecordsrdd.map(lambda r: 1).collect()
        trainingdata = traininggoodrecordsrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:])).collect()
        trainingdata+=trainingfailedrecordsrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:])).collect()
        traininggoodrecordsrdd.unpersist()
        trainingfailedrecordsrdd.unpersist()
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
        drivedatadf.unpersist()
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

    # Clean up at the end
    # drivedatadf.unpersist()
