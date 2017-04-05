from svmutil import svm_train, svm_predict
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from datetime import date
import random

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

# Options to be passed to SVM for training
svm_options = '-s 0 -t 2 -c 10'

# rddEntry format: (key, [[[training_labels], [training_data]], [[expected_labels], [test_data]]])
# Output format: (numFailedPredictions, expectedFailedPredictions, numFalseAlarms, numGoodRecords)
def getPredictionStats(rddEntry):
    model = svm_train(rddEntry[1][0][0], rddEntry[1][0][1], svm_options)
    labels, acc, values = svm_predict(rddEntry[1][1][0], rddEntry[1][1][1], model)
    numFailedPredictions, expectedFailedPredictions, numFalseAlarms, numGoodRecords = 0, 0, 0, 0
    for idx prediction in enumerate(labels):
        if rddEntry[1][1][0][idx]==1:
            expectedFailedPredictions+=1
            if labels[idx]==rddEntry[1][1][0][idx]:
                numFailedPredictions+=1
        else:
            numGoodRecords+=1
            if labels[idx]!=rddEntry[1][1][0][idx]:
                numFalseAlarms+=1
    return (numFailedPredictions, expectedFailedPredictions, numFalseAlarms, numGoodRecords)

# Prepare desired columns
desiredsmartnos = [1, 3, 5, 7, 9, 187, 189, 194, 195, 197]
desiredcolumns = ['date', 'serial_number', 'model', 'failure']
for sno in desiredsmartnos:
    desiredcolumns.append('smart_'+str(sno)+'_normalized')
    desiredcolumns.append('smart_'+str(sno)+'_raw')

if __name__ == "__main__":
    sparkconf = SparkConf().setAppName('hddpredict')
    sparkcontext = SparkContext(conf=sparkconf)
    sparksql = SparkSession.builder.master('local').appName('hddpredict').getOrCreate()

    # Load the entire data and project wanted columns
    # Then parition by individual hard disk and sort by date so we can
    # model partition as time series and compute rate of change of attributes.
    drivedatadf = sparksql.read.csv('/user/zixian/project/input/*.csv', inferSchema = True, header = True)
    drivedatadf = drivedatadf.select(desiredcolumns).fillna(0)
    # drivedatadf = drivedatadf.repartition('serial_number', 'model')
    drivedatadf.cache()

    # Get list of distinct drives
    distinctdrivesdf = drivedatadf.groupBy('serial_number', 'model').agg({'failure': 'max'}).withColumnRenamed('max(failure)', 'failure').select('serial_number', 'model', 'failure')
    distinctdrivesdf.cache()

    # Generate list of good and bad drives
    faileddrivedf = distinctdrivesdf.filter('failure = 1').distinct().select('serial_number', 'model')
    gooddrivesdf = distinctdrivesdf.filter('failure = 0').distinct().select('serial_number', 'model')
    distinctdrivesdf.unpersist()

    # Split failed drives into testing and training
    trainfaileddf, testfaileddf = faileddrivedf.randomSplit([0.7, 0.3])
    trainfailedset = sparkcontext.broadcast(set(trainfaileddf.rdd.map(lambda r: (r.serial_number, r.model)).collect()))
    testfailedset = sparkcontext.broadcast(set(testfaileddf.rdd.map(lambda r: (r.serial_number, r.model)).collect()))

    # Split good drives into testing and training
    traingooddf, testgooddf = gooddrivesdf.randomSplit([0.7, 0.3])
    traingoodset = sparkcontext.broadcast(set(traingooddf.rdd.map(lambda r: (r.serial_number, r.model)).collect()))
    testgoodset = sparkcontext.broadcast(set(testgooddf.rdd.map(lambda r: (r.serial_number, r.model)).collect()))

    # Compute rate of change of raw values of the 2 smart values(how?)
    # Leaving this out for now. Prioritizing whole flow of SVM

    # Data extractions and transformations begin here!
    # Get records for good training drives
    traingoodrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in traingoodset.value)
    # Keyed by drive identifier
    traingoodrdd = traingoodrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Pick 4 sample records per good drive for training. Output format: Row
    traingoodrdd = traingoodrdd.flatMap(lambda r: sampleK(r[1], 4))
    # Convert to (model, [0, features])
    traingoodrdd = traingoodrdd.map(lambda r: (r.model, [0]+map(lambda col: r[col], desiredcolumns[4:])))

    # Get records for failed training drives
    trainfailedrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in trainfailedset.value)
    # Outputformat: ((sn, model), Row)
    trainfailedrdd = trainfailedrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Sort all records by record's date and extract last 10 days of operations. Output format: Row
    trainfailedrdd = trainfailedrdd.flatMap(lambda r: sorted(r[1], None, lambda p: p.date, True)[:10])
    # Output format: (model, [1, features])
    trainfailedrdd = trainfailedrdd.map(lambda r: (r.model, [1]+map(lambda col: r[col], desiredcolumns[4:])))

    # Combine the training data sets
    # Output format: (model, [[0 or 1, features]])
    trainingrdd = traingoodrdd.union(trainfailedrdd).groupByKey().mapValues(list)

    # Get records for good test drives
    # testgoodrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in testgoodset.value)
    # Convert records to data for SVM
    # testgoodrdd = testgoodrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:]), True)
    # testgoodrdd.cache()
    # testgoodrdd.repartition(1).saveAsTextFile('/user/zixian/project/test-good-data')

    # Get records for failed test drives
    # testfailedrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in testfailedset.value)
    # Keyed by drive identifier and groups records by drive identifier
    # testfailedrdd = testfailedrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Sort all records by record's date and extract last 10 days of operations
    # testfailedrdd = testfailedrdd.flatMap(lambda r: sorted(r[1], None, lambda p: p.date, True)[:10])
    # Convert records to data for SVM
    # testfailedrdd = testfailedrdd.map(lambda r: map(lambda col: r[col], desiredcolumns[4:]), True)
    # testfailedrdd.cache()
    # testrdd = testgoodrdd.union(testfailedrdd)
    # testfailedrdd.repartition(1).saveAsTextFile('/user/zixian/project/test-failed-data')

    # Done preprocessing all records. Proceed to free up memory.
    drivedatadf.unpersist()

    # Train the SVM model
    # svm = SVMWithSGD.train(trainingrdd)
    # svm.setThreshold(0.5)

    # Run predictions
    # mocktestrdd = trainfailedrdd.map(lambda lp: lp.features)
    # trainfailedrdd.unpersist()
    # mocktestrdd.cache()
    # mocktestrdd.repartition(1).saveAsTextFile('/user/zixian/project/mock-test-failed-data')
    # svm.predict(mocktestrdd).repartition(1).saveAsTextFile('/user/zixian/project/failed-drive-prediction')
    # mocktestrdd.unpersist()
    # predictedFailures = svm.predict(testfailedrdd).reduce(lambda a, b: a+b)
    # actualFailures = testfailedrdd.count()
    # predictionRate = predictedFailures*100.0/actualFailures
    # testfailedrdd.unpersist()

    # predictedFailures = svm.predict(testgoodrdd).filter(lambda x: x==0).count()
    # falseAlarmRate = predictedFailures*100.0/testgoodrdd.count()
    # testgoodrdd.unpersist()

    # print 'Prediction Rate: ', str(predictionRate)+'%'
    # print 'False Alarm Rate: ', str(falseAlarmRate)+'%'
