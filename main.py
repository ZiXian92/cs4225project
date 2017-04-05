from svmutil import svm_train, svm_predict
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
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
    for idx, prediction in enumerate(labels):
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
    sparkcontext.addFile('hdfs://ec2-34-204-54-226.compute-1.amazonaws.com:9000/libsvm-322', True)
    sparksql = SparkSession.builder.master('local').appName('hddpredict').getOrCreate()

    # Load the entire data and project wanted columns
    # Then parition by individual hard disk and sort by date so we can
    # model partition as time series and compute rate of change of attributes.
    # drivedatadf = sparksql.read.csv('/user/zixian/project/input/*.csv', inferSchema = True, header = True)
    drivedatadf = sparksql.read.csv('hdfs://ec2-34-204-54-226.compute-1.amazonaws.com:9000/data/*.csv', inferSchema = True, header = True)
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

    # Data extractions and transformations begin here!
    # Get records for good training drives. Output format: Row.
    traingoodrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in traingoodset.value)
    # Keyed by drive identifier
    traingoodrdd = traingoodrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Pick 4 sample records per good drive for training. Output format: Row
    traingoodrdd = traingoodrdd.flatMap(lambda r: sampleK(r[1], 4))
    # Convert to (model, [0, features]). Change the key as needed.
    traingoodrdd = traingoodrdd.map(lambda r: (r.model, [0]+map(lambda col: r[col], desiredcolumns[4:])))

    # Get records for failed training drives. Output format: Row.
    trainfailedrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in trainfailedset.value)
    # Outputformat: ((sn, model), Row)
    trainfailedrdd = trainfailedrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Sort all records by record's date and extract last 10 days of operations. Output format: Row
    trainfailedrdd = trainfailedrdd.flatMap(lambda r: sorted(r[1], None, lambda p: p.date, True)[:10])
    # Output format: (key, [1, features]). Change the key as needed.
    trainfailedrdd = trainfailedrdd.map(lambda r: (r.model, [1]+map(lambda col: r[col], desiredcolumns[4:])))

    # Combine the training data sets. This part should not need any changes.
    # Output format: (key, [[0 or 1, features]])
    trainingrdd = traingoodrdd.union(trainfailedrdd).groupByKey().mapValues(list)
    # Output format: (key, [[training_labels], [[features]]])
    trainingrdd = trainingrdd.map(lambda r: (r[0], [map(lambda x: x[0], r[1]), map(lambda x: x[1:], r[1])]))

    # Get records for good test drives. Output format: Row.
    testgoodrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in testgoodset.value)
    # Any further processing if needed
    # Output format: (key, [0, features]). Change the key as you need.
    testgoodrdd = testgoodrdd.map(lambda r: (r.model, [0]+map(lambda col: r[col], desiredcolumns[4:])))

    # Get records for failed test drives.
    testfailedrdd = drivedatadf.rdd.filter(lambda r: (r.serial_number, r.model) in testfailedset.value)
    # Extract last 10 days of records. Output format: ((sn, model), Row). Add futher processing as needed.
    testfailedrdd = testfailedrdd.map(lambda r: ((r.serial_number, r.model), r)).groupByKey().mapValues(list)
    # Sort all records by record's date and extract last 10 days of operations. Output format: Row
    testfailedrdd = testfailedrdd.flatMap(lambda r: sorted(r[1], None, lambda p: p.date, True)[:10])
    # Output format: (key, [1, features]). Change the key as you need.
    testfailedrdd = testfailedrdd.map(lambda r: (r.model, [1]+map(lambda col: r[col], desiredcolumns[4:])))
    # End of modifiable transformations

    # Combine test data sets. Should not need any changes.
    # Output format: (key, [0 or 1, features])
    testrdd = testgoodrdd.union(testfailedrdd).groupByKey().mapValues(list)
    # Output format: (key, [[expected_labels], [[features]]])
    testrdd = testrdd.map(lambda r: (r[0], [map(lambda x: x[0], r[1]), map(lambda x: x[1:], r[1])]))

    # Combine training and test data set. union and groupByKeys seem to preserve ordering of values on collect().
    # This part shouldn't need any change.
    # Output format: (key, [[[training_labels], [[features]]], [[expected_labels], [[features]]]])
    modellingrdd = trainingrdd.union(testrdd).groupByKey().mapValues(list)

    # Run SVM per key
    predictionStats = modellingrdd.map(getPredictionStats).reduce(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]))
    predictionRate = predictionStats[0]*100.0/predictionStats[1]
    falseAlarmRate = predictionStats[2]*100.0/predictionStats[3]

    # Print prediction rate and false alarm rate
    print 'Prediction Rate: ', str(predictionRate)+'%'
    print 'False Alarm Rate: ', str(falseAlarmRate)+'%'

    # Done preprocessing all records. Proceed to free up memory.
    drivedatadf.unpersist()
