from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD, SVMModel


sc = SparkContext()

pospath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/poscomp"
negpath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/negcomp"

testpath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/test"

htf = HashingTF()
positiveData = sc.wholeTextFiles(pospath)
posdata = positiveData.map(lambda (name,text) : LabeledPoint(1, htf.transform(text.split(" "))))
negtiveData = sc.wholeTextFiles(negpath)
negdata = negtiveData.map(lambda (name,text) : LabeledPoint(0, htf.transform(text.split(" "))))

print "====================No. of Positive Sentences: ====================" + str(posdata.count())
print "====================No. of Negative Sentences: ====================" + str(negdata.count())


posdata.persist()
negdata.persist()

ptrain, ptest = posdata.randomSplit([0.6, 0.4])
ntrain, ntest = negdata.randomSplit([0.6, 0.4])

trainh = ptrain.union(ntrain)
testh = ptest.union(ntest)
model = SVMWithSGD.train(trainh, iterations=100)

prediction_and_labels = testh.map(lambda point: (model.predict(point.features), point.label))

correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

accuracy = correct.count() / float(testh.count())

print "====================Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"


truePos = prediction_and_labels.filter(lambda (predicted, actual): (actual == 1) and (predicted == 1))
falsePos = prediction_and_labels.filter(lambda (predicted, actual): (actual == 0) and (predicted == 1))
trueNeg = prediction_and_labels.filter(lambda (predicted, actual): (actual == 0) and (predicted == 0))
falseNeg = prediction_and_labels.filter(lambda (predicted, actual): (actual == 1) and (predicted == 0))

truePositive = truePos.count() / float(testh.count())
falsePositive = falsePos.count() / float(testh.count())
trueNegative = trueNeg.count() / float(testh.count())
falseNegative = falseNeg.count() / float(testh.count())

recall = truePos.count() / float(truePos.count() + falseNeg.count())
precision = truePos.count() / float(truePos.count() + falsePos.count())

print "====================True Positive " + str(truePositive * 100) + "===================="
print "====================False Positive " + str(falsePositive * 100) + "===================="
print "====================True Negative " + str(trueNegative * 100) + "===================="
print "====================False Negative " + str(falseNegative * 100) + "===================="

print "====================Recall " + str(recall * 100) + "===================="
print "====================Precision " + str(precision * 100) + "===================="

sc.stop()


