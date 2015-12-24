from pyspark import SparkContext
from math import log
from nltk.corpus import stopwords
import operator
import re
import string



sc = SparkContext()
#print "Running Spark Version %s" % (sc.version)
stop = stopwords.words('english')


pospath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/poscomp"
negpath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/negcomp"
#posTest = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/postest"
#path2 = "/Users/Jialu/Documents/Fall2015/BigData/hw3/tf_idf"
#path3 = "/Users/Jialu/Documents/Fall2015/BigData/hw3/result"
result = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/result"


pos_file = sc.textFile(pospath)
pos_counts = pos_file.flatMap(lambda line: ''.join([i for i in line if i not in string.punctuation]).encode('utf8').strip().lower().split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b, 1) \
             .map(lambda (a, b): (b, a)) \
    		 .sortByKey(0, 1) \
    		 .map(lambda (a, b): (b, a))

neg_file = sc.textFile(negpath)
neg_counts = neg_file.flatMap(lambda line: ''.join([i for i in line if i not in string.punctuation]).encode('utf8').strip().lower().split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b, 1) \
             .map(lambda (a, b): (b, a)) \
    		 .sortByKey(0, 1) \
    		 .map(lambda (a, b): (b, a))

pos_output = pos_counts.collect()
neg_output = neg_counts.collect()

word_dict = {}

for (word, count) in pos_output:
	if word in word_dict:
		word_dict[word] = word_dict[word] + count
	else:
		word_dict[word] = count

for (word, count) in neg_output:
	if word in word_dict:
		word_dict[word] = word_dict[word] + count
	else:
		word_dict[word] = count

sorted_dict = {}

for (word, count) in pos_output:
	if word not in stop:
		measure = (count * log(count)) / float(word_dict[word])
		measure2 = count / float(word_dict[word])
		if measure > 1 and measure2 > 0.55:
			sorted_dict[word] = measure

sorted_dict_x = sorted(sorted_dict.iteritems(), key=operator.itemgetter(1))
for t in sorted_dict_x:
	print("%s: %.2f" % (t[0], t[1]))
#print sorted_dict_x
			#print("%s: %.2f" % (word, measure))
#counts.saveAsTextFile(result)


sc.stop()



