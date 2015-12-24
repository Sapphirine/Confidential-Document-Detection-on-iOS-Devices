from pyspark import SparkContext
from math import log
from nltk.corpus import stopwords
import operator
import re
import string
import csv
import json



sc = SparkContext()
#print "Running Spark Version %s" % (sc.version)
stop = stopwords.words('english')


pospath = "/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/glossary"

pos_file = sc.textFile(pospath)
pos_counts = pos_file.flatMap(lambda line: ''.join([i for i in line if i not in string.punctuation]).encode('utf8').strip().lower().split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b, 1) \
             .map(lambda (a, b): (b, a)) \
    		 .sortByKey(0, 1) \
    		 .map(lambda (a, b): (b, a))

pos_output = pos_counts.collect()

key_dict = {}
total = 0
word_count = 0
for (word, count) in pos_output:
	if word not in stop and len(word) > 1 and count > 1:
		key_dict[word] = count
		total = total + count
		if count > 100:
			word_count = word_count + 1
		#print("%s: %i" % (word, count))
# print total 
data = {"total" : total, "key" : key_dict}
print "word_count" + str(word_count)
with open('/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/output/key2.json', 'w') as fp:
    json.dump(data, fp)


sc.stop()
