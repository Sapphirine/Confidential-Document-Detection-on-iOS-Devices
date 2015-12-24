import json
import sys

def main():
	if len(sys.argv) < 2:
		print "Usage:  classifyByKey <text>"
		sys.exit(1)

	with open('/Users/Jialu/Documents/Fall2015/BigData/FinalProj/data/output/key2.json', 'r') as fp:
		data = json.load(fp)

	input_str = sys.argv[1]
	text = input_str.split(" ")
	total = data["total"]
	key_dict = data["key"]

	score = 0

	for w in text:
		if w in key_dict:
			print("%s: %i" % (w, key_dict[w]))
			score = score + key_dict[w] / float (total)

	print score

	# change threshold
	if score >= 0.01
		print "CONFIDENTIAL"
	else:
		print "SAFE"

if __name__ == '__main__': main()