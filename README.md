Project: Confidential Photo Detection on IOS
Teammember: Chang Chen (cc3757), Changchang Wang (cw2826), Liang Wu (lw2589), Jialu Zhong (jz2612)

4 components of the project are included in the 4 directories.
Server:
This directory consists of a maven built JAVA server used to run on AWS EC2 intances.

TextRecognition:
A C++ code for text recognition is included in this directory. Dependencies needed: OpenCV, OpenCV_contrib, Tesseract

models:
python codes for text classification. Dependencies needed: Spark, Python NLTK

s3-swift:
swift codes for IOS App
