#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::text;

//Calculate edit distance netween two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_lenght(const string &a, const string &b);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
vector<string> text_recog(Mat &image);
int subdivide(const cv::Mat &img, const int rowDivisor, const int colDivisor, std::vector<cv::Mat> &blocks);

//Perform text detection and recognition and evaluate results using edit distance
int main(int argc, char* argv[])
{
    int ROW = 3, COL = 3;
    Mat image;
    if(argc > 1) {
        image  = imread(argv[1]);
        if (argc > 2)
            ROW = atoi(argv[2]);
        if (argc > 3)
            COL = atoi(argv[3]);
    }
    else 
        return(0);
    vector<Mat> blocks;
    subdivide(image, ROW, COL, blocks);
    vector<string> words;
    for (int i = 0; i < (int)blocks.size(); i++) {
        vector<string> v = text_recog(blocks[i]);
        words.insert(words.end(), v.begin(), v.end());
    }
    cout << "\n###result:\n" << endl;
    for (int i = 0; i < (int)words.size(); i++)
        cout << words[i] << ' ';
    cout << endl;
    return 0;
}

int subdivide(const cv::Mat &img, const int rowDivisor, const int colDivisor, std::vector<cv::Mat> &blocks)
{        
    /* Checking if the image was passed correctly */
    if(!img.data || img.empty())
        exit(1);
    /* Cloning the image to another for visualization later, if you do not want to visualize the result just comment every line related to visualization */
    cv::Mat maskImg = img.clone();
    /* Checking if the clone image was cloned correctly */
    if(!maskImg.data || maskImg.empty())
        exit(1);
    // check if divisors fit to image dimensions
    int rowW = img.rows / rowDivisor + 80, colW = img.cols / colDivisor + 80;
    for(int y = 0; y < img.cols; y += img.cols / colDivisor)
    {
        for(int x = 0; x < img.rows; x += img.rows / rowDivisor)
        {   
            cv::Rect rect;
            rect =  cv::Rect(y,x, colW <= img.cols - y ? colW : img.cols - y, rowW <= img.rows - x ? rowW : img.rows - x);
            
            blocks.push_back(cv::Mat(img, rect));
        }
    }
    return EXIT_SUCCESS;
}

vector<string> text_recog(Mat &image)
{
    /*Text Detection*/

    // Extract channels to be processed individually
    vector<Mat> channels;

    Mat grey;
    cvtColor(image,grey,COLOR_RGB2GRAY);

    // Notice here we are only using grey channel, see textdetection.cpp for example with more channels
    channels.push_back(grey);
    channels.push_back(255-grey);

    double t_d = (double)getTickCount();
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);

    /*Text Recognition (OCR)*/

    Ptr<OCRTesseract> ocr = OCRTesseract::create();
    string output;

    float scale_img  = 600.f/image.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;

    for (int i=0; i<(int)nm_boxes.size(); i++)
    {

        Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));

        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());

        if (output.size() < 3)
            continue;

        for (int j=0; j<(int)boxes.size(); j++)
        {
            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < 60)) ||
                    isRepetitive(words[j]))
                continue;
            words_detection.push_back(words[j]);
        }

    }

    return words_detection;
}

size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

bool   sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}
