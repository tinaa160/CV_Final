#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

vector<KeyPoint> keypoint_ROI(vector<KeyPoint> keypoint,int x,int y, int w, int h);

int main(){
/*	VideoCapture video("VideoTest.avi");
	Mat frame;
	int index = 0;

	while(1){
		video >> frame;
		if(frame.empty()){
			cout << "no video!" << endl;
			break;
		}
		index ++;
		Mat img_kp;

		int minHessian = 400;
		Ptr<SURF> detector = SURF::create(minHessian);
		vector<KeyPoint> keypoint;
		detector->detect(frame,keypoint);

		drawKeypoints( frame, keypoint, img_kp, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

		imshow("video",img_kp);
		waitKey(33);

		if( index > 450){
			cout << "save " << index << "image" << endl;
			imwrite("frame"+to_string(index)+".jpg",frame);
			//waitKey(0);
		}

	}*/
	int x = 200;
	int y = 100;
	int w = 250;
	int h = 150;

	Mat img_1 = imread("frame436.jpg");
	Mat img_2 = imread("frame505.jpg");
	Mat img_kp;

	Ptr<ORB> orb = ORB::create();
	
	vector<KeyPoint> tmp_keypoint1, tmp_keypoint2;
	vector<KeyPoint> keypoint1, keypoint2;
	

    orb-> detect(img_1, tmp_keypoint1);
    orb -> detect(img_2, tmp_keypoint2);

    keypoint1 = keypoint_ROI(tmp_keypoint1,x,y,w,h);
    keypoint2 = keypoint_ROI(tmp_keypoint2,x,y,w,h);

	Mat descriptors_1, descriptors_2;

    orb-> compute(img_1, keypoint1, descriptors_1);
    orb -> compute(img_2, keypoint2, descriptors_2);

    vector<DMatch> matches;
    BFMatcher bfMatcher(NORM_HAMMING);
    bfMatcher.match(descriptors_1, descriptors_2, matches);


	double max_dist = 0; double min_dist = 100;

	for(int i = 0; i < descriptors_1.rows; i++){
		double dist = matches[i].distance;
		if( dist < min_dist) min_dist = dist;
		if( dist > max_dist) max_dist = dist;
	}

	printf("--Max dist : %f \n", max_dist);
	printf("--Min dist : %f \n", min_dist);

	vector< DMatch > good_matches;

	for(int i = 0; i < descriptors_1.rows; i++){
		if(matches[i].distance <= max(10*min_dist, 0.02)){
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches( img_1, keypoint1, img_2, keypoint2,
				 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow("Good Matches", img_matches);

	waitKey(0);

	return 1;
}

vector<KeyPoint> keypoint_ROI(vector<KeyPoint> keypoint, int x, int y, int w, int h){
	vector<KeyPoint> ROI_keypoint;

	for(int i = 0; i < keypoint.size(); i++){
		if(keypoint[i].pt.x >= x && keypoint[i].pt.x < x+w && keypoint[i].pt.y >= y && keypoint[i].pt.y < y+h){
			ROI_keypoint.push_back(keypoint[i]);
		}
	}
	return ROI_keypoint;
}