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
vector<DMatch> matches_larger(vector<DMatch> good_matches, int size_diff, vector<KeyPoint> kp1, vector<KeyPoint> kp2);
void find_keypoint(Mat img1, vector<KeyPoint> &keypoint1, Mat img2, vector<KeyPoint> &keypoint2);
void find_matches(Mat img_1, Mat img_2,vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2, int size_diff, float sensitivity, vector<DMatch> &filter_matches);
void detect_obstacle(Mat img_1, Mat img_2);
vector<bool> template_matching(Mat img_1, Mat img_2, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<DMatch> matches);

int main(){
	VideoCapture video("chessBoard.avi");
	Mat frame;
	int index = 0;
	bool detect_mode = false;
	vector<Mat> previous_frame;
	int n_skip = 20;

	while(1){
		video >> frame;
		previous_frame.push_back(frame.clone());

		if(frame.empty()){
			cout << "no video!" << endl;
			break;
		}


		if( index > n_skip){
			detect_mode = true;
		}

		if(detect_mode && index > 400){
			//cout << "previous_frame:" << previous_frame.size() << endl;
			Mat img_1 = previous_frame[index - n_skip];
			Mat img_2 = frame;

			cvtColor(img_1,img_1,CV_BGR2GRAY);
			cvtColor(img_2,img_2,CV_BGR2GRAY);

			cout << "current index:" << index << endl;

	
			detect_obstacle(img_1,img_2);
		}

		index ++;
		
		imshow("video",frame);
		waitKey(33);
	}

/*
	Mat img_1 = imread("./H/frame533.jpg",0);
	Mat img_2 = imread("./H/frame553.jpg",0);
	
	detect_obstacle(img_1,img_2);*/


	return 1;
}
void detect_obstacle(Mat img_1, Mat img_2){
	vector<KeyPoint> keypoint1, keypoint2;
	vector<DMatch> filter_matches;
	int size_diff = 5;
	float sensitivity = 3.0;
	Mat img_matches;

	find_keypoint(img_1,keypoint1, img_2, keypoint2);
	find_matches(img_1,img_2,keypoint1,keypoint2, size_diff, sensitivity, filter_matches);


// template matching to find obstacles
	vector<bool> obstacles;
	obstacles = template_matching(img_1, img_2, keypoint1, keypoint2, filter_matches);
	for(int i=0; i < obstacles.size(); i++) {
		if (obstacles[i] == true) {
			printf("obstacle index: %d\n", i);
		}
	}

	vector<KeyPoint> obstacle_keypoint;
	for(int i=0; i < filter_matches.size(); i++) {
		if (obstacles[i] == true) {
			obstacle_keypoint.push_back(keypoint2[filter_matches[i].trainIdx]);
		}
	}

	
	Mat img_obstacle;
    drawKeypoints(img_2, obstacle_keypoint, img_obstacle, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("obstacles in train image", img_obstacle);

	drawMatches( img_1, keypoint1, img_2, keypoint2,
			 filter_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	imshow("Good Matches", img_matches);


	waitKey(0);
}

void find_matches(Mat img_1, Mat img_2,vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2, int size_diff, float sensitivity, vector<DMatch> &filter_matches){
	Mat descriptors_1, descriptors_2;
	Ptr<SURF> extractor = SURF::create();

    extractor-> compute(img_1, keypoint1, descriptors_1);
    extractor -> compute(img_2, keypoint2, descriptors_2);

    vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	matcher->match(descriptors_1, descriptors_2, matches);


	double max_dist = 0; double min_dist = 100;

	for(int i = 0; i < descriptors_1.rows; i++){
		double dist = matches[i].distance;
		if( dist < min_dist) min_dist = dist;
		if( dist > max_dist) max_dist = dist;
	}

/*	printf("--Max dist : %f \n", max_dist);
	printf("--Min dist : %f \n", min_dist);*/

	vector< DMatch > good_matches;
	for(int i = 0; i < descriptors_1.rows; i++){
		if(matches[i].distance <= max(sensitivity*min_dist, 0.02)){
			good_matches.push_back(matches[i]);
		}
	}

	filter_matches = matches_larger(good_matches, size_diff, keypoint1, keypoint2);
}

void find_keypoint(Mat img1, vector<KeyPoint> &keypoint1, Mat img2, vector<KeyPoint> &keypoint2){
	int minHessian = 400;
	vector<KeyPoint> tmp_keypoint1, tmp_keypoint2;
	int x = 200;
	int y = 100;
	int w = 250;
	int h = 150;

	Ptr<SURF> detector = SURF::create(minHessian);
	
    detector -> detect(img1, tmp_keypoint1);
    detector -> detect(img2, tmp_keypoint2);

/*    rectangle(img1, Point(200,100), Point(450,250), Scalar(0,0,255), 1,8,0);
    imshow("ROI",img1);
    waitKey(0);*/

    keypoint1 = keypoint_ROI(tmp_keypoint1,x,y,w,h);
    keypoint2 = keypoint_ROI(tmp_keypoint2,x,y,w,h);
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

vector<DMatch> matches_larger(vector<DMatch> good_matches, int size_diff, vector<KeyPoint> kp1, vector<KeyPoint> kp2){
	vector<DMatch> filter_matches;
	cout << "previous size:" << good_matches.size() << endl;
	for(int i = 0; i < good_matches.size(); i++){
		float size1 = kp1[good_matches[i].queryIdx].size;
		float size2 = kp2[good_matches[i].trainIdx].size;
/*		cout << "size1:" << size1 << endl;
		cout << "size2:" << size2 << endl;
		cout << "size1 - size2:" << size1-size2 << endl;
		cout << "----------------" << endl;*/
		if(abs(size2 - size1) > size_diff){
			filter_matches.push_back(good_matches[i]);
		}
	}
	cout << "later size:" << filter_matches.size() << endl;
	return filter_matches;
}

vector<bool> template_matching(Mat img_1, Mat img_2, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2, vector<DMatch> matches) {
    
    vector<bool> obstacles(matches.size(), false);
	int image_width, image_height;
	image_width =  img_1.cols;
	image_height = img_1.rows;
	// printf("image size (height, width): (%d, %d)\n", image_height, image_width);
	int num_matches = matches.size();
	int fixed_size = 0;

    for(int j=0; j < matches.size(); j++) {
        // create template image from previous image (img_1)
		Point2f kpt_coord_1;
		float size;
        kpt_coord_1 = keypoint_1[matches[j].queryIdx].pt;
        // size = keypoint_1[matches[j].queryIdx].size;
		size = 20;

		// printf("keypoint crop origin coord: (%d, %d)\n", int(kpt_coord_1.x-size), int(kpt_coord_1.y-size));
		/*if (kpt_coord_1.x-size < 0 || kpt_coord_1.y-size < 0 || kpt_coord_1.x+size > image_width || kpt_coord_1.y+size > image_height) {
			// use fixed size
			size = 10;
			fixed_size ++;
		}*/

		int x = 0, y = 0;
		x = kpt_coord_1.x - size;
		y = kpt_coord_1.y - size;

		// printf("keypoint crop bottom right coord: (%d, %d)\n", int(x+size*2), int(y+size*2));
        Rect template_1_ROI(x, y, size*2, size*2);
		Mat template_1 = img_1(template_1_ROI);

        Mat min_diff;
        Mat one_diff;
        float best_scale;

        for(float scale=1; scale <= 1.7; scale += 0.1) {
            // create template image from current image (img_2)
			Point2f kpt_coord_2;
			float scaled_size;
            kpt_coord_2 = keypoint_2[matches[j].trainIdx].pt;
            scaled_size = size * scale;

			// x = kpt_coord_2.x-scaled_size > 0 ? kpt_coord_2.x-scaled_size : 0;
			// y = kpt_coord_2.y-scaled_size > 0 ? kpt_coord_2.y-scaled_size : 0;
			x = kpt_coord_2.x - scaled_size;
			y = kpt_coord_2.y - scaled_size;

            Rect template_2_ROI(x, y, scaled_size*2, scaled_size*2);
			Mat template_2 = img_2(template_2_ROI);

            // resize template_2 to the size as template_1
            Mat resized_template_2;
            resize(template_2, resized_template_2, Size(template_1.cols, template_1.rows), 0, 0, CV_INTER_AREA);
			// pyrDown(template_2, resized_template_2, Size(template_1.cols, template_1.rows));

            // compute the difference of templates
            Mat diff;
            absdiff(template_1, resized_template_2, diff);
			//diff = diff / pow(scale, 2);

            if (scale == 1) {
                one_diff = diff;
                min_diff = diff;
                best_scale = scale;
            }
            else {
                if ((sum(diff)[0]) < (sum(min_diff)[0])) {
                    min_diff = diff;
                    best_scale = scale;
                }
            }
        }

/*        bool satisfied = true;
        int threshold = 100;
        Mat diff_thres;

        absdiff(min_diff, one_diff, diff_thres);

        for(int m=0; m < min_diff.rows; m++) {
            for(int n=0; n < min_diff.cols; n++) {
                if ( int(diff_thres.at<unsigned char>(m, n)) * 0.2 > threshold) {
                    satisfied = false;
                    cout << "m:" << m << ", n:" << n << endl;
                    cout << "diff:" << int(diff_thres.at<unsigned char>(m, n)) << endl;
                    cout << "min_diff:" << min_diff.at<float>(m, n) << endl;
                    cout << "one_diff:" << one_diff.at<float>(m, n) << endl;

                }
            }
        }
        Mat result;
        divide(min_diff,one_diff, result);
        cout << "result:" << endl;
        cout << result << endl;
        cout << "-----" << endl;

        if ((best_scale > 1.2) && satisfied) {
            // it's an obstacle
            obstacles[j] = true;
		}*/
		if (best_scale > 1.5) {
            // it's an obstacle
            obstacles[j] = true;
		}
    }

	// printf("number of fixed size: %d/%d\n", fixed_size, num_matches);

    return obstacles;
}