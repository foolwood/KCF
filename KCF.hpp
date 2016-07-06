//
//  KCF.hpp
//  KCF
//
//  Created by 王强 on 16/7/4.
//  Copyright © 2016年 王强. All rights reserved.
//

#ifndef KCF_hpp
#define KCF_hpp

#define PI 3.141592653589

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"



using namespace cv;
using namespace std;

static inline cv::Point centerRect(const cv::Rect& r)
{
    return cv::Point(r.x+cvFloor(float(r.width) / 2.0), r.y+cvFloor(float(r.height) / 2.0));
}

static inline cv::Rect scale_rect(const cv::Rect& r, float scale)
{
    cv::Point m=centerRect(r);
    float width  = r.width  * scale;
    float height = r.height * scale;
    int x=cvFloor(m.x - width/2.0);
    int y=cvFloor(m.y - height/2.0);
    
    return cv::Rect(x, y, cvFloor(width), cvFloor(height));
}

static inline cv::Size scale_size(const cv::Size& r, float scale)
{
    float width  = float(r.width)  * scale;
    float height = float(r.height) * scale;
    
    return cv::Size(cvFloor(width), cvFloor(height));
}

static inline cv::Size scale_sizexy(const cv::Size& r, float scalex,float scaley)
{
    float width  = float(r.width)  * scalex;
    float height = float(r.height) * scaley;
    
    return cv::Size(cvFloor(width), cvFloor(height));
}



cv::Mat gaussian_shaped_labels(float sigma, Size sz, int ktype = CV_32F);

cv::Mat fft(Mat x);

vector<cv::Mat> nchannelsfft(Mat x);

cv::Mat calculateHann(Size sz);

cv::Mat get_subwindow(Mat &frame, Point pos, Size window_sz);

cv::Mat gaussian_correlation(Mat &zf, Mat &model_xf, float sigma);

cv::Mat gaussian_correlation(vector<Mat> &xf,vector<Mat> &yf, float sigma);

cv::Mat complexMul(Mat x1, Mat x2);

cv::Mat complexDiv(Mat x1, Mat x2);






#endif /* KCF_hpp */
