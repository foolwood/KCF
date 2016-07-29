//
//  feature.h
//  opencv3
//
//  Created by 王强 on 16/7/4.
//  Copyright © 2016年 王强. All rights reserved.
//

#ifndef feature_h
#define feature_h
#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc.hpp"

#include "fhog.hpp"

#include <string>
#include <vector>

using namespace cv;
using namespace std;

static const int dimHOG = 31;

void computeHOG31D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y);
vector<cv::Mat> get_features(Mat patch, string features, int cell_size, Mat &cos_window, FHoG &f_hog);

#endif /* feature_h */
