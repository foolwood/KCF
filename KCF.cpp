//
//  KCF.cpp
//  KCF
//
//  Created by 王强 on 16/7/4.
//  Copyright © 2016年 王强. All rights reserved.
//

#include "KCF.hpp"


cv::Mat getGaussian(int n, float sigma, int ktype = CV_32F)
{
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        { 1.f },
        { 0.25f, 0.5f, 0.25f },
        { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
        { 0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f }
    };
    
    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
    small_gaussian_tab[n >> 1] : 0;
    
    CV_Assert(ktype == CV_32F || ktype == CV_64F);
    Mat kernel(n, 1, ktype);
    float* cf = kernel.ptr<float>();
    double* cd = kernel.ptr<double>();
    
    double sigmaX = sigma > 0 ? sigma : ((n - 1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX*sigmaX);
    double sum = 0;
    
    int i;
    for (i = 0; i < n; i++)
    {
        double x = i - floor(n / 2);
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
        if (ktype == CV_32F)
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }
    
    return kernel;
}

cv::Mat gaussian_shaped_labels(float sigma, Size sz, int ktype)
{
    Mat a = getGaussian(sz.height, sigma, ktype);
    Mat b = getGaussian(sz.width, sigma, ktype);
    Mat labels = a*b.t();
    
    int cxFloor = cvFloor(float(labels.cols)/2.0);
    int cyFloor = cvFloor(float(labels.rows)/2.0);
    int cxCeil = cvCeil(float(labels.cols)/2.0);
    int cyCeil = cvCeil(float(labels.rows)/2.0);
    Mat temp1,temp2;
    Mat q0(labels, Rect(0, 0, cxFloor, cyFloor));
    Mat q1(labels, Rect(cxFloor, 0, cxCeil, cyFloor));
    Mat q2(labels, Rect(0, cyFloor, cxFloor, cyCeil));
    Mat q3(labels, Rect(cxFloor, cyFloor, cxCeil, cyCeil));
    
    cv::hconcat(q3, q2, temp1);
    cv::hconcat(q1, q0, temp2);
    cv::vconcat(temp1, temp2, labels);

    return labels;
}

cv::Mat fft(Mat x)
{
    Mat planes[] = { Mat_<float>(x), Mat::zeros(x.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    return complexI;
}

vector<cv::Mat> nchannelsfft(Mat x)
{
    vector<Mat>plane,result;
    split(x,plane);
    for (int i = 0 ; i<plane.size(); i++) {
//        Mat planes[] = { Mat_<float>(plane[i]), Mat::zeros(x.size(), CV_32F) };
//        Mat complexI;
//        merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//        dft(complexI, complexI);            // this way the result may fit in the source matrix
//        cout<<complexI<<endl;
//        result.push_back(complexI);
        result.push_back(fft(plane[i]));
    }
    
    return result;
}


cv::Mat calculateHann(Size sz){
    Mat temp1(Size(sz.width, 1), CV_64FC1);
    Mat temp2(Size(sz.height, 1), CV_64FC1);
    for (int i = 0; i < sz.width; ++i)
        temp1.at<double>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.width - 1)));
    for (int i = 0; i < sz.height; ++i)
        temp2.at<double>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.height - 1)));
    return temp2.t()*temp1;
}


cv::Mat get_subwindow(Mat &frame, Point pos, Size window_sz){
	Point lefttop(cvFloor(pos.x) +1- cvFloor(window_sz.width / 2), cvFloor(pos.y) +1- cvFloor(window_sz.height / 2));
	Point rightbottom(lefttop.x + window_sz.width, lefttop.y + window_sz.height);
    
	Rect roi(lefttop, rightbottom);
	Rect region = roi;

	if (roi.x<0){ region.x = 0; region.width += roi.x; }
	if (roi.y<0){ region.y = 0; region.height += roi.y; }
	if (roi.x + roi.width>frame.cols)region.width = frame.cols - roi.x;
	if (roi.y + roi.height>frame.rows)region.height = frame.rows - roi.y;
	if (region.width>frame.cols)region.width = frame.cols;
	if (region.height>frame.rows)region.height = frame.rows;

	Mat patch = frame(region).clone();

	int addTop, addBottom, addLeft, addRight;
	addTop = region.y - roi.y;
	addBottom = (roi.height + roi.y>frame.rows ? roi.height + roi.y - frame.rows : 0);
	addLeft = region.x - roi.x;
	addRight = (roi.width + roi.x>frame.cols ? roi.width + roi.x - frame.cols : 0);

	if (addTop != 0 && addBottom != 0 && addLeft != 0 && addRight!=0)
		copyMakeBorder(patch, patch, addTop, addBottom, addLeft, addRight, cv::BORDER_CONSTANT);
   
	return patch;
}


cv::Mat gaussian_correlation(Mat &xf, Mat &yf, float sigma){

    float N = xf.cols*xf.rows;
    
    float xx = norm(xf)*norm(xf)/ N;  //squared norm of x
    float yy = norm(yf)*norm(yf)/ N;  //squared norm of y
    
    //cross-correlation term in Fourier domain
    Mat yf_conj;
    vector<Mat> planes;
    split(yf, planes);
    planes[1] = planes[1] * -1;
    merge(planes, yf_conj);
    Mat xyf = complexMul(xf, yf_conj);
    
    Mat xy;
    cv::idft(xyf, xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
    double numelx1 = xf.cols*xf.rows;
    Mat k;
    exp(-1 / (sigma*sigma) * max(0, (xx + yy - 2 * xy) / numelx1),k);
    return fft(k);
}

cv::Mat gaussian_correlation(vector<Mat> &xf,vector<Mat> &yf, float sigma){
    
    float N = xf[0].cols*xf[0].rows;
    
    float xx = 0,yy = 0;
    for (int i = 0; i<xf.size(); i++) {
        xx += norm(xf[i])*norm(xf[i])/ N;  //squared norm of x
        yy += norm(yf[i])*norm(yf[i])/ N;  //squared norm of y
    }
    cout<<xx<<endl;
    
    
    //cross-correlation term in Fourier domain
    Mat xy;
    
    for (int i = 0; i<xf.size(); i++) {
        Mat yf_conj;
        vector<Mat> planes;
        split(yf[i], planes);
        planes[1] = planes[1] * -1;
        merge(planes, yf_conj);
        Mat xyf = complexMul(xf[i], yf_conj);
        Mat temp;
        cv::idft(xyf, temp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
        if(i == 0)
            temp.copyTo(xy);
        else
            xy += temp;
        
    }
    
    double numelx1 = xf[0].cols*xf[0].rows*xf.size();
    Mat k;
    exp(-1 / (sigma*sigma) * max(0, (xx + yy - 2 * xy) / numelx1),k);
    return fft(k);
}




cv::Mat complexMul(Mat x1, Mat x2)
{
    vector<Mat> planes1;
    split(x1, planes1);
    vector<Mat> planes2;
    split(x2, planes2);
    vector<Mat>complex(2);
    complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
    complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
    Mat result;
    merge(complex, result);
    return result;
}

cv::Mat complexDiv(Mat x1, Mat x2)
{
    vector<Mat> planes1;
    split(x1, planes1);
    vector<Mat> planes2;
    split(x2, planes2);
    vector<Mat>complex(2);
    Mat cc = planes2[0].mul(planes2[0]);
    Mat dd = planes2[1].mul(planes2[1]);
    
    complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
    complex[1] = (planes1[0].mul(planes2[1]) - planes1[1].mul(planes2[0])) / (cc + dd);
    Mat result;
    merge(complex, result);
    return result;
}





























