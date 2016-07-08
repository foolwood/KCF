#include "opencv2/opencv.hpp"
#include <iostream>
#include "load_video_info.h"
#include "feature.h"
#include "KCF.hpp"

using namespace std;
using namespace cv;

//string benchmarkPath = "E:/benchmark50/";
string benchmarkPath = "/users/wangqiang/desktop/benchmark100/";
string videoName = "Boy";
string videoPath = benchmarkPath + videoName;
vector<Rect> groundtruthRect;
vector<String>fileName;



int main(){
    
    if (load_video_info(videoPath, groundtruthRect,fileName) != 1)
        return -1;
    float padding = 1.5;  //extra area surrounding the target
    float lambda = 1e-4;  //regularization
    float output_sigma_factor = 0.1;  //spatial bandwidth (proportional to target)
    float interp_factor = 0.02;
    
    float kernel_sigma = 0.5;
    int hog_orientations = 9;
    int cell_size = 4;
    groundtruthRect[0].x -=1;
    groundtruthRect[0].y -=1;
    Point pos = centerRect(groundtruthRect[0]);
    Size target_sz(groundtruthRect[0].width,groundtruthRect[0].height);
    Size window_sz = scale_size(target_sz,(1+padding));
    
    float output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor / double(cell_size);
    
    Mat yf = fft(gaussian_shaped_labels(output_sigma, scale_size(window_sz, 1./float(cell_size))));
    
    Mat cos_window = calculateHann(yf.size());
    
    Mat im,im_gray,patch;
    Mat kf,kzf,model_alphaf,alphaf,new_z;
    vector<Mat>xf,zf,model_xf;
    Mat response;
    
    for (int frame = 0; frame < fileName.size(); frame++) {
        im = imread(fileName[frame], IMREAD_COLOR);
        im_gray = imread(fileName[frame], IMREAD_GRAYSCALE);
        
        
        if(frame > 0){
            patch = get_subwindow(im, pos, window_sz);
            zf = nchannelsfft(get_features(patch, hog_orientations, cell_size, cos_window));
            kzf = gaussian_correlation(zf, model_xf, kernel_sigma);
            idft(complexMul(model_alphaf,kzf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            Point maxLoc;
            minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
            if(float(maxLoc.x) > float(zf[0].cols)/2.0)
                maxLoc.x = maxLoc.x -zf[0].cols;
            if(float(maxLoc.y) > float(zf[0].rows)/2.0)
                maxLoc.y = maxLoc.y -zf[0].rows;
            pos.x = pos.x + cell_size*(maxLoc.x);
            pos.y = pos.y + cell_size*(maxLoc.y);
        }
        
        patch = get_subwindow(im, pos, window_sz);
        xf = nchannelsfft(get_features(patch, hog_orientations, cell_size, cos_window));
        kf = gaussian_correlation(xf, xf, kernel_sigma);

        vector<Mat> planes;
        split(kf, planes);
        planes[0] = planes[0] + lambda;
        merge(planes, kf);
        alphaf = complexDiv(yf,kf);
        
        if(frame == 0) //first frame, train with a single image
        {
            model_alphaf = alphaf;
            model_xf = xf;
        }
        else
        {
            model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            for (int i = 0; i<xf.size(); i++) {
                model_xf[i] = (1 - interp_factor) * model_xf[i] + interp_factor * xf[i];
            }
            
        }

        Rect rect_position(pos.x - target_sz.width /2, pos.y - target_sz.height/2, target_sz.width, target_sz.height);
        rectangle(im, rect_position, Scalar(0, 255, 0), 2);
        putText(im, to_string(frame), Point(20, 40), 6, 1, Scalar(0, 255, 255), 2);
        imshow(videoName, im);
        char key = waitKey(1);
        if (key == 27)
            break;
        
    }
    

    return 0;
}
