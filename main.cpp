#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "kcf.h"
#include "benchmark_info.h"
#include "feature.h"

using namespace std;
using namespace cv;

double tracker(string video_path, vector<String>img_files, Point pos, Size target_sz, vector<Point>&positions,
  KCFKernel kernel, KCFFeature features, double padding, double lambda, double output_sigma_factor,
  double interp_factor, int cell_size, bool show_visualization);

string benchmark_path = "E:/100Benchmark/";
string video_name = "Deer";
string video_path = benchmark_path + video_name +"/";
vector<Rect> groundtruth_rect;
vector<String>img_files;

int main(){
  bool show_visualization = true;
  string kernel_type = "gaussian";
  string feature_type = "hog";
  
  KCFKernel kernel;
  kernel.type = kernel_type;

  KCFFeature features;
  features.gray = false;
  features.hog = false;

	double padding = 1.5;
  double lambda = 1e-4;
	double output_sigma_factor = 0.1;

  double sigma, interp_factor;
  int cell_size;
  if (feature_type == "gray")
  {
    interp_factor = 0.075;
    kernel.sigma = 0.2;
    kernel.poly_a = 1;
    kernel.poly_b = 7;
    features.gray = true;
    cell_size = 1;
  }
  else if (feature_type == "hog"){
    interp_factor = 0.02;
    kernel.sigma = 0.5;
    kernel.poly_a = 1;
    kernel.poly_b = 9;
    features.hog = true;
    features.hog_orientations = 9;
    cell_size = 4;
  }
  else{
    cout << "Unknown feature." << endl;
    return -1;
  }

  if (!strcmp(kernel_type.c_str(), "linear") && !strcmp(kernel_type.c_str(), "polynomial") && !strcmp(kernel_type.c_str(), "gaussian")){
    cout << "Unknown kernel." << endl; 
    return -1;
  }

  if (load_video_info(video_path, groundtruth_rect, img_files) != 1)
    return -1;
  groundtruth_rect[0].x -= 1;     //cpp is zero based
  groundtruth_rect[0].y -= 1;
  Point pos = centerRect(groundtruth_rect[0]);
  Size target_sz(groundtruth_rect[0].width, groundtruth_rect[0].height);
  vector<Point>positions;
  double time = tracker(video_path, img_files, pos, target_sz, positions, kernel, features,
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization);
  
  waitKey();
	return 0;
}



double tracker(string video_path, vector<String>img_files, Point pos, Size target_sz, vector<Point>&positions,
  KCFKernel kernel, KCFFeature features, double padding, double lambda, double output_sigma_factor,
  double interp_factor, int cell_size, bool show_visualization)
{
  bool resize_image = false;
  if (std::sqrt(target_sz.area()) >= 1000){
    pos.x = cvFloor(double(pos.x) / 2);
    pos.y = cvFloor(double(pos.y) / 2);
    target_sz.width = cvFloor(double(target_sz.width) / 2);
    target_sz.height = cvFloor(double(target_sz.height) / 2);
    resize_image = true;
  }

  Size window_sz = scale_size(target_sz, (1.0 + padding));


  double output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor/double(cell_size);

  //Mat yf;
  //dft(GaussianShapedLabels(output_sigma, scale_size(window_sz, 1. / cell_size)), yf, DFT_COMPLEX_OUTPUT);
  //Mat cos_window(yf.size(), CV_64FC1);
  //CalculateHann(cos_window, yf.size());
  Size response_size = window_sz;
  if (features.hog){
    response_size.width = cvRound((double)window_sz.width / (double)cell_size) - 2;
    response_size.height = cvRound((double)window_sz.height / (double)cell_size) - 2;
  }
  Mat yf;
  dft(GaussianShapedLabels(output_sigma, response_size), yf, DFT_COMPLEX_OUTPUT);
  Mat cos_window(yf.size(), CV_64FC1);
  CalculateHann(cos_window, response_size);


  

  Mat im;
  Mat im_gray;
  Mat alphaf, modal_alphaf;
  Mat patch;
  Mat k, kf, kzf;
  Mat response;
  vector<Mat> modal_xf,zf;
  double time = 0;
  int64 tic, toc;
  for (int frame = 0; frame < img_files.size(); ++frame)
  {

    im = imread(img_files[frame], IMREAD_COLOR);
    im_gray = imread(img_files[frame], IMREAD_GRAYSCALE);
    if (resize_image){
      resize(im, im, im.size() / 2, 0, 0, INTER_CUBIC);
      resize(im_gray, im_gray, im.size() / 2, 0, 0, INTER_CUBIC);
    }

    tic = getTickCount();
    
    if (frame > 0)
    {
      GetSubwindow(im_gray, patch, pos, window_sz);
      vector<Mat> z = get_features(patch, "hog", cell_size, cos_window);
      vector<Mat> zf_vector(z.size());
      for (int i = 0; i < z.size(); i++)
      {
        dft(z[i], zf_vector[i], DFT_COMPLEX_OUTPUT);
      }

      if (kernel.type == "gaussian"){
        kzf = GaussianCorrelation(zf_vector, modal_xf, kernel.sigma);
      }
      else if (kernel.type == "polynomial"){
        kzf = GaussianCorrelation(zf_vector, modal_xf, kernel.sigma);
      }
      else if (kernel.type == "linear"){
        kzf = GaussianCorrelation(zf_vector, modal_xf, kernel.sigma);
      }

      cv::idft(ComplexMul(modal_alphaf, kzf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
      Point maxLoc;
      minMaxLoc(response, NULL, NULL, NULL, &maxLoc);

      if ((maxLoc.x + 1) > (zf_vector[0].cols / 2))
        maxLoc.x = maxLoc.x - zf_vector[0].cols;

      if ((maxLoc.y + 1) > (zf_vector[0].rows / 2))
        maxLoc.y = maxLoc.y - zf_vector[0].rows;

      pos.x = pos.x + cell_size*maxLoc.x;
      pos.y = pos.y + cell_size*maxLoc.y;
    }


    //get subwindow at current estimated target position, to train classifer
    GetSubwindow(im_gray, patch, pos, window_sz);
    vector<Mat> x = get_features(patch, "hog", cell_size, cos_window);
    vector<Mat> xf_vector(x.size());
    for (int i = 0; i < x.size(); i++)
    {
      dft(x[i], xf_vector[i], DFT_COMPLEX_OUTPUT);
    }
    if (kernel.type == "gaussian"){
      kf = GaussianCorrelation(xf_vector, xf_vector, kernel.sigma);
    }
    else if (kernel.type == "polynomial"){
      kf = GaussianCorrelation(xf_vector, xf_vector, kernel.sigma);
    }
    else if (kernel.type == "linear"){
      kf = GaussianCorrelation(xf_vector, xf_vector, kernel.sigma);
    }

    alphaf = ComplexDiv(yf, kf + Scalar(lambda, 0));

    if (frame == 0)
    {
      modal_alphaf = alphaf;
      modal_xf = xf_vector;
    }
    else
    {
      modal_alphaf = (1.0 - interp_factor) * modal_alphaf + interp_factor * alphaf;
      for (int i = 0; i < modal_xf.size(); i++)
      {
        modal_xf[i] = (1.0 - interp_factor) * modal_xf[i] + interp_factor * xf_vector[i];
      }
    }
    toc = getTickCount() - tic;
    time += toc;
    Rect rect_position(pos.x - target_sz.width / 2, pos.y - target_sz.height / 2, target_sz.width, target_sz.height);
    rectangle(im, rect_position, Scalar(0, 255, 0), 2);
    putText(im, to_string(frame), Point(20, 40), 6, 1, Scalar(0, 255, 255), 2);
    imshow(video_name, im);
    char key = waitKey(1);
    if (key == 27)
      break;

  }
  time = time / getTickFrequency();
  std::cout << "Time: " << time << "    fps:" << img_files.size() / time << endl;
  return time;
}