/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/
#include <iostream>
#include <numeric>      // std::accumulate
#include <string>
#include "opencv2/opencv.hpp"
#include "benchmark_info.h"
#include "kcf.h"

using namespace std;
using namespace cv;

int tracker(string base_path, string video_name, double &precision, double &fps);
vector<double>PrecisionCalculate(vector<Rect>groundtruth_rect, vector<Rect>result_rect);

int main(int argc, char** argv){
  string benchmark_path = "E:\\50Benchmark\\";
  vector<string>video_path_list, video_name_list;

  getFiles(benchmark_path, video_path_list, video_name_list);

  string mode = "simple video";
  if (argc == 2)
  {
    mode = argv[1];

  }

  if (mode == "all")
  {
    cout << ">> run_tracker('all')" << endl;
    vector<double>all_precision, all_fps;
    double precision, fps;
    for (int i = 0; i < video_name_list.size(); i++)
    {
      string video_name = video_name_list[i];
      tracker(benchmark_path, video_name, precision, fps);
      all_precision.push_back(precision);
      all_fps.push_back(fps);
    }
    double mean_precision = std::accumulate(all_precision.begin(), all_precision.end(), 0.0) / double(all_precision.size());
    double mean_fps = std::accumulate(all_fps.begin(), all_fps.end(), 0.0) / double(all_fps.size());
    printf("\nAverage precision (20px):%1.3f, Average FPS:%4.2f\n\n", mean_precision, mean_fps);
  }
  else{
    int choice = 0;
    for (int i = 0; i < video_name_list.size(); i++)
    {
      std::printf(" %02d  %12s\n", i, video_name_list[i]);
    }
    cout << "\n\nChoice One Video!!" << endl;
    cin >> choice;
    if (choice >= video_name_list.size() || choice < 0)
    {
      cout << "No such video" << endl;
      return 0;
    }
    string video_name = video_name_list[choice];
    double precision, fps;
    tracker(benchmark_path, video_name, precision, fps);
  }
  system("PAUSE");
  return 0;
}

int tracker(string base_path, string video_name, double &precision, double &fps){

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

vector<Rect> groundtruth_rect;
vector<String>img_files;
if (load_video_info(base_path, video_name, groundtruth_rect, img_files) != 1)
return -1;
Point pos = centerRect(groundtruth_rect[0]);
Size target_sz(groundtruth_rect[0].width, groundtruth_rect[0].height);


/****************************************************************
 *  tracker begin
 ***************************************************************/
bool resize_image = false;
if (std::sqrt(target_sz.area()) >= 1000){
  pos.x = cvFloor(double(pos.x) / 2);
  pos.y = cvFloor(double(pos.y) / 2);
  target_sz.width = cvFloor(double(target_sz.width) / 2);
  target_sz.height = cvFloor(double(target_sz.height) / 2);
  resize_image = true;
}

Size window_sz = scale_size(target_sz, (1.0 + padding));

double output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor / double(cell_size);

Mat yf;
dft(GaussianShapedLabels(output_sigma, scale_size(window_sz, 1. / cell_size)), yf, DFT_COMPLEX_OUTPUT);
Mat cos_window(yf.size(), CV_64FC1);
CalculateHann(cos_window, yf.size());
FHoG f_hog;


Mat im;
Mat im_gray;
Mat alphaf, modal_alphaf;
Mat patch;
Mat kf, kzf;
Mat response;
vector<Mat> modal_xf, zf;
double time = 0;
vector<Rect>result_rect;
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
    vector<Mat> z = get_features(patch, feature_type, cell_size, cos_window, f_hog);
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
  vector<Mat> x = get_features(patch, feature_type, cell_size, cos_window, f_hog);
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
  result_rect.push_back(rect_position);
  if (show_visualization){
    rectangle(im, rect_position, Scalar(0, 255, 0), 2);
    putText(im, to_string(frame), Point(20, 40), 6, 1, Scalar(0, 255, 255), 2);
    imshow(video_name, im);
    char key = waitKey(1);
    if (key == 27)
      break;
  }


}
time = time / double(getTickFrequency());
vector<double>precisions = PrecisionCalculate(groundtruth_rect, result_rect);
printf("%12s - Precision (20px):%1.3f, FPS:%4.2f\n", video_name, precisions[20], double(img_files.size()) / time);
destroyAllWindows();
precision = precisions[20];
fps = double(img_files.size()) / time;
/****************************************************************
*  tracker end
***************************************************************/
return 0;
}

vector<double>PrecisionCalculate(vector<Rect>groundtruth_rect, vector<Rect>result_rect){
  int max_threshold = 50;
  vector<double>precisions(max_threshold+1, 0);
  if (groundtruth_rect.size() != result_rect.size()){
    int n = min(groundtruth_rect.size(), result_rect.size());
    groundtruth_rect.erase(groundtruth_rect.begin() + n, groundtruth_rect.end());
    result_rect.erase(groundtruth_rect.begin() + n, groundtruth_rect.end());
  }
  vector<double>distances;
  for (int  i = 0; i < result_rect.size(); i++)
  {
    double distemp = sqrt(double(pow(result_rect[i].x + result_rect[i].width / 2 - groundtruth_rect[i].x - groundtruth_rect[i].width / 2, 2) +
      pow(result_rect[i].y + result_rect[i].height / 2 - groundtruth_rect[i].y - groundtruth_rect[i].height / 2, 2)));
    distances.push_back(distemp);
  }
  for (int i = 0; i <= max_threshold; i++)
  {
    for (int j = 0; j < distances.size(); j++)
    {
      if (distances[j] < double(i))
        precisions[i]++;

    }
    precisions[i] = precisions[i] / distances.size();
  }
  return precisions;
}