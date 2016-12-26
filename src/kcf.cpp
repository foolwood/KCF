/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/

#include "kcf.hpp"

KCF::KCF(std::string kernel_type, std::string feature_type) {
  padding_ = 1.5;
  lambda_ = 1e-4;
  output_sigma_factor_ = 0.1;

  if (strcmp(feature_type.c_str(), "gray") == 0) {
    interp_factor_ = 0.075;

    kernel_sigma_ = 0.2;

    kernel_poly_a_ = 1;
    kernel_poly_b_ = 7;

    features_gray_ = true;
    cell_size_ = 1;
  } else if (strcmp(feature_type.c_str(), "hog") == 0) {
    interp_factor_ = 0.02;

    kernel_sigma_ = 0.5;

    kernel_poly_a_ = 1;
    kernel_poly_b_ = 9;

    features_hog_ = true;
    features_hog_orientations_ = 9;
    cell_size_ = 4;
  }

  kernel_type_ = kernel_type;
}

void KCF::Init(cv::Mat image, cv::Rect rect_init) {
  result_rect_ = rect_init;

  pos_ = cv::Point(rect_init.x+ cvFloor((float)(rect_init.width)/2.),
		           rect_init.y+ cvFloor((float)(rect_init.height)/2.));
  target_sz_ = rect_init.size();

  resize_image_ = std::sqrt(target_sz_.area()) >= 100;
  if (resize_image_) {
    pos_ = FloorPointScale(pos_, 1./2);
    target_sz_ = FloorSizeScale(target_sz_, 1./2);
  }

  window_sz_ = FloorSizeScale(target_sz_, 1 + padding_);

  float output_sigma = std::sqrt(float(target_sz_.area())) * output_sigma_factor_ / cell_size_;

  cv::dft(GaussianShapedLabels(output_sigma, FloorSizeScale(window_sz_, 1. / cell_size_)),
		  yf_, DFT_COMPLEX_OUTPUT);

  cos_window_ = CalculateHann(yf_.size());

  cv::Mat patch = GetSubwindow(image, pos_, window_sz_);

  Learn(patch, 1.);
}

cv::Rect KCF::Update(cv::Mat image) {
  cv::Mat patch = GetSubwindow(image, pos_, window_sz_);
  std::vector<cv::Mat> z = GetFeatures(patch);
  std::vector<cv::Mat> zf_vector(z.size());
  for (unsigned int i = 0; i < z.size(); ++i)
	cv::dft(z[i], zf_vector[i], DFT_COMPLEX_OUTPUT);

  cv::Mat kzf;
  if (strcmp(kernel_type_.c_str(), "gaussian") == 0)
	kzf = GaussianCorrelation(zf_vector, model_xf_);
  else if (strcmp(kernel_type_.c_str(), "polynomial") == 0)
	kzf = PolynomialCorrelation(zf_vector,model_xf_);
  else
	kzf = LinearCorrelation(zf_vector,model_xf_);

  cv::Mat response;
  cv::idft(ComplexMul(model_alphaf_, kzf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT

  cv::Point maxLoc;
  cv::minMaxLoc(response, NULL, NULL, NULL, &maxLoc);

  if ((maxLoc.x + 1) > (zf_vector[0].cols / 2))
	maxLoc.x = maxLoc.x - zf_vector[0].cols;
  if ((maxLoc.y + 1) > (zf_vector[0].rows / 2))
	maxLoc.y = maxLoc.y - zf_vector[0].rows;

  pos_.x += cell_size_*maxLoc.x;
  pos_.y += cell_size_*maxLoc.y;
  result_rect_.x += cell_size_*maxLoc.x;
  result_rect_.y += cell_size_*maxLoc.y;

  patch = GetSubwindow(image, pos_, window_sz_);
  Learn(patch, interp_factor_);

  return result_rect_;
}

void KCF::Learn(cv::Mat &patch, float lr) {
  std::vector<cv::Mat> x = GetFeatures(patch);

  std::vector<cv::Mat> xf(x.size());

  for (unsigned int i = 0; i < x.size(); i++)
	cv::dft(x[i], xf[i], DFT_COMPLEX_OUTPUT);

  cv::Mat kf;
  if (strcmp(kernel_type_.c_str(), "gaussian") == 0)
    kf = GaussianCorrelation(xf, xf);
  else if(strcmp(kernel_type_.c_str(), "polynomial") == 0)
    kf = PolynomialCorrelation(xf, xf);
  else
    kf = LinearCorrelation(xf, xf);

  cv::Mat alphaf = ComplexDiv(yf_, kf + cv::Scalar(lambda_, 0));

  if (lr > 0.99) {
	model_alphaf_ = alphaf;
	model_xf_.clear();
	for (unsigned int i = 0; i < xf.size(); ++i)
      model_xf_.push_back(xf[i]);
  } else {
    model_alphaf_ = (1.0 - lr) * model_alphaf_ + lr * alphaf;
    for (unsigned int i = 0; i < xf.size(); ++i)
      model_xf_[i] = (1.0 - lr) * model_xf_[i] + lr * xf[i];
  }
}

cv::Mat KCF::CreateGaussian1(int n, float sigma) {
  cv::Mat kernel(n, 1, CV_32F);
  float* cf = kernel.ptr<float>();

  double sigmaX = sigma > 0 ? sigma : ((n - 1)*0.5 - 1)*0.3 + 0.8;
  double scale2X = -0.5 / (sigmaX*sigmaX);

  for (int i = 0; i < n; ++i) {
    double x = i - floor(n / 2) + 1;
    double t = std::exp(scale2X * x * x);
    cf[i] = (float)t;
  }

  return kernel;
}

cv::Mat KCF::CreateGaussian2(cv::Size sz, float sigma) {
  cv::Mat a = CreateGaussian1(sz.height, sigma);
  cv::Mat b = CreateGaussian1(sz.width, sigma);
  return a*b.t();
}

void CircShift(cv::Mat &x,cv::Size k) {
  int cx, cy;
  if (k.width < 0)
    cx = -k.width;
  else
    cx = x.cols - k.width;

  if (k.height < 0)
    cy = -k.height;
  else
    cy = x.rows - k.height;

  cv::Mat q0(x, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(x, cv::Rect(cx, 0, x.cols - cx, cy));  // Top-Right
  cv::Mat q2(x, cv::Rect(0, cy, cx, x.rows -cy));  // Bottom-Left
  cv::Mat q3(x, cv::Rect(cx, cy, x.cols -cx, x.rows-cy)); // Bottom-Right

  cv::Mat tmp1, tmp2;                           // swap quadrants (Top-Left with Bottom-Right)
  cv::hconcat(q3, q2, tmp1);
  cv::hconcat(q1, q0, tmp2);
  cv::vconcat(tmp1, tmp2, x);

}

cv::Mat KCF::GaussianShapedLabels(float sigma, cv::Size sz) {
  cv::Mat labels = CreateGaussian2(sz, sigma);
  cv::Size shift_temp = cv::Size(-cvFloor(sz.width * (1./2)), -cvFloor(sz.height * (1./2)));
  shift_temp.width += 1;
  shift_temp.height += 1;
  CircShift(labels, shift_temp);
  return labels;
}

cv::Mat KCF::CalculateHann(cv::Size sz) {
  cv::Mat temp1(Size(sz.width, 1), CV_32FC1);
  cv::Mat temp2(Size(sz.height, 1), CV_32FC1);
  for (int i = 0; i < sz.width; ++i)
	temp1.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.width - 1)));
  for (int i = 0; i < sz.height; ++i)
	temp2.at<float>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.height - 1)));
  return temp2.t()*temp1;
}

cv::Mat KCF::GetSubwindow(const cv::Mat &frame, cv::Point centerCoor, cv::Size sz) {
  cv::Mat subWindow;
  cv::Point lefttop(min(frame.cols-2, max(-sz.width+1, centerCoor.x - cvFloor(float(sz.width) / 2.0) + 1)),
		  	  	  	min(frame.rows-2, max(-sz.height+1, centerCoor.y - cvFloor(float(sz.height) / 2.0) + 1)));
  cv::Point rightbottom(lefttop.x + sz.width, lefttop.y + sz.height);

  cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
		  max(rightbottom.x - frame.cols+1, 0), max(rightbottom.y - frame.rows+1, 0));
  cv::Point lefttopLimit(max(lefttop.x, 0), max(lefttop.y, 0));
  cv::Point rightbottomLimit(min(rightbottom.x, frame.cols - 1), min(rightbottom.y, frame.rows - 1));

  cv::Rect roiRect(lefttopLimit, rightbottomLimit);

  frame(roiRect).copyTo(subWindow);

  if (border != Rect(0,0,0,0))
	cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
  return subWindow;
}

std::vector<cv::Mat> KCF::GetFeatures(cv::Mat patch) {
  cv::Mat x;
  std::vector<Mat> x_vector;
  if (features_hog_) {
    if (patch.channels() == 3)
      cv::cvtColor(patch, patch, CV_BGR2GRAY);
    patch.convertTo(patch, CV_32FC1);

    x_vector = f_hog_.extract(patch);

    for (unsigned int i = 0; i < x_vector.size(); ++i)
      x_vector[i] = x_vector[i].mul(cos_window_);
  }

  if (features_gray_) {
    if(patch.channels() == 3)
      cv::cvtColor(patch, patch, CV_BGR2GRAY);
    patch.convertTo(x, CV_32FC1, 1.0 / 255);
    x = x - cv::mean(x).val[0];
    x = x.mul(cos_window_);
    x_vector.push_back(x);
  }

  return x_vector;
}

cv::Mat KCF::GaussianCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf) {
  int N = xf[0].size().area();
  double xx = 0, yy = 0;

  std::vector<cv::Mat> xyf_vector(xf.size());
  cv::Mat xy(xf[0].size(), CV_32FC1, Scalar(0.0)), xyf, xy_temp;
  for (unsigned int i = 0; i < xf.size(); ++i) {
    xx += cv::norm(xf[i]) * cv::norm(xf[i]) / N;
    yy += cv::norm(yf[i]) * cv::norm(yf[i]) / N;
    cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
    cv::idft(xyf, xy_temp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
    xy += xy_temp;
  }
  float numel_xf = N * xf.size();
  cv::Mat k, kf;
  exp((-1 / (kernel_sigma_ * kernel_sigma_)) * max(0.0, (xx + yy - 2 * xy) / numel_xf), k);
  k.convertTo(k, CV_32FC1);
  dft(k, kf, DFT_COMPLEX_OUTPUT);
  return kf;
}

cv::Mat KCF::PolynomialCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf) {
  std::vector<cv::Mat> xyf_vector(xf.size());
  cv::Mat xy(xf[0].size(), CV_32FC1, Scalar(0)), xyf, xy_temp;
  for (unsigned int i = 0; i < xf.size(); ++i) {
    cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
    cv::idft(xyf, xy_temp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
    xy += xy_temp;
  }
  float numel_xf = xf[0].size().area() * xf.size();
  cv::Mat k, kf;
  cv::pow(xy / numel_xf + kernel_poly_a_, kernel_poly_b_, k);
  k.convertTo(k, CV_32FC1);
  cv::dft(k, kf, DFT_COMPLEX_OUTPUT);
  return kf;
}

cv::Mat KCF::LinearCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf) {
  cv::Mat kf(xf[0].size(), CV_32FC2, cv::Scalar(0)), xyf;
  for (unsigned int i = 0; i < xf.size(); ++i) {
    cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
    kf += xyf;
  }
  float numel_xf = xf[0].size().area() * xf.size();
  return kf/numel_xf;
}

cv::Mat KCF::ComplexMul(const cv::Mat &x1, const cv::Mat &x2) {
  std::vector<cv::Mat> planes1;
  cv::split(x1, planes1);
  std::vector<cv::Mat> planes2;
  cv::split(x2, planes2);
  std::vector<cv::Mat>complex(2);
  complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
  complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
  Mat result;
  cv::merge(complex, result);
  return result;
}

cv::Mat KCF::ComplexDiv(const cv::Mat &x1, const cv::Mat &x2) {
  std::vector<cv::Mat> planes1;
  cv::split(x1, planes1);
  std::vector<cv::Mat> planes2;
  cv::split(x2, planes2);
  std::vector<cv::Mat>complex(2);
  cv::Mat cc = planes2[0].mul(planes2[0]);
  cv::Mat dd = planes2[1].mul(planes2[1]);

  complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
  complex[1] = (-planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0])) / (cc + dd);
  cv::Mat result;
  cv::merge(complex, result);
  return result;
}


