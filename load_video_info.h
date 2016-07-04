#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


int load_video_info(string videoPath, vector<Rect> &groundtruthRect,vector<String> &fileName);