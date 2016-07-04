#include "load_video_info.h"

int load_video_info(string videoPath, vector<Rect> &groundtruthRect, vector<String> &fileName){
	
	string txt_path = videoPath + "/groundtruth_rect.txt";
	ifstream infile;
	infile.open(txt_path);
	vector<Rect>result;
	string s;
	int tmp1, tmp2, tmp3, tmp4;

	while (getline(infile, s))
	{
		//cout << s << endl;
		replace(s.begin(), s.end(), ',', ' ');
		stringstream ss;
		ss.str(s);
		ss >> tmp1 >> tmp2 >> tmp3 >> tmp4;
		groundtruthRect.push_back(Rect(tmp1, tmp2, tmp3, tmp4));
	}
	infile.close();


	string image_path = videoPath + "/img/";
	
	vector<String> filenames;
	cv::glob(image_path, filenames);

	string format = "jpg";
	for (size_t i = 0; i < filenames.size(); i++)
	{
		string nameTemp = filenames[i];
		if (regex_match(nameTemp, std::regex(".*." + format)))
			fileName.push_back(filenames[i]);
	}

	format = "png";
	for (size_t i = 0; i < filenames.size(); i++)
	{
		string nameTemp = filenames[i];
		if (regex_match(nameTemp, std::regex(".*." + format)))
			fileName.push_back(filenames[i]);
	}

	sort(fileName.begin(), fileName.end());

	return 1;
}
