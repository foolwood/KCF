#include "benchmark_info.h"

int load_video_info(string videoPath, vector<Rect> &groundtruthRect, vector<String> &fileName){
	
	string txt_path = videoPath + "/groundtruth_rect.txt";
	ifstream infile;
	infile.open(txt_path);
  if (!infile){ cout << "No groundtruth_rect.txt" << endl; return -1; }

	vector<Rect>result;
	string s;
	int tmp1, tmp2, tmp3, tmp4;
  

	while (getline(infile, s))
	{
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

  if (fileName.size() == 0){
    cout << "No image!!" << endl; return -1;
  }
	sort(fileName.begin(), fileName.end());

	return 1;
}

void getFiles(string path, vector<string>& files,vector<string>& names)
{

  long   hFile = 0;

  struct _finddata_t fileinfo;
  string p;
  if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
  {
    do
    {
      //if ((fileinfo.attrib &  _A_SUBDIR))
      //{
      //  if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
      //    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
      //}
      //else
      //{
      //  files.push_back(p.assign(path).append("\\").append(fileinfo.name));
      //}
      if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
        names.push_back(fileinfo.name);
        files.push_back(p.assign(path).append("\\").append(fileinfo.name));
      }
    } while (_findnext(hFile, &fileinfo) == 0);
    _findclose(hFile);
  }
}