#include "benchmark_info.h"

int load_video_info(string base_path, string video_name, vector<Rect> &groundtruthRect, vector<String> &fileName){
  string txt_path;
  if (strcmp(video_name.c_str(), "Jogging.1") == 0 || strcmp(video_name.c_str(), "Skating2.1") == 0)
  {
    video_name = video_name.substr(0, video_name.size() - 2);
    txt_path = base_path + video_name + "/groundtruth_rect.1.txt";
  }
  else  if (strcmp(video_name.c_str(), "Jogging.2") == 0 || strcmp(video_name.c_str(), "Skating2.2") == 0)
  {
    video_name = video_name.substr(0, video_name.size() - 2);
    txt_path = base_path + video_name + "/groundtruth_rect.2.txt";
  }
  else
    txt_path = base_path + video_name + "/groundtruth_rect.txt";
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
    groundtruthRect.push_back(Rect(--tmp1, --tmp2, tmp3, tmp4));
  }
  infile.close();


  string image_path = base_path + video_name + "/img/";

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

  if (video_name == "David")
  {
    fileName.erase(fileName.begin() + 770, fileName.end());
    fileName.erase(fileName.begin(), fileName.begin() + 300 - 1);
  }
  if (video_name == "Football1")
  {
    fileName.erase(fileName.begin() + 74, fileName.end());
  }
  if (video_name == "Freeman3")
  {
    fileName.erase(fileName.begin() + 460, fileName.end());
  }
  if (video_name == "Freeman4")
  {
    fileName.erase(fileName.begin() + 283, fileName.end());
  }

  return 1;
}

void getFiles(string path, vector<string>& files, vector<string>& names)
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
        if (strcmp(fileinfo.name, "Jogging") == 0 || strcmp(fileinfo.name, "Skating2") == 0)
        {
          string name_temp = fileinfo.name;
          names.push_back(name_temp + ".1");
          files.push_back(p.assign(path).append("\\").append(fileinfo.name));
          names.push_back(name_temp + ".2");
          files.push_back(p.assign(path).append("\\").append(fileinfo.name));
        }
        else
        {
          names.push_back(fileinfo.name);
          files.push_back(p.assign(path).append("\\").append(fileinfo.name));
        }

      }
    } while (_findnext(hFile, &fileinfo) == 0);
    _findclose(hFile);
  }
}