#ifndef _UTILS_H
#define _UTILS_H

#include <io.h>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include <chrono>
#include <opencv.hpp>
#include <trt_engine.h>


using namespace std;
using namespace chrono;
using namespace cv;


void getAllFiles(string path, vector<string>& files, string format);
double get_microsec();
void mat2array(float* data, Mat img, int img_h, int img_w);
int load_img(string img_path, Mat& dst);



#endif