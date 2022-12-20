#include <utils.h>





void getAllFiles(string path, vector<string>& files, string format) {
	//�ļ����
	long long hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) {  //�Ƚ��ļ������Ƿ����ļ���
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					//�ݹ�����
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, format);
				}
			}
			else {
				string file_path = p.assign(path).append("\\").append(fileinfo.name);
				if (file_path.substr(file_path.size()-format.size(), format.size())==format)
					files.push_back(file_path);
			}
		} while (_findnext(hFile, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1
		_findclose(hFile);
	}
}


double get_microsec() {
	system_clock::time_point time_point_now = system_clock::now(); // ��ȡ��ǰʱ���
	system_clock::duration duration_since_epoch
		= time_point_now.time_since_epoch(); // ��1970-01-01 00:00:00����ǰʱ����ʱ��
	time_t microseconds_since_epoch
		= duration_cast<microseconds>(duration_since_epoch).count(); // ��ʱ��ת��Ϊ΢����
	time_t seconds_since_epoch = microseconds_since_epoch / 1000000; // ��ʱ��ת��Ϊ����
	std::tm current_time = *std::localtime(&seconds_since_epoch); // ��ȡ��ǰʱ�䣨��ȷ���룩
	time_t tm_microsec = microseconds_since_epoch % 1000; // ��ǰʱ���΢����
	time_t tm_millisec = microseconds_since_epoch / 1000 % 1000; // ��ǰʱ��ĺ�����

	string sec = to_string(microseconds_since_epoch / 1000 / 1000);
	string milli_sec = to_string(tm_millisec);
	string micro_sec = to_string(tm_microsec);

	string res = sec + "." + std::string(3 - milli_sec.size(), '0') + milli_sec +
		std::string(3 - micro_sec.size(), '0') + micro_sec;

	return atof(res.c_str());
}

void mat2array(float* data, Mat img, int img_h, int img_w)
{
	int index = 0;
	for (int i = 0; i < img_h; i++) {
		for (int j = 0; j < img_w; j++) {
			data[index] = img.at<Vec3b>(i, j)[0] / 255.0;
			data[index + img_h*img_w] = img.at<Vec3b>(i, j)[1] / 255.0;
			data[index + 2 * img_h*img_w] = img.at<Vec3b>(i, j)[2] / 255.0;
			index++;
		}
	}

}

bool str_exist(string a, char* b)
{
	return !strstr(a.c_str(), b) == NULL;
}

int load_img(string img_path, Mat& dst)
{
	Mat img = imread(img_path);
	if (img.data == NULL)
	{
		cout << "err" << endl;
		return -1;
	}
	resize(img, dst, Size(IN_W, IN_H), 0, 0, INTER_NEAREST);
	return str_exist(img_path, "Cat") ? 0 : 1;
}
