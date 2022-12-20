#include <fstream> 
#include <iostream> 
#include <trt_engine.h>

#include <string>

#include <utils.h>
#include <cstring>



using namespace std;
using namespace cv;



int main(int argc, char** argv)
{
	// create a model using the API directly and serialize it to a stream 
	char *trtModelStream{ nullptr };
	size_t size{ 0 };
	string engine_file = "D:/prj/me/deploy/catdog/ckpts/model.engine";
	string img_dir = "D:/Datasets/kagglecatsanddogs_5340/PetImages/test";
	vector<string> files;

	getAllFiles(img_dir, files, ".jpg");


	cout << files.size() << endl;


	std::ifstream file(engine_file, std::ios::binary);
	if (file.good()) {
		printf("OK");
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	else
		printf("False");

	Logger m_logger;
	IRuntime* runtime = createInferRuntime(m_logger);
	assert(runtime != nullptr);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);


	// Pointers to input and output device buffers to pass to engine. 
	// Engine requires exactly IEngine::getNbBindings() number of buffers. 
	assert(engine->getNbBindings() == 2);
	// generate input data 
	float data[BATCH_SIZE * 3 * IN_H * IN_W];
	// Run inference 
	float prob[BATCH_SIZE * 2];
	double tap, time=0.0;

	float correct = 0;
	int valid_cnt = 0, err_cnt=0;

	Mat img;
	for (int i = 0; i < files.size(); i++)
	{
		string file_path = files[i];
		int res = load_img(file_path, img);

		if (res == -1)
		{
			err_cnt++;
			continue;
		}

		mat2array(data, img, IN_H, IN_W);
		//cout << i << "/" << files.size() << endl;
		
		tap = get_microsec();
		doInference(*context, data, prob);
		double gap = get_microsec() - tap;
		//cout << gap << endl;
		time += gap;

		valid_cnt++;

		if (((prob[0] > prob[1]) && res == 0) || (prob[0] <= prob[1]) && res == 1)
			correct += 1;
	}

	printf("acc: %f, fps: %f", correct / files.size(), (float)valid_cnt/time );
	printf("total: %d, err: %d", files.size(), err_cnt);


	// Destroy the engine 
	context->destroy();
	engine->destroy();
	runtime->destroy();
	return 0;
}