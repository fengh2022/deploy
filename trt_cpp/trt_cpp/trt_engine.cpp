#include <trt_engine.h>




void CHECK(int status)
{
	do
	{
		auto ret = (status);
		if (ret != 0)
		{
			std::cerr << "Cuda failure: " << ret << std::endl;
			abort();
		}
	} while (0);
}


//void doInference(IExecutionContext& context, float* input, float* output, void** buffers, int inputIndex, int outputIndex, cudaStream_t& stream)
//{
//	//const ICudaEngine& engine = context.getEngine();
//
//	//// Pointers to input and output device buffers to pass to engine. 
//	//// Engine requires exactly IEngine::getNbBindings() number of buffers. 
//	//assert(engine.getNbBindings() == 2);
//	//void* buffers[2];
//
//	//// In order to bind the buffers, we need to know the names of the input and output tensors. 
//	//// Note that indices are guaranteed to be less than IEngine::getNbBindings() 
//	//const int inputIndex = engine.getBindingIndex(IN_NAME);
//	//const int outputIndex = engine.getBindingIndex(OUT_NAME);
//
//	//// Create GPU buffers on device 
//	//CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));
//	//CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * 2 * sizeof(float)));
//
//	//// Create stream 
//	//cudaStream_t stream;
//	//CHECK(cudaStreamCreate(&stream));
//
//	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
//	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//	context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
//	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
//	cudaStreamSynchronize(stream);
//
//
//}



void doInference(IExecutionContext& context, float* input, float* output)
{
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine. 
	// Engine requires exactly IEngine::getNbBindings() number of buffers. 
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors. 
	// Note that indices are guaranteed to be less than IEngine::getNbBindings() 
	const int inputIndex = engine.getBindingIndex(IN_NAME);
	const int outputIndex = engine.getBindingIndex(OUT_NAME);

	// Create GPU buffers on device 
	CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * 3 * IN_H * IN_W / 4 * sizeof(float)));

	// Create stream 
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], BATCH_SIZE * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers 
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}
