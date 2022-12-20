#ifndef __TRT_ENGINE_H
#define __TRT_ENGINE_H

#include <NvInfer.h> 
#include <../samples/common/logger.h> 

using namespace nvinfer1;
using namespace sample;


#define IN_NAME "input"
#define OUT_NAME "output"
#define IN_H 256
#define IN_W 256
#define BATCH_SIZE 1
#define EXPLICIT_BATCH 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)



void CHECK(int status);
void doInference(IExecutionContext& context, float* input, float* output);


#endif