#ifndef ARCFACE_INTERFACE_H
#define ARCFACE_INTERFACE_H
#include "net.h"
#include "base.h"
#include "arcface.h"
#include "mtcnn.h"
#include <vector>

void initMtcnn(ncnn::Net &pnet, ncnn::Net &rnet, ncnn::Net &onet, ncnn::Net &lnet);
void init_arcface(ncnn::Net &net);

vector<FaceInfo> face_detect(ncnn::Mat ncnn_img);
vector<float> face_exactfeature(ncnn::Mat img, FaceInfo info);

float face_calcSimilar(std::vector<float> feature1, std::vector<float> feature2);

#endif