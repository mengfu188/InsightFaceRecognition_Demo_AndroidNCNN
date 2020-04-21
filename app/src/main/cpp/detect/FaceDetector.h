//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

//#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include <vector>
#include "net.h"
#include <chrono>
#include <math.h>
#include <android/log.h>

using namespace std::chrono;
#define TAG "LightFaceSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

class Timer {
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic() {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true) {
        double diff = duration_cast<milliseconds>(
                high_resolution_clock::now() - tictoc_stack.top()).count();
        if (msg.size() > 0) {
            if (flag)
                LOGD("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }

    void reset() {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

struct Point {
    float _x;
    float _y;
};
struct bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box {
    float cx;
    float cy;
    float sx;
    float sy;
};

float *bbox2float(const bbox bbox_);
float *bboxs2float(std::vector<bbox> &bboxes_);

class Detector {

public:
    Detector();

    void Init(const std::string &model_param, const std::string &model_bin);

    void Init(const ncnn::Net &net);


    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

//    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

    void Detect(ncnn::Net &net, ncnn::Mat &bgr, std::vector<bbox> &boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    void set_threshold(float threshold);

    ~Detector();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;

    ncnn::Net *Net;

    void
    post_process(ncnn::Mat &out, ncnn::Mat &out1, ncnn::Mat &out2, std::vector<bbox> &boxes,
                 int src_cols, int src_rows,
                 int scale_cols,
                 int scale_rows, float scale);
};

#endif //
