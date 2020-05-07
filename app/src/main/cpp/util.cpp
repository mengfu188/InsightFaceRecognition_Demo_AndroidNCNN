//
// Created by cmf on 20-5-7.
//

#include <string>
#include <stack>
#include <vector>
#include <chrono>
#include <android/log.h>
#include <mat.h>

using namespace std::chrono;
#define TAG "UtilSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

namespace nn {
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

//    void tiny_print(ncnn::Mat &n) {
//        LOGD("%f %f %f %f %f ", n[0], n[1], n[2], n[3], n[4]);
//    }
} // namespace nn