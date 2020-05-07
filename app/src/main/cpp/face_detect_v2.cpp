//
// Created by cmf on 20-5-7.
//


#include "net.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <android/log.h>

#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include <android/bitmap.h>
#include "util.cpp"
#include "detect_v2/UltraFace.hpp"

#define TAG "FaceDetectV2So"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

#include <chrono>
#include <math.h>
#include <stack>

using namespace std;

static nn::Timer timer;
static ncnn::Net net;
static UltraFace *ultraface;


extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_chenty_testncnn_FaceDetectV2_detectByBitmap(JNIEnv *env, jclass clazz,
                                                            jobject data) {
    timer.tic();
    int width, height, target_w, target_h, long_size;

    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, data, &info);
        width = info.width;
        height = info.height;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void *indata;
        AndroidBitmap_lockPixels(env, data, &indata);

        in = ncnn::Mat::from_pixels((const unsigned char *) indata,
                                           ncnn::Mat::PIXEL_RGBA2RGB,
                                           width, height);

        AndroidBitmap_unlockPixels(env, data);
    }

    LOGD("detect image width is %d, height is %d", width, height);

    timer.toc("pre process");

    std::vector<FaceInfo> face_info;
    timer.tic();
    ultraface->detect(in, face_info);
    LOGD("detect %d num of face", face_info.size());
    timer.toc("model infer");
    if (face_info.size() == 0){
        return nullptr;
    }

    int box_size = 5;
    float *arr = new float[box_size];
    if(face_info.size() > 0){

        FaceInfo info = face_info[0];
        arr[0] = info.score;
        arr[1] = info.x1;
        arr[2] = info.y1;
        arr[3] = info.x2;
        arr[4] = info.y2;
    }

    jfloatArray tFaceInfo = env->NewFloatArray(box_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, box_size, arr);

    return tFaceInfo;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_FaceDetectV2_init(JNIEnv *env, jclass clazz, jobject manager,
        jint face_size, jboolean horizontal) {
    timer.tic();
    AAssetManager *mgr = AAssetManager_fromJava(env, manager);

    if (mgr == NULL) {
        LOGD(" %s", "AAssetManager==NULL");
    }

    {
        const char *mfile = "face_detector_slim_320.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        net.load_param(param_asset);
    }
    {
        const char *mfile = "face_detector_slim_320.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        net.load_model(model_asset);
    }
    static UltraFace ultraface_ = UltraFace(net, std::to_string(face_size), 2, 0.7, 0.3, -1, horizontal);
    ultraface = &ultraface_;
    timer.toc("init face detector v2 model");
}


