//
// Created by cmf on 20-4-21.
//


#ifndef JNI_SSD_DETECT_CPP
#define JNI_SSD_DETECT_CPP

#include <stdio.h>
#include <time.h>
#include <vector>
#include <jni.h>

#include <android/log.h>

#define TAG "SSD_DETECT"

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO , TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN , TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__))


#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include "detect/FaceDetector.h"

// ===============================================

static ncnn::Net ssd_detector_net;
static Timer timer;
static Detector ssd_detector;
static int max_size = 320;

extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_FaceDetect_init(JNIEnv *env, jclass clazz,
                                         jobject asset_manager) {
    timer.tic();
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);

    if (mgr == NULL) {
        LOGD(" %s", "AAssetManager==NULL");
    }

    {
        const char *mfile = "faceDetector.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        ssd_detector_net.load_param(param_asset);
    }
    {
        const char *mfile = "faceDetector.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        ssd_detector_net.load_model(model_asset);
    }
    timer.toc("init face detect model");
}extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_chenty_testncnn_FaceDetect_detectByBitmap(JNIEnv *env, jclass clazz, jobject data) {

    timer.tic();

    int width, height, target_w, target_h, long_size;

    float scale;
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, data, &info);
        width = info.width;
        height = info.height;
        long_size = std::max(width, height);
        scale = max_size * 1.0 / long_size;
        target_h = height * scale;
        target_w = width * scale;

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void *indata;
        AndroidBitmap_lockPixels(env, data, &indata);

        in = ncnn::Mat::from_pixels_resize((const unsigned char *) indata,
                                           ncnn::Mat::PIXEL_RGBA2BGR,
                                           width, height, target_w, target_h);

        AndroidBitmap_unlockPixels(env, data);
    }

    LOGD("detect image width is %d, height is %d", width, height);
    LOGD("resize image width is %d, height is %d, scale is %f", target_w, target_h, scale);

    float _mean_val[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(_mean_val, 0);
    timer.toc("ssd pre process");
    timer.tic();
    ncnn::Extractor ex = ssd_detector_net.create_extractor();

    ex.set_light_mode(true);
    ex.set_num_threads(2);

    ex.input(0, in);
//    LOGD("ssd extract input");
    ncnn::Mat out, out1, out2;

// loc
    ex.extract("output0", out);
//    LOGD("ssd extract output0");

// class
    ex.extract("530", out1);
//    LOGD("ssd extract 530");

//landmark
    ex.extract("529", out2);
//    LOGD("ssd extract 529");

    timer.toc("ssd model infer");

    std::vector<bbox> boxes;

    ssd_detector.post_process(out, out1, out2, boxes, width, height, target_w, target_h, scale);

    if (boxes.size() == 0) {

        return nullptr;
    }
    int box_size = 15 * boxes.size();

    float *arr = bboxs2float(boxes);


    jfloatArray tFaceInfo = env->NewFloatArray(box_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, box_size, arr);

    return tFaceInfo;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_FaceDetect_setThreshold(JNIEnv *env, jclass clazz, jfloat threshold) {
    ssd_detector.set_threshold(threshold);
}

#endif