//
// Created by cmf on 20-4-21.
//

#ifndef JNI_TRACK
#define JNI_TRACK

#include <string>
#include <jni.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "detect/FaceDetector.h"

#define TAG "OPENCV_TRACKER"

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__))

cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();

static int max_side=270;
static int bbox_size = 4;
static Timer timer;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_chenty_testncnn_Tracker_init(JNIEnv *env, jclass clazz, jintArray buf, jint w, jint h, jfloatArray bbox) {
    /*
     * bbox: x1, y1, x2, y2. norm value of buf.
     *
     * return: x1, y1, x2, y2. norm value of buf.
     */



    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE );

    float cbbox[4];
    env->GetFloatArrayRegion(bbox, 0, bbox_size, cbbox);

    if (cbuf == NULL) {
        return 0;
    }

    int width=w, height=h, target_w, target_h, long_size;
    float scale;

    long_size = std::max(width, height);
    scale = max_side * 1.0 / long_size;
    target_h = height * scale;
    target_w = width * scale;


    cv::Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);
    cv::cvtColor(imgData, imgData, cv::COLOR_BGRA2BGR);


//    for(int i = 0; i < bbox_size; i++){
//        cbbox[i] = cbbox[i] * scale;
//
//    }

    cbbox[0] = cbbox[0] * target_w;
    cbbox[1] = cbbox[1] * target_h;
    cbbox[2] = cbbox[2] * target_w;
    cbbox[3] = cbbox[3] * target_h;

    cv::Mat scale_img;
    cv::resize(imgData, scale_img, cv::Size(target_w, target_h));
    cv::Rect2d roi(cbbox[0], cbbox[1], cbbox[2] - cbbox[0], cbbox[3] - cbbox[1]);

    timer.tic();
    bool state = tracker->init(scale_img, roi);
    timer.toc("tracker init");



    env->ReleaseIntArrayElements(buf, cbuf, 0);
//    env->ReleaseFloatArrayElements(bbox, cbbox, 0);
//    env->ReleaseIntArrayElements(bbox, cbbox, 0);
    return false;
}extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_chenty_testncnn_Tracker_update(JNIEnv *env, jclass clazz, jintArray buf, jint w, jint h, jfloatArray bbox) {
    /*
     * bbox: x1, y1, x2, y2. norm value of buf.
     *
     * return: x1, y1, x2, y2. norm value of buf.
     */
    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE );
//    jfloat *cbbox = env->GetFloatArrayElements(bbox, JNI_FALSE);

    float cbbox[4];
    env->GetFloatArrayRegion(bbox, 0, bbox_size, cbbox);

    if (cbuf == NULL) {
        return 0;
    }

    int width=w, height=h, target_w, target_h, long_size;
    float scale;

    long_size = std::max(width, height);
    scale = max_side * 1.0 / long_size;
    target_h = height * scale;
    target_w = width * scale;

    cv::Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);
    cv::cvtColor(imgData, imgData, cv::COLOR_BGRA2BGR);


//    for(int i = 0; i < bbox_size; i++){
//        cbbox[i] = cbbox[i] * scale;
//    }

    cbbox[0] = cbbox[0] * target_w;
    cbbox[1] = cbbox[1] * target_h;
    cbbox[2] = cbbox[2] * target_w;
    cbbox[3] = cbbox[3] * target_h;

    cv::Mat scale_img;
    cv::resize(imgData, scale_img, cv::Size(target_w, target_h));
    cv::Rect2d roi(cbbox[0], cbbox[1], cbbox[2] - cbbox[0], cbbox[3] - cbbox[1]);

    timer.tic();
    bool state = tracker->update(scale_img, roi);
    timer.toc("tracker update");



    if(!state){
        roi = cv::Rect2d(0, 0, 0, 0);
        LOGD("track fail");
    }

//    int result[4] = {roi.x, roi.y, roi.x+roi.width, roi.y+roi.height};
    float result[4] = {float(roi.x / target_w), float(roi.y / target_h), float((roi.x+roi.width) / target_w), float((roi.y+roi.height) / target_h)};

    jfloatArray resultInfo = env->NewFloatArray(bbox_size);
    env->SetFloatArrayRegion(resultInfo, 0, bbox_size, result);

    env->ReleaseIntArrayElements(buf, cbuf, 0);
//    env->ReleaseFloatArrayElements(bbox, cbbox, 0);
//    env->ReleaseIntArrayElements(bbox, cbbox, 0);
    return resultInfo;
}extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_Tracker_reset(JNIEnv *env, jclass clazz) {
//    if(tracker == NULL){
//        return ;
//    }
    timer.tic();
    tracker->clear();
    tracker.release();
    tracker = cv::TrackerKCF::create();
    timer.toc("tracker reset");
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_chenty_testncnn_Tracker_gray(JNIEnv *env, jclass clazz, jintArray buf, jint w, jint h) {
    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE );
    if (cbuf == NULL) {
        return 0;
    }

    cv::Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);

    uchar* ptr = imgData.ptr(0);
    for(int i = 0; i < w*h; i ++){
        //计算公式：Y(亮度) = 0.299*R + 0.587*G + 0.114*B
        //对于一个int四字节，其彩色值存储方式为：BGRA
        int grayScale = (int)(ptr[4*i+2]*0.299 + ptr[4*i+1]*0.587 + ptr[4*i+0]*0.114);
        ptr[4*i+1] = grayScale;
        ptr[4*i+2] = grayScale;
        ptr[4*i+0] = grayScale;
    }

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, cbuf);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

#endif