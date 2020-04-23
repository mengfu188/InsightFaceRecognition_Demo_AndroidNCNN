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
#include <android/asset_manager_jni.h>
#include "detect/FaceDetector.h"

#define TAG "OPENCV_TRACKER"

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__))

cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();

static int max_side = 150;
static int bbox_size = 4;
static ncnn::Net onet;
static float onet_threshold = 0.8;
static int onet_size = 48;
static float onet_expand = 0.05;
static float mean_vals[3] = {127.5f, 127.5f, 127.5f};
static float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
static Timer timer;

bool onet_detect(cv::Mat img, float *box);

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_chenty_testncnn_Tracker_initTrack___3III_3F(JNIEnv *env, jclass clazz, jintArray buf,
                                                     jint w, jint h, jfloatArray bbox) {
    /*
     * bbox: x1, y1, x2, y2. norm value of buf.
     *
     * return: x1, y1, x2, y2. norm value of buf.
     */



    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE);

    float cbbox[4];
    env->GetFloatArrayRegion(bbox, 0, bbox_size, cbbox);

    if (cbuf == NULL) {
        return 0;
    }

    int width = w, height = h, target_w, target_h, long_size;
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
Java_com_chenty_testncnn_Tracker_updateTrack___3III_3F(JNIEnv *env, jclass clazz, jintArray buf,
                                                       jint w, jint h, jfloatArray bbox) {
    /*
     * bbox: x1, y1, x2, y2. norm value of buf.
     *
     * return: x1, y1, x2, y2. norm value of buf.
     */
    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
//    jfloat *cbbox = env->GetFloatArrayElements(bbox, JNI_FALSE);

    //     4 [xmin, ymin, xmax, ymax]
    // or 14 [xmin, ymin, xmax, ymax, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    jsize len = env->GetArrayLength(bbox);

    float cbbox[len];
    env->GetFloatArrayRegion(bbox, 0, len, cbbox);

    if (cbuf == NULL) {
        return 0;
    }

    int width = w, height = h, target_w, target_h, long_size;
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

    if (!state) {
        roi = cv::Rect2d(0, 0, 0, 0);
        LOGD("track fail");
    }

    float result[len];
    float *resultp = result;

    result[0] = float(roi.x / target_w);
    result[1] = float(roi.y / target_h);
    result[2] = float((roi.x + roi.width) / target_w);
    result[3] = float((roi.y + roi.height) / target_h);

    if (len == 14) {
        bool ret = onet_detect(scale_img, resultp);
        if(!ret){
            result[0] = result[1] = result[2]=result[3] =0;
        }else{

        }
    }

    jfloatArray resultInfo = env->NewFloatArray(len);
    env->SetFloatArrayRegion(resultInfo, 0, len, result);

    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return resultInfo;
}

float clamp(float val, float min, float max) {
    return val < min ? min : (val > max ? max : val);
}

bool onet_detect(cv::Mat img, float *box) {
    /**
     * img: bgr
     * box: x1, y1, x2, y2; (norm)
     */

    int img_h = img.rows;
    int img_w = img.cols;

    box[0] -= onet_expand;
    box[1] -= onet_expand;
    box[2] += onet_expand;
    box[3] += onet_expand;

    for (int i = 0; i < 4; i++) {
        box[i] = clamp(box[i], 0, 1);
    }
    LOGD("raw box [%f, %f, %f, %f] , %f, %f", box[0], box[1], box[2], box[3], box[4], box[5]);

    int roi_x = box[0] * img_w;
    int roi_y = box[1] * img_h;
    int roi_w = (box[2] - box[0]) * img_w;
    int roi_h = (box[3] - box[1]) * img_h;
    int roi_r = box[2] * img_w;
    int roi_b = box[3] * img_h;
    cv::Mat img_c = img(cv::Rect(roi_x, roi_y, roi_w, roi_h));
    cv::Mat img_s;
    cv::resize(img_c, img_s, cv::Size(onet_size, onet_size));

    timer.tic();
    ncnn::Mat in = ncnn::Mat::from_pixels(img_s.data, ncnn::Mat::PIXEL_BGR, onet_size, onet_size);
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = onet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(2);
    ex.input("data", in);
    ncnn::Mat score, bbox, point;
    ex.extract("prob1", score);
    ex.extract("conv6_2", bbox);
    ex.extract("conv6_3", point);
    timer.toc("onet infer ");
    if ((float) score[1] > onet_threshold) {

        LOGD("net bbox [%f, %f, %f, %f]", bbox[0], bbox[1], bbox[2], bbox[3]);
        // TODO fix problem
        box[0] = ((float) bbox[0] * roi_w + roi_x) / img_w;
        box[1] = ((float) bbox[1] * roi_h + roi_y) / img_h;
        box[2] = ((float) bbox[2] * roi_w + roi_r) / img_w;
        box[3] = ((float) bbox[3] * roi_h + roi_b) / img_h;

        for (int p = 0; p < 5; p++) {
            box[2 * p + 4] = (roi_x + roi_w * point[p]) / img_w;
            box[2 * p + 1 + 4] = (roi_y + roi_h * point[p + 5]) / img_h;
        }

        for (int i = 0; i < 14; i++) {
            box[i] = clamp(box[i], 0, 1);
        }
        LOGD("tune box [%f, %f, %f, %f] , %f, %f", box[0], box[1], box[2], box[3], box[4], box[5]);
        return true;

    } else {
        LOGD("onet did not detect face from score %f and threshold %f", (float) score[1],
             onet_threshold);
        return false;
    }
}

extern "C"
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
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_Tracker_init(JNIEnv *env, jclass clazz, jobject manager) {
    timer.tic();
    AAssetManager *mgr = AAssetManager_fromJava(env, manager);

    if (mgr == NULL) {
        LOGD(" %s", "AAssetManager==NULL");
    }

    {
        const char *mfile = "det3.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        onet.load_param(param_asset);
    }
    {
        const char *mfile = "det3.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        onet.load_model(model_asset);
    }
    timer.toc("init onet model");
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_chenty_testncnn_Tracker_gray(JNIEnv *env, jclass clazz, jintArray buf, jint w, jint h) {
    jint *cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    cv::Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);

    uchar *ptr = imgData.ptr(0);
    for (int i = 0; i < w * h; i++) {
        //计算公式：Y(亮度) = 0.299*R + 0.587*G + 0.114*B
        //对于一个int四字节，其彩色值存储方式为：BGRA
        int grayScale = (int) (ptr[4 * i + 2] * 0.299 + ptr[4 * i + 1] * 0.587 +
                               ptr[4 * i + 0] * 0.114);
        ptr[4 * i + 1] = grayScale;
        ptr[4 * i + 2] = grayScale;
        ptr[4 * i + 0] = grayScale;
    }

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, cbuf);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

#endif