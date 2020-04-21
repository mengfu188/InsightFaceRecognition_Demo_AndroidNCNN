// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <time.h>
#include <vector>
#include <jni.h>

#include <android/log.h>

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "jni", __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO , "jni", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN , "libssd", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "libssd", __VA_ARGS__))

#include "arcface/interface.h"
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include "detect/FaceDetector.h"


#ifdef __cplusplus
extern "C" {
#endif

#define FACE_DETECT_SIZEH   448
int sizeh, sizev;
int global_src_width, global_src_htight;


int faceinfo2float(float *out, FaceInfo *in)
{
    if(sizeh == 0 || sizev == 0)
        return 20;
    *out = (float)in->x[0]/sizeh;out++;
    *out = (float)in->y[0]/sizev;out++;
    *out = (float)in->x[1]/sizeh;out++;
    *out = (float)in->y[1]/sizev;out++;
    *out = in->landmark[0];out++;
    *out = in->landmark[1];out++;
    *out = in->landmark[2];out++;
    *out = in->landmark[3];out++;
    *out = in->landmark[4];out++;
    *out = in->landmark[5];out++;
    *out = in->landmark[6];out++;
    *out = in->landmark[7];out++;
    *out = in->landmark[8];out++;
    *out = in->landmark[9];out++;
    *out = in->score;out++;
    *out = in->area;out++;
    *out = in->regreCoord[0];out++;
    *out = in->regreCoord[1];out++;
    *out = in->regreCoord[2];out++;
    *out = in->regreCoord[3];out++;
    return 20;
}

int float2faceinfo(FaceInfo *out, float *in)
{
    if(sizeh == 0 || sizev == 0)
        return 20;
    out->x[0]=(*in)*sizeh;in++;
    out->y[0]=(*in)*sizev;in++;
    out->x[1]=(*in)*sizeh;in++;
    out->y[1]=(*in)*sizev;in++;
    out->landmark[0]=*in;in++;
    out->landmark[1]=*in;in++;
    out->landmark[2]=*in;in++;
    out->landmark[3]=*in;in++;
    out->landmark[4]=*in;in++;
    out->landmark[5]=*in;in++;
    out->landmark[6]=*in;in++;
    out->landmark[7]=*in;in++;
    out->landmark[8]=*in;in++;
    out->landmark[9]=*in;in++;
    out->score=*in;in++;
    out->area=*in;in++;
    out->regreCoord[0]=*in;in++;
    out->regreCoord[1]=*in;in++;
    out->regreCoord[2]=*in;in++;
    out->regreCoord[3]=*in;in++;
    return 20;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_chenty_testncnn_CameraNcnnFragment_detectface(JNIEnv *env, jobject thiz, jbyteArray frame, jint src_width,
                                                        jint src_height) {
    char *yuv_frame = (char*)env->GetPrimitiveArrayCritical(frame, NULL);
    LOGD("MTCNN detectface src_width %d, src_height %d", src_width, src_height);
    int size = env->GetArrayLength(frame);
    int objectcnt = 0;
    int i;


    sizeh = FACE_DETECT_SIZEH;
    sizev = FACE_DETECT_SIZEH*src_height/src_width;

    LOGD("MTCNN detectface scale_width is %d, scale_height is %d", sizeh, sizev);

    //shift argb to rgba
    char *yuv = (char *)malloc(size);
    memcpy(yuv, yuv_frame, size);

    env->ReleasePrimitiveArrayCritical(frame, yuv_frame, JNI_ABORT);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((const unsigned char *)yuv, ncnn::Mat::PIXEL_GRAY2BGR, src_width, src_height, sizeh, sizev);
//    LOGD("detect face  in %dX%d", src_width, src_height);
    vector<FaceInfo> faceinfo = face_detect(in);

    free(yuv);
    int gap = 20;
    int cnt = faceinfo.size();
    if(cnt)
    {
        float detect_out[240];
        float *out = detect_out;

        if(cnt > 240/gap)
            cnt = 240/gap;

        for (int i = 0 ; i < cnt ; i ++)
        {
            FaceInfo face = faceinfo[i];
            LOGD("MTCNN detectface get face %d %d %d %d", face.x[0], face.y[0], face.x[1], face.y[1]);
            int res = faceinfo2float(out, &face);
            LOGD("MTCNN detectface faceinfo2float get face %f %f %f %f", out[0],out[1],out[2],out[3]);
            out=out+res;
        }

        FaceInfo face = faceinfo[0];
        vector<float> feature = face_exactfeature(in, face);

        // scale size for landmark
        for(int i = 0; i < cnt; i++){

            // landmark
            for(int j = 4; j <= 13; j+=2){
                detect_out[gap * i + j]  = detect_out[gap * i + j] / sizeh ;
                detect_out[gap * i + j + 1]  = detect_out[gap * i + j + 1] / sizev ;
            }

        }

        LOGD("MTCNN detectface get feature %d ", feature.size());
        float feature_f[256];
        jfloatArray detect = env->NewFloatArray(cnt*gap + 128);
        LOGD("MTCNN detectface vect2float %d", feature.size());
        out = feature_f;
        for(int i = 0 ; i < feature.size(); i++)
        {
            *out = feature[i];
            out++;
        }
        //vect2float(feature_f, feature);
        //memcpy(feature_f, &feature[0], feature.size()*sizeof(float));
//        LOGD("MTCNN detectface set feature %d ", feature.size());
        env->SetFloatArrayRegion(detect,0,128, feature_f);
        LOGD("MTCNN detectface set face count %d, gap %d ", cnt, gap);
        env->SetFloatArrayRegion(detect, 128, cnt*gap, detect_out);

        return detect;
    } else{

        return nullptr;
    }
}



extern "C" JNIEXPORT jfloat JNICALL
Java_com_chenty_testncnn_CameraNcnnFragment_compareface(JNIEnv *env, jobject thiz, jfloatArray face0, jfloatArray face1) {

    float face0cache[128], face1cache[128];

    env->GetFloatArrayRegion(face0,0,128,face0cache);
    env->GetFloatArrayRegion(face1,0,128,face1cache);

    vector<float> face0feature(face0cache, face0cache+sizeof(face0cache)/sizeof(float));
    vector<float> face1feature(face1cache, face1cache+sizeof(face1cache)/sizeof(float));

    jfloat result = face_calcSimilar(face0feature, face1feature);
    LOGD("calcSimilar result %f\n", result);

    return result;


}

extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_CameraNcnnFragment_initMtcnn(JNIEnv *env, jobject thiz,
                                                      jobject asset_manager) {
    // TODO: implement initMtcnn()
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
    static ncnn::Net pnet, rnet, onet, lnet;



    if (mgr == NULL) {
        LOGD(" %s", "AAssetManager==NULL");
    }else{
        LOGD("AAssetManager is not NULL");
    }

    {
        const char *mfile = "det1.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        pnet.load_param(param_asset);
    }
    {
        const char *mfile = "det1.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        pnet.load_model(model_asset);
    }

    {
        const char *mfile = "det2.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        rnet.load_param(param_asset);
    }
    {
        const char *mfile = "det2.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        rnet.load_model(model_asset);
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

    {
        const char *mfile = "det4.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        lnet.load_param(param_asset);
    }
    {
        const char *mfile = "det4.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        lnet.load_model(model_asset);
    }

//    g_mtcnnDetector = MtcnnDetector(pnet, rnet, onet, lnet);
    initMtcnn(pnet, rnet, onet, lnet);

}

#ifdef __cplusplus  
}  
#endif
extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_CameraNcnnFragment_intiArcface(JNIEnv *env, jobject thiz,
                                                        jobject asset_manager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
    static ncnn::Net net;

    if (mgr == NULL) {
        LOGD(" %s", "AAssetManager==NULL");
    }else{
        LOGD("AAssetManager is not NULL");
    }

    {
        const char *mfile = "faceRecognize.param";
        AAsset *param_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (param_asset == NULL) {
            LOGD(" %s", "param_asset==NULL");
        }
        net.load_param(param_asset);
    }
    {
        const char *mfile = "faceRecognize.bin";
        AAsset *model_asset = AAssetManager_open(mgr, mfile, AASSET_MODE_UNKNOWN);
        if (model_asset == NULL) {
            LOGD(" %s", "model_asset==NULL");
        }
        net.load_model(model_asset);
    }
    init_arcface(net);
}

// ===============================================

static ncnn::Net ssd_detector_net;
static Timer timer;
static Detector ssd_detector;
static int max_size = 320;

extern "C"
JNIEXPORT void JNICALL
Java_com_chenty_testncnn_CameraNcnnFragment_initSsd(JNIEnv *env, jobject thiz,
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
Java_com_chenty_testncnn_CameraNcnnFragment_detectSsd(JNIEnv *env, jobject thiz, jobject data) {

    timer.tic();

    int width, height, target_w, target_h, long_size;

    float scale;
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, data, &info);
        width = info.width;
        height = info.height;
        long_size = max(width, height);
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
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract("output0", out);

    // class
    ex.extract("530", out1);

    //landmark
    ex.extract("529", out2);
    timer.toc("ssd model infer");

    vector<bbox> boxes;

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