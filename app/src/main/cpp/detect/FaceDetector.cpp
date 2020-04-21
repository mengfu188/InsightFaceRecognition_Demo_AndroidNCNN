#include <algorithm>
//#include "omp.h"
#include "FaceDetector.h"
//#include "faceDetector.id.h"
//#include "faceDetector.h"
#include <vector>

#include <android/log.h>

#define TAG "LightFaceSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

static Timer timer;

Detector::Detector() :
        _nms(0.4),
        _threshold(0.9),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(false),
        Net(new ncnn::Net()) {

}

inline void Detector::Release() {
    if (Net != nullptr) {
        delete Net;
        Net = nullptr;
    }
}

void Detector::Init(const std::string &model_param, const std::string &model_bin) {
    int ret = Net->load_param(model_param.c_str());
    ret = Net->load_model(model_bin.c_str());
}

void Detector::Init(const ncnn::Net &net_) {
//    Net = net_;
}

void Detector::Detect(ncnn::Net &net, ncnn::Mat &bgr, std::vector<bbox> &boxes) {

    timer.tic();

    int rows = bgr.h;
    int cols = bgr.w;
    const int max_side = 320;
    // scale
    float long_side = std::max(cols, rows);
    float scale = max_side / long_side;
    int target_h = rows * scale;
    int target_w = cols * scale;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char *) bgr.data, ncnn::Mat::PIXEL_BGR,
                                                 bgr.w, bgr.h, target_w, target_h);
    in.substract_mean_normalize(_mean_val, 0);

    LOGD("resize image width is %d, height is %d", in.w, in.h);
    timer.toc("precoss:");
    timer.tic();
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract("output0", out);

    // class
    ex.extract("530", out1);

    //landmark
    ex.extract("529", out2);
    timer.toc("model infer :");

    post_process(out, out1, out2, boxes, cols, rows, target_w, target_h, scale);
}

void Detector::set_threshold(float threshold) {
    this->_threshold = threshold;
    LOGD("threshold set to %f", this->_threshold);
}

void Detector::post_process(
        ncnn::Mat &out,
        ncnn::Mat &out1,
        ncnn::Mat &out2,
        std::vector<bbox> &boxes,
        int src_cols,
        int src_rows,
        int scale_cols,
        int scale_rows,
        float scale) {
    Timer timer;

    std::vector<box> anchor;
    timer.tic();
    if (_retinaface)
        create_anchor_retinaface(anchor, src_cols, src_rows);
    else
        create_anchor(anchor, src_cols, src_rows);
    timer.toc("create anchor:");
    timer.tic();
    std::vector<bbox> total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i) {
        if (*(ptr1 + 1) > _threshold) {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr + 3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx / 2) * scale_cols / scale;
            if (result.x1 < 0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy / 2) * scale_rows / scale;
            if (result.y1 < 0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx / 2) * scale_cols / scale;
            if (result.x2 > scale_cols)
                result.x2 = scale_cols;
            result.y2 = (tmp1.cy + tmp1.sy / 2) * scale_rows / scale;
            if (result.y2 > scale_rows)
                result.y2 = scale_rows;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j) {
                result.point[j]._x =
                        (tmp.cx + *(landms + (j << 1)) * 0.1 * tmp.sx) * scale_cols / scale;
                result.point[j]._y =
                        (tmp.cy + *(landms + (j << 1) + 1) * 0.1 * tmp.sy) * scale_rows / scale;
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    timer.tic();
    nms(total_box, _nms);
    timer.toc("nms");
    LOGD("total box %d\n", (int) total_box.size());

    for (int j = 0; j < total_box.size(); ++j) {
        boxes.push_back(total_box[j]);
    }
    timer.toc("post process");

}

float *bbox2float(const bbox bbox_) {

    float *arr = new float[15];
//    arr[]
    arr[0] = bbox_.s;
    arr[1] = bbox_.x1;
    arr[2] = bbox_.y1;
    arr[3] = bbox_.x2;
    arr[4] = bbox_.y2;

    arr[5] = bbox_.point[0]._x;
    arr[6] = bbox_.point[0]._y;
    arr[7] = bbox_.point[1]._x;
    arr[8] = bbox_.point[1]._y;
    arr[9] = bbox_.point[2]._x;
    arr[10] = bbox_.point[2]._y;
    arr[11] = bbox_.point[3]._x;
    arr[12] = bbox_.point[3]._y;
    arr[13] = bbox_.point[4]._x;
    arr[14] = bbox_.point[4]._y;
    return arr;
}

float *bboxs2float(std::vector<bbox> &bboxes_){
    int size = bboxes_.size();
    float *arr = new float[size*15];
    for(int i = 0; i < size; i++){
        for(int j = 0 ; j < 15; j++){
            float * arr_ = bbox2float(bboxes_[i]);
            arr[i*15 + j] = arr_[j];
        }
    }
    return arr;
}

inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detector::SetDefaultParams() {
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;

}

Detector::~Detector() {
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h) {
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detector::create_anchor_retinaface(std::vector<box> &anchor, int w, int h) {
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH) {
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
