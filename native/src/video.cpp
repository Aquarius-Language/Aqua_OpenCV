// video: optical flow, background subtraction, mean shift tracking, Kalman filtering and object trackers.
// video：光流、背景相減、均值漂移追蹤、卡爾曼濾波及物件追蹤器。
#include "binding.hpp"
#include <opencv2/video.hpp>

namespace acv {
namespace {

using Subtractor = cv::Ptr<cv::BackgroundSubtractor>;
using TrackerBox = cv::Ptr<cv::Tracker>;
using Kalman = cv::KalmanFilter;
using DisBox = cv::Ptr<cv::DISOpticalFlow>;

// A Kalman matrix as a Mat object sharing the filter's data, so scripts set it in place (copyTo, setTo, set).
template <cv::Mat Kalman::*Field> void kalman_matrix(aqua_class *cls, const char *names) {
    aqua_define_property(cls, names, call<[](Args &a) -> aqua_value { return a.out(a.self<Kalman>().*Field); }>);
}

} // namespace

void register_video(aqua_library *library, aqua_class *cv) {
    FN(cv, "calcOpticalFlowPyrLK|金字塔光流", "vvv|vvnvnn", {
        std::vector<cv::Point2f> next = a.has(3) ? a.points2f(3) : std::vector<cv::Point2f>();
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(a.mat(0), a.mat(1), a.points2f(2), next, status, error, a.size(4, cv::Size(21, 21)), a.integer(5, 3),
            a.criteria(6, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)), a.integer(7, 0), a.number(8, 1e-4));
        std::vector<double> errors(error.begin(), error.end());
        return a.tuple(next, std::vector<int>(status.begin(), status.end()), errors);
    });
    FN(cv, "calcOpticalFlowFarneback|稠密光流", "vvvnnnnnn|n", {
        cv::Mat flow = a.has(2) ? a.mat(2).clone() : cv::Mat();
        cv::calcOpticalFlowFarneback(a.mat(0), a.mat(1), flow, a.number(3), a.integer(4), a.integer(5), a.integer(6), a.integer(7), a.number(8), a.integer(9, 0));
        return a.out(flow);
    });
    aqua_class *dis = define_class<DisBox>(library, "DISOpticalFlow");
    FN(cv, "DISOpticalFlow_create|建立DIS光流", "|n", { return make<DisBox>(a.rt, cv::DISOpticalFlow::create(a.integer(0, cv::DISOpticalFlow::PRESET_FAST))); });
    FN(dis, "calc|計算", "vv|v", {
        cv::Mat flow = a.has(2) ? a.mat(2).clone() : cv::Mat();
        a.self<DisBox>()->calc(a.mat(0), a.mat(1), flow);
        return a.out(flow);
    });
    FN(cv, "meanShift|均值漂移", "vvv", {
        cv::Rect window = a.rect(1);
        int iterations = cv::meanShift(a.mat(0), window, a.criteria(2, cv::TermCriteria()));
        return a.tuple(iterations, window);
    });
    FN(cv, "CamShift|CamShift追蹤", "vvv", {
        cv::Rect window = a.rect(1);
        cv::RotatedRect box = cv::CamShift(a.mat(0), window, a.criteria(2, cv::TermCriteria()));
        return a.tuple(box, window);
    });
    FN(cv, "findTransformECC|ECC對齊", "vvvn|v", {
        cv::Mat warp = a.mat(2).clone();
        warp.convertTo(warp, CV_32F);
        double rho = cv::findTransformECC(a.mat(0), a.mat(1), warp, a.integer(3),
            a.criteria(4, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001)), cv::noArray());
        return a.tuple(rho, warp);
    });

    aqua_class *mog = define_class<Subtractor>(library, "BackgroundSubtractorMOG2");
    aqua_class *knn = define_class<Subtractor>(library, "BackgroundSubtractorKNN");
    for (aqua_class *cls : {mog, knn}) {
        FN(cls, "apply|套用", "v|n", { cv::Mat mask; a.self<Subtractor>()->apply(a.mat(0), mask, a.number(1, -1)); return a.out(mask); });
        FN(cls, "getBackgroundImage|背景影像", "", { cv::Mat d; a.self<Subtractor>()->getBackgroundImage(d); return a.out(d); });
    }
    FN(cv, "createBackgroundSubtractorMOG2|建立MOG2背景相減", "|nnb", {
        return make<Subtractor>(a.rt, cv::createBackgroundSubtractorMOG2(a.integer(0, 500), a.number(1, 16), a.flag(2, true)), Kind<Subtractor>::classes[0]);
    });
    FN(cv, "createBackgroundSubtractorKNN|建立KNN背景相減", "|nnb", {
        return make<Subtractor>(a.rt, cv::createBackgroundSubtractorKNN(a.integer(0, 500), a.number(1, 400), a.flag(2, true)), Kind<Subtractor>::classes[1]);
    });

    aqua_class *kalman = define_class<Kalman>(library, "KalmanFilter");
    FN(cv, "KalmanFilter|卡爾曼濾波器", "nn|nn", {
        return make<Kalman>(a.rt, Kalman(a.integer(0), a.integer(1), a.integer(2, 0), a.integer(3, CV_32F)));
    });
    FN(kalman, "predict|預測", "|v", { return a.out(a.self<Kalman>().predict(a.mat(0, cv::Mat())).clone()); });
    FN(kalman, "correct|修正", "v", {
        cv::Mat measurement;
        a.mat(0).convertTo(measurement, a.self<Kalman>().measurementMatrix.type());
        return a.out(a.self<Kalman>().correct(measurement).clone());
    });
    kalman_matrix<&Kalman::statePre>(kalman, "statePre|預測狀態");
    kalman_matrix<&Kalman::statePost>(kalman, "statePost|修正狀態");
    kalman_matrix<&Kalman::transitionMatrix>(kalman, "transitionMatrix|轉移矩陣");
    kalman_matrix<&Kalman::controlMatrix>(kalman, "controlMatrix|控制矩陣");
    kalman_matrix<&Kalman::measurementMatrix>(kalman, "measurementMatrix|量測矩陣");
    kalman_matrix<&Kalman::processNoiseCov>(kalman, "processNoiseCov|過程雜訊");
    kalman_matrix<&Kalman::measurementNoiseCov>(kalman, "measurementNoiseCov|量測雜訊");
    kalman_matrix<&Kalman::errorCovPre>(kalman, "errorCovPre|預測誤差");
    kalman_matrix<&Kalman::errorCovPost>(kalman, "errorCovPost|修正誤差");
    kalman_matrix<&Kalman::gain>(kalman, "gain|增益");

    aqua_class *mil = define_class<TrackerBox>(library, "TrackerMIL");
    FN(cv, "TrackerMIL_create|建立MIL追蹤器", "", { return make<TrackerBox>(a.rt, cv::TrackerMIL::create()); });
    FN(mil, "init|初始化", "vv", { a.self<TrackerBox>()->init(a.mat(0), a.rect(1)); return a.none(); });
    FN(mil, "update|更新", "v", {
        cv::Rect box;
        bool ok = a.self<TrackerBox>()->update(a.mat(0), box);
        return a.tuple(ok, box);
    });
}

} // namespace acv
