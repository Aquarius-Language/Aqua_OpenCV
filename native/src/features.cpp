// features2d: keypoint detectors and descriptor extractors, descriptor matchers and their drawing functions.
// features2d：特徵點偵測、描述子擷取、描述子匹配及其繪圖函式。
#include "binding.hpp"
#include <opencv2/features2d.hpp>

namespace acv {
namespace {

using Feature = cv::Ptr<cv::Feature2D>;
using Matcher = cv::Ptr<cv::DescriptorMatcher>;

aqua_value keypoints_of(Args &a, const std::vector<cv::KeyPoint> &keys) { return a.out(keys); }

// Methods every detector shares (cv2.Feature2D).
void feature_methods(aqua_class *cls) {
    FN(cls, "detect|偵測", "v|v", {
        std::vector<cv::KeyPoint> keys;
        a.self<Feature>()->detect(a.mat(0), keys, a.mat(1, cv::Mat()));
        return keypoints_of(a, keys);
    });
    FN(cls, "compute|計算描述子", "vv", {
        std::vector<cv::KeyPoint> keys = a.keypoints(1);
        cv::Mat descriptors;
        a.self<Feature>()->compute(a.mat(0), keys, descriptors);
        return a.tuple(keys, descriptors);
    });
    FN(cls, "detectAndCompute|偵測並計算", "v|v", {
        std::vector<cv::KeyPoint> keys;
        cv::Mat descriptors;
        a.self<Feature>()->detectAndCompute(a.mat(0), a.mat(1, cv::Mat()), keys, descriptors);
        return a.tuple(keys, descriptors);
    });
    FN(cls, "descriptorSize|描述子長度", "", { return a.out(a.self<Feature>()->descriptorSize()); });
    FN(cls, "defaultNorm|預設範數", "", { return a.out(a.self<Feature>()->defaultNorm()); });
    FN(cls, "getDefaultName|名稱", "", { return a.out(a.self<Feature>()->getDefaultName()); });
}

aqua_class *feature_class(aqua_library *library, const char *name) {
    aqua_class *cls = define_class<Feature>(library, name);
    feature_methods(cls);
    return cls;
}

// cv2.SimpleBlobDetector_Params as a hash of the same field names.
cv::SimpleBlobDetector::Params blob_params(Args &a, unsigned i) {
    cv::SimpleBlobDetector::Params p;
    auto number = [&](const char *name, float &target) { aqua_value v = a.field(i, name); if (aqua_is_number(v)) target = float(v.number); };
    auto flag = [&](const char *name, bool &target) { aqua_value v = a.field(i, name); if (v.type == AQUA_BOOL) target = v.number != 0; };
    number("thresholdStep", p.thresholdStep); number("minThreshold", p.minThreshold); number("maxThreshold", p.maxThreshold);
    number("minDistBetweenBlobs", p.minDistBetweenBlobs); number("minArea", p.minArea); number("maxArea", p.maxArea);
    number("minCircularity", p.minCircularity); number("maxCircularity", p.maxCircularity);
    number("minInertiaRatio", p.minInertiaRatio); number("maxInertiaRatio", p.maxInertiaRatio);
    number("minConvexity", p.minConvexity); number("maxConvexity", p.maxConvexity);
    flag("filterByColor", p.filterByColor); flag("filterByArea", p.filterByArea); flag("filterByCircularity", p.filterByCircularity);
    flag("filterByInertia", p.filterByInertia); flag("filterByConvexity", p.filterByConvexity);
    aqua_value color = a.field(i, "blobColor");
    if (aqua_is_number(color)) p.blobColor = uchar(color.number);
    aqua_value repeat = a.field(i, "minRepeatability");
    if (aqua_is_number(repeat)) p.minRepeatability = size_t(repeat.number);
    return p;
}

} // namespace

void register_features(aqua_library *library, aqua_class *cv) {
    // Kind<Feature>::classes keeps this order; each factory below creates objects of its class.
    for (const char *name : {"ORB", "SIFT", "AKAZE", "KAZE", "BRISK", "FastFeatureDetector", "AgastFeatureDetector", "GFTTDetector",
             "SimpleBlobDetector"})
        feature_class(library, name);
    aqua_class *mser = feature_class(library, "MSER");

    FN(cv, "ORB_create|建立ORB", "|nnnnnnnnn", {
        return make<Feature>(a.rt, cv::ORB::create(a.integer(0, 500), float(a.number(1, 1.2)), a.integer(2, 8), a.integer(3, 31), a.integer(4, 0),
            a.integer(5, 2), cv::ORB::ScoreType(a.integer(6, cv::ORB::HARRIS_SCORE)), a.integer(7, 31), a.integer(8, 20)), Kind<Feature>::classes[0]);
    });
    FN(cv, "SIFT_create|建立SIFT", "|nnnnn", {
        return make<Feature>(a.rt, cv::SIFT::create(a.integer(0, 0), a.integer(1, 3), a.number(2, 0.04), a.number(3, 10), a.number(4, 1.6)),
            Kind<Feature>::classes[1]);
    });
    FN(cv, "AKAZE_create|建立AKAZE", "|nnnnnnn", {
        return make<Feature>(a.rt, cv::AKAZE::create(cv::AKAZE::DescriptorType(a.integer(0, cv::AKAZE::DESCRIPTOR_MLDB)), a.integer(1, 0),
            a.integer(2, 3), float(a.number(3, 0.001)), a.integer(4, 4), a.integer(5, 4), cv::KAZE::DiffusivityType(a.integer(6, cv::KAZE::DIFF_PM_G2))),
            Kind<Feature>::classes[2]);
    });
    FN(cv, "KAZE_create|建立KAZE", "|bbnnn", {
        return make<Feature>(a.rt, cv::KAZE::create(a.flag(0), a.flag(1), float(a.number(2, 0.001)), a.integer(3, 4), a.integer(4, 4)),
            Kind<Feature>::classes[3]);
    });
    FN(cv, "BRISK_create|建立BRISK", "|nnn", {
        return make<Feature>(a.rt, cv::BRISK::create(a.integer(0, 30), a.integer(1, 3), float(a.number(2, 1))), Kind<Feature>::classes[4]);
    });
    FN(cv, "FastFeatureDetector_create|建立FAST偵測器", "|nbn", {
        return make<Feature>(a.rt, cv::FastFeatureDetector::create(a.integer(0, 10), a.flag(1, true),
            cv::FastFeatureDetector::DetectorType(a.integer(2, cv::FastFeatureDetector::TYPE_9_16))), Kind<Feature>::classes[5]);
    });
    FN(cv, "AgastFeatureDetector_create|建立AGAST偵測器", "|nb", {
        return make<Feature>(a.rt, cv::AgastFeatureDetector::create(a.integer(0, 10), a.flag(1, true)), Kind<Feature>::classes[6]);
    });
    FN(cv, "GFTTDetector_create|建立GFTT偵測器", "|nnnnbn", {
        return make<Feature>(a.rt, cv::GFTTDetector::create(a.integer(0, 1000), a.number(1, 0.01), a.number(2, 1), a.integer(3, 3),
            a.flag(4), a.number(5, 0.04)), Kind<Feature>::classes[7]);
    });
    FN(cv, "SimpleBlobDetector_create|建立斑點偵測器", "|h", {
        return make<Feature>(a.rt, cv::SimpleBlobDetector::create(blob_params(a, 0)), Kind<Feature>::classes[8]);
    });
    FN(cv, "MSER_create|建立MSER", "|nnnn", {
        return make<Feature>(a.rt, cv::MSER::create(a.integer(0, 5), a.integer(1, 60), a.integer(2, 14400), a.number(3, 0.25)),
            Kind<Feature>::classes[9]);
    });
    FN(mser, "detectRegions|偵測區域", "v", {
        auto detector = a.self<Feature>().dynamicCast<cv::MSER>();
        std::vector<std::vector<cv::Point>> regions;
        std::vector<cv::Rect> boxes;
        detector->detectRegions(a.mat(0), regions, boxes);
        return a.tuple(regions, boxes);
    });

    aqua_class *bf = define_class<Matcher>(library, "BFMatcher");
    aqua_class *flann = define_class<Matcher>(library, "FlannBasedMatcher");
    for (aqua_class *cls : {bf, flann}) {
        FN(cls, "match|匹配", "vv|v", {
            std::vector<cv::DMatch> matches;
            a.self<Matcher>()->match(a.mat(0), a.mat(1), matches, a.mat(2, cv::Mat()));
            return a.out(matches);
        });
        FN(cls, "knnMatch|K近鄰匹配", "vvn|v", {
            std::vector<std::vector<cv::DMatch>> matches;
            a.self<Matcher>()->knnMatch(a.mat(0), a.mat(1), matches, a.integer(2), a.mat(3, cv::Mat()));
            return a.out(matches);
        });
        FN(cls, "radiusMatch|半徑匹配", "vvn|v", {
            std::vector<std::vector<cv::DMatch>> matches;
            a.self<Matcher>()->radiusMatch(a.mat(0), a.mat(1), matches, float(a.number(2)), a.mat(3, cv::Mat()));
            return a.out(matches);
        });
    }
    FN(cv, "BFMatcher|暴力匹配器", "|nb", {
        return make<Matcher>(a.rt, cv::BFMatcher::create(a.integer(0, cv::NORM_L2), a.flag(1)), Kind<Matcher>::classes[0]);
    });
    FN(cv, "BFMatcher_create|建立暴力匹配器", "|nb", {
        return make<Matcher>(a.rt, cv::BFMatcher::create(a.integer(0, cv::NORM_L2), a.flag(1)), Kind<Matcher>::classes[0]);
    });
    FN(cv, "FlannBasedMatcher|FLANN匹配器", "", { return make<Matcher>(a.rt, cv::FlannBasedMatcher::create(), Kind<Matcher>::classes[1]); });
    FN(cv, "FlannBasedMatcher_create|建立FLANN匹配器", "", {
        return make<Matcher>(a.rt, cv::FlannBasedMatcher::create(), Kind<Matcher>::classes[1]);
    });

    FN(cv, "drawKeypoints|繪製特徵點", "vv|vvn", {
        cv::Mat d = a.has(2) ? a.target(2) : cv::Mat();
        cv::drawKeypoints(a.mat(0), a.keypoints(1), d, a.scalar(3, cv::Scalar::all(-1)), cv::DrawMatchesFlags(a.integer(4, 0)));
        return a.out(d);
    });
    FN(cv, "drawMatches|繪製匹配", "vvvvv|vvvvn", {
        cv::Mat d = a.has(5) ? a.target(5) : cv::Mat();
        std::vector<char> mask;
        if (a.has(8)) for (int m : a.ints(8)) mask.push_back(char(m != 0));
        cv::drawMatches(a.mat(0), a.keypoints(1), a.mat(2), a.keypoints(3), a.matches(4), d, a.scalar(6, cv::Scalar::all(-1)),
            a.scalar(7, cv::Scalar::all(-1)), mask, cv::DrawMatchesFlags(a.integer(9, 0)));
        return a.out(d);
    });
    FN(cv, "drawMatchesKnn|繪製K近鄰匹配", "vvvvv|vvvn", {
        cv::Mat d = a.has(5) ? a.target(5) : cv::Mat();
        cv::drawMatches(a.mat(0), a.keypoints(1), a.mat(2), a.keypoints(3), a.knn_matches(4), d, a.scalar(6, cv::Scalar::all(-1)),
            a.scalar(7, cv::Scalar::all(-1)), std::vector<std::vector<char>>(), cv::DrawMatchesFlags(a.integer(8, 0)));
        return a.out(d);
    });
}

} // namespace acv
