// objdetect: cascade classifiers, HOG people detection, QR codes, barcodes, ArUco markers and YuNet/SFace faces.
// Model and cascade files are read from packaged assets (or files on native targets), as cv2 reads paths.
// objdetect：串接分類器、HOG 行人偵測、QR 碼、條碼、ArUco 標記與 YuNet／SFace 人臉。
#include "binding.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/barcode.hpp>
#include <opencv2/objdetect/face.hpp>

namespace acv {
namespace {

using Cascade = cv::CascadeClassifier;
using Hog = cv::HOGDescriptor;
using QrDetector = cv::QRCodeDetector;
using QrEncoder = cv::Ptr<cv::QRCodeEncoder>;
using Barcode = cv::barcode::BarcodeDetector;
using Dictionary = cv::aruco::Dictionary;
using Aruco = cv::aruco::ArucoDetector;
using FaceYN = cv::Ptr<cv::FaceDetectorYN>;
using FaceSF = cv::Ptr<cv::FaceRecognizerSF>;

void load_cascade(Args &a, Cascade &cascade, const std::string &path) {
    std::vector<uchar> bytes = read_resource(a, path);
    cv::FileStorage storage(std::string(bytes.begin(), bytes.end()), cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!cascade.read(storage.getFirstTopLevelNode())) a.fail("'%s' is not a cascade classifier", path.c_str());
}

std::vector<std::vector<cv::Point2f>> corner_groups(const cv::Mat &points) {
    std::vector<std::vector<cv::Point2f>> groups;
    if (points.empty()) return groups;
    cv::Mat p = points.reshape(2, int(points.total() * points.channels() / 8));
    for (int r = 0; r < p.rows; r++) {
        std::vector<cv::Point2f> group;
        for (int c = 0; c < 4; c++) group.push_back(p.ptr<cv::Point2f>(r)[c]);
        groups.push_back(group);
    }
    return groups;
}

void register_cascades(aqua_library *library, aqua_class *cv) {
    aqua_class *cascade = define_class<Cascade>(library, "CascadeClassifier");
    FN(cv, "CascadeClassifier|串接分類器", "|s", {
        Cascade c;
        if (a.has(0)) load_cascade(a, c, a.text(0));
        return make<Cascade>(a.rt, c);
    });
    FN(cascade, "load|載入", "s", { load_cascade(a, a.self<Cascade>(), a.text(0)); return a.out(true); });
    FN(cascade, "empty|是否為空", "", { return a.out(a.self<Cascade>().empty()); });
    FN(cascade, "detectMultiScale|多尺度偵測", "v|nnnvv", {
        std::vector<cv::Rect> objects;
        a.self<Cascade>().detectMultiScale(a.mat(0), objects, a.number(1, 1.1), a.integer(2, 3), a.integer(3, 0),
            a.size(4, cv::Size()), a.size(5, cv::Size()));
        return a.out(objects);
    });

    aqua_class *hog = define_class<Hog>(library, "HOGDescriptor");
    FN(cv, "HOGDescriptor|HOG描述子", "|vvvvn", {
        if (!a.has(0)) return make<Hog>(a.rt, Hog());
        return make<Hog>(a.rt, Hog(a.size(0), a.size(1), a.size(2), a.size(3), a.integer(4)));
    });
    FN(cv, "HOGDescriptor_getDefaultPeopleDetector|預設行人偵測器", "", {
        std::vector<float> detector = Hog::getDefaultPeopleDetector();
        return a.out(cv::Mat(detector, true));
    });
    FN(hog, "setSVMDetector|設定SVM偵測器", "v", { a.self<Hog>().setSVMDetector(a.mat(0)); return a.none(); });
    FN(hog, "compute|計算", "v|vv", {
        std::vector<float> descriptors;
        a.self<Hog>().compute(a.mat(0), descriptors, a.size(1, cv::Size()), a.size(2, cv::Size()));
        return a.out(cv::Mat(descriptors, true));
    });
    FN(hog, "detectMultiScale|多尺度偵測", "v|nvvnn", {
        std::vector<cv::Rect> found;
        std::vector<double> weights;
        a.self<Hog>().detectMultiScale(a.mat(0), found, weights, a.number(1, 0), a.size(2, cv::Size()), a.size(3, cv::Size()),
            a.number(4, 1.05), a.number(5, 2));
        return a.tuple(found, weights);
    });
}

void register_codes(aqua_library *library, aqua_class *cv) {
    aqua_class *qr = define_class<QrDetector>(library, "QRCodeDetector");
    FN(cv, "QRCodeDetector|QR碼偵測器", "", { return make<QrDetector>(a.rt, QrDetector()); });
    FN(qr, "detect|偵測", "v", {
        std::vector<cv::Point2f> points;
        bool ok = a.self<QrDetector>().detect(a.mat(0), points);
        return a.tuple(ok, points);
    });
    FN(qr, "decode|解碼", "vv", {
        cv::Mat straight;
        std::string text = a.self<QrDetector>().decode(a.mat(0), a.points2f(1), straight);
        return a.tuple(text, straight);
    });
    FN(qr, "detectAndDecode|偵測並解碼", "v", {
        std::vector<cv::Point2f> points;
        cv::Mat straight;
        std::string text = a.self<QrDetector>().detectAndDecode(a.mat(0), points, straight);
        return a.tuple(text, points, straight);
    });
    FN(qr, "detectAndDecodeMulti|偵測並解碼多個", "v", {
        std::vector<std::string> texts;
        cv::Mat points;
        bool ok = a.self<QrDetector>().detectAndDecodeMulti(a.mat(0), texts, points);
        return a.tuple(ok, texts, corner_groups(points));
    });

    aqua_class *encoder = define_class<QrEncoder>(library, "QRCodeEncoder");
    FN(cv, "QRCodeEncoder_create|建立QR碼編碼器", "", { return make<QrEncoder>(a.rt, cv::QRCodeEncoder::create()); });
    FN(encoder, "encode|編碼", "s", { cv::Mat d; a.self<QrEncoder>()->encode(a.text(0), d); return a.out(d); });

    aqua_class *barcode = define_class<Barcode>(library, "BarcodeDetector");
    FN(cv, "BarcodeDetector|條碼偵測器", "", { return make<Barcode>(a.rt, Barcode()); });
    FN(barcode, "detectAndDecode|偵測並解碼", "v", {
        std::vector<cv::Point2f> points;
        std::string text = a.self<Barcode>().detectAndDecode(a.mat(0), points);
        return a.tuple(text, points);
    });
    FN(barcode, "detectAndDecodeWithType|偵測解碼含類型", "v", {
        std::vector<std::string> texts, types;
        std::vector<cv::Point2f> points;
        bool ok = a.self<Barcode>().detectAndDecodeWithType(a.mat(0), texts, types, points);
        cv::Mat corners(points, true);
        return a.tuple(ok, texts, types, corner_groups(corners));
    });
}

void register_aruco(aqua_library *library, aqua_class *cv) {
    define_class<Dictionary>(library, "Dictionary");
    FN(cv, "getPredefinedDictionary|預設標記字典", "n", { return make<Dictionary>(a.rt, cv::aruco::getPredefinedDictionary(a.integer(0))); });
    FN(cv, "generateImageMarker|產生標記影像", "vnn|n", {
        cv::Mat d;
        cv::aruco::generateImageMarker(a.object<Dictionary>(0), a.integer(1), a.integer(2), d, a.integer(3, 1));
        return a.out(d);
    });
    aqua_class *detector = define_class<Aruco>(library, "ArucoDetector");
    FN(cv, "ArucoDetector|ArUco偵測器", "v", { return make<Aruco>(a.rt, Aruco(a.object<Dictionary>(0))); });
    FN(detector, "detectMarkers|偵測標記", "v", {
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        a.self<Aruco>().detectMarkers(a.mat(0), corners, ids, rejected);
        return a.tuple(corners, ids, rejected);
    });
    FN(cv, "drawDetectedMarkers|繪製偵測標記", "vv|vv", {
        cv::Mat ids = a.has(2) ? cv::Mat(a.ints(2), true) : cv::Mat();
        cv::aruco::drawDetectedMarkers(a.target(0), a.contours2f(1), ids, a.scalar(3, cv::Scalar(0, 255, 0)));
        return a.arg(0);
    });
}

void register_faces(aqua_library *library, aqua_class *cv) {
    aqua_class *yn = define_class<FaceYN>(library, "FaceDetectorYN");
    FN(cv, "FaceDetectorYN_create|建立YuNet人臉偵測器", "svv|nnn", {
        std::vector<uchar> model = read_resource(a, a.text(0));
        std::vector<uchar> config = a.has(1) && !a.text(1).empty() ? read_resource(a, a.text(1)) : std::vector<uchar>();
        return make<FaceYN>(a.rt, cv::FaceDetectorYN::create("onnx", model, config, a.size(2), float(a.number(3, 0.9)),
            float(a.number(4, 0.3)), a.integer(5, 5000)));
    });
    FN(yn, "setInputSize|設定輸入尺寸", "v", { a.self<FaceYN>()->setInputSize(a.size(0)); return a.none(); });
    FN(yn, "getInputSize|輸入尺寸", "", { return a.out(a.self<FaceYN>()->getInputSize()); });
    FN(yn, "setScoreThreshold|設定分數門檻", "n", { a.self<FaceYN>()->setScoreThreshold(float(a.number(0))); return a.none(); });
    FN(yn, "setNMSThreshold|設定NMS門檻", "n", { a.self<FaceYN>()->setNMSThreshold(float(a.number(0))); return a.none(); });
    FN(yn, "setTopK|設定前K個", "n", { a.self<FaceYN>()->setTopK(a.integer(0)); return a.none(); });
    FN(yn, "detect|偵測", "v", {
        cv::Mat faces;
        int found = a.self<FaceYN>()->detect(a.mat(0), faces);
        return a.tuple(found, faces);
    });

    aqua_class *sf = define_class<FaceSF>(library, "FaceRecognizerSF");
    FN(cv, "FaceRecognizerSF_create|建立SFace人臉辨識器", "sv", {
        std::vector<uchar> model = read_resource(a, a.text(0));
        std::vector<uchar> config = a.has(1) && !a.text(1).empty() ? read_resource(a, a.text(1)) : std::vector<uchar>();
        return make<FaceSF>(a.rt, cv::FaceRecognizerSF::create("onnx", model, config));
    });
    FN(sf, "alignCrop|對齊裁切", "vv", { cv::Mat d; a.self<FaceSF>()->alignCrop(a.mat(0), a.mat(1), d); return a.out(d); });
    FN(sf, "feature|特徵", "v", { cv::Mat d; a.self<FaceSF>()->feature(a.mat(0), d); return a.out(d.clone()); });
    FN(sf, "match|比對", "vv|n", { return a.out(a.self<FaceSF>()->match(a.mat(0), a.mat(1), a.integer(2, cv::FaceRecognizerSF::FR_COSINE))); });
}

} // namespace

void register_objdetect(aqua_library *library, aqua_class *cv) {
    register_cascades(library, cv);
    register_codes(library, cv);
    register_aruco(library, cv);
    register_faces(library, cv);
    aqua_define_int(cv, "FaceRecognizerSF_FR_COSINE", cv::FaceRecognizerSF::FR_COSINE);
    aqua_define_int(cv, "FaceRecognizerSF_FR_NORM_L2", cv::FaceRecognizerSF::FR_NORM_L2);
}

} // namespace acv
