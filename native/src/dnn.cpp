// dnn (cv2.dnn.*): networks read from model bytes (ONNX, TensorFlow, Caffe, Darknet, TFLite), blobs and box
// suppression. Runs on OpenCV's own CPU backend on every target.
// dnn：由模型位元組讀取的網路（ONNX、TensorFlow、Caffe、Darknet、TFLite）、blob 與框抑制。
#include "binding.hpp"
#include <opencv2/dnn.hpp>
#include <cctype>

namespace acv {
namespace {

using Net = cv::dnn::Net;

std::string extension(const std::string &path) {
    size_t dot = path.find_last_of('.');
    std::string e = dot == std::string::npos ? "" : path.substr(dot + 1);
    for (char &c : e) c = char(std::tolower(static_cast<unsigned char>(c)));
    return e;
}

Net read_net(Args &a, const std::string &model, const std::string &config, std::string framework) {
    std::vector<uchar> weights = read_resource(a, model);
    std::vector<uchar> settings = config.empty() ? std::vector<uchar>() : read_resource(a, config);
    if (framework.empty()) framework = extension(model);
    if (framework == "pb") framework = "tensorflow";
    if (framework == "caffemodel") framework = "caffe";
    if (framework == "weights") framework = "darknet";
    return cv::dnn::readNet(framework, weights, settings);
}

} // namespace

void register_dnn(aqua_library *library, aqua_class *cv) {
    aqua_class *net = define_class<Net>(library, "Net");
    FN(cv, "readNet|讀取網路", "s|ss", { return make<Net>(a.rt, read_net(a, a.text(0), a.text(1, ""), a.text(2, ""))); });
    FN(cv, "readNetFromONNX|讀取ONNX網路", "s", { return make<Net>(a.rt, read_net(a, a.text(0), "", "onnx")); });
    FN(cv, "readNetFromTensorflow|讀取TensorFlow網路", "s|s", { return make<Net>(a.rt, read_net(a, a.text(0), a.text(1, ""), "tensorflow")); });
    FN(cv, "readNetFromCaffe|讀取Caffe網路", "s|s", { return make<Net>(a.rt, read_net(a, a.text(1, ""), a.text(0), "caffe")); });
    FN(cv, "readNetFromDarknet|讀取Darknet網路", "s|s", { return make<Net>(a.rt, read_net(a, a.text(1, ""), a.text(0), "darknet")); });
    FN(cv, "readNetFromTFLite|讀取TFLite網路", "s", { return make<Net>(a.rt, read_net(a, a.text(0), "", "tflite")); });
    FN(cv, "blobFromImage|影像轉Blob", "v|nvvbbn", {
        return a.out(cv::dnn::blobFromImage(a.mat(0), a.number(1, 1), a.size(2, cv::Size()), a.scalar(3, cv::Scalar()), a.flag(4), a.flag(5, true),
            a.integer(6, CV_32F)));
    });
    FN(cv, "blobFromImages|多影像轉Blob", "v|nvvbbn", {
        return a.out(cv::dnn::blobFromImages(a.mats(0), a.number(1, 1), a.size(2, cv::Size()), a.scalar(3, cv::Scalar()), a.flag(4), a.flag(5, true),
            a.integer(6, CV_32F)));
    });
    FN(cv, "imagesFromBlob|Blob轉影像", "v", { std::vector<cv::Mat> images; cv::dnn::imagesFromBlob(a.mat(0), images); return a.out(images); });
    FN(cv, "NMSBoxes|非極大值抑制", "vvnn|nn", {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(a.rects(0), a.floats(1), float(a.number(2)), float(a.number(3)), indices, float(a.number(4, 1)), a.integer(5, 0));
        return a.out(indices);
    });

    FN(net, "setInput|設定輸入", "v|snv", {
        a.self<Net>().setInput(a.mat(0), a.text(1, ""), a.number(2, 1), a.scalar(3, cv::Scalar()));
        return a.none();
    });
    FN(net, "forward|前向傳播", "|v", {
        Net &n = a.self<Net>();
        if (a.has(0) && a.arg(0).type == AQUA_ARRAY) {
            std::vector<cv::Mat> outputs;
            n.forward(outputs, a.texts(0));
            std::vector<cv::Mat> copies;
            for (auto &o : outputs) copies.push_back(o.clone());
            return a.out(copies);
        }
        return a.out(n.forward(a.text(0, "")).clone());
    });
    FN(net, "getLayerNames|層名稱", "", { return a.out(a.self<Net>().getLayerNames()); });
    FN(net, "getUnconnectedOutLayersNames|輸出層名稱", "", { return a.out(a.self<Net>().getUnconnectedOutLayersNames()); });
    FN(net, "setPreferableBackend|設定後端", "n", { a.self<Net>().setPreferableBackend(a.integer(0)); return a.none(); });
    FN(net, "setPreferableTarget|設定運算目標", "n", { a.self<Net>().setPreferableTarget(a.integer(0)); return a.none(); });
    FN(net, "empty|是否為空", "", { return a.out(a.self<Net>().empty()); });
    FN(net, "getPerfProfile|效能資訊", "", {
        std::vector<double> timings;
        int64 total = a.self<Net>().getPerfProfile(timings);
        return a.tuple(double(total), timings);
    });
}

} // namespace acv
