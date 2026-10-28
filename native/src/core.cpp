// core and imgcodecs: the Mat class, array operations, image files, conversions to the
// runtime's image objects, and the constants.
// core 與 imgcodecs：Mat 類別、陣列運算、影像檔案、與執行環境影像物件的轉換及常數。
#include "binding.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <fstream>

namespace acv {
namespace {

// Pixels as the runtime's RGBA image, with imshow's scaling: 16-bit / 256, floating point x 255. An image object of the
// same size passed as `reuse` is overwritten and returned, so a frame loop makes no new image objects (whose pixels
// the collector cannot see).
aqua_value to_image(Args &a, const cv::Mat &m, unsigned reuse) {
    if (m.empty()) a.fail("the Mat is empty");
    if (m.dims > 2 || m.channels() == 2 || m.channels() > 4) a.fail("toImage needs a 2-D Mat with 1, 3 or 4 channels");
    cv::Mat bytes = m;
    if (m.depth() == CV_16U) m.convertTo(bytes, CV_8U, 1.0 / 256);
    else if (m.depth() == CV_32F || m.depth() == CV_64F) m.convertTo(bytes, CV_8U, 255);
    else if (m.depth() != CV_8U) m.convertTo(bytes, CV_8U);
    cv::Mat rgba;
    int code = bytes.channels() == 1 ? cv::COLOR_GRAY2RGBA : bytes.channels() == 3 ? cv::COLOR_BGR2RGBA : cv::COLOR_BGRA2RGBA;
    aqua_alpha alpha = bytes.channels() == 4 ? AQUA_ALPHA_STRAIGHT : AQUA_ALPHA_OPAQUE;
    if (a.has(reuse)) {
        aqua_image *target = aqua_image_of(a.rt, a.arg(reuse));
        if (!target || target->width != m.cols || target->height != m.rows) a.fail("toImage can reuse only an image of the same size");
        cv::Mat into(target->height, target->width, CV_8UC4, target->rgba);
        cv::cvtColor(bytes, into, code);
        target->alpha = alpha;
        return a.arg(reuse);
    }
    cv::cvtColor(bytes, rgba, code);
    return aqua_image_new(a.rt, rgba.cols, rgba.rows, rgba.data, alpha);
}

// One element as a number (one channel) or an array of channel values.
aqua_value element(Args &a, const cv::Mat &m, int row, int col) {
    if (row < 0 || col < 0 || row >= m.rows || col >= m.cols) a.fail("element (%d, %d) is outside the %dx%d Mat", row, col, m.rows, m.cols);
    cv::Mat value;
    m(cv::Rect(col, row, 1, 1)).convertTo(value, CV_64F);
    const double *v = value.ptr<double>();
    if (m.channels() == 1) return aqua_double(v[0]);
    return aqua_array_from_f64(a.rt, v, uint32_t(m.channels()), AQUA_DOUBLE);
}

// All elements as nested arrays: rows x cols (x channels when more than one).
aqua_value to_array(Args &a, const cv::Mat &m) {
    cv::Mat d;
    m.convertTo(d, CV_64F);
    aqua_type kind = m.depth() == CV_32F || m.depth() == CV_64F || m.depth() == CV_16F ? AQUA_DOUBLE : AQUA_INT;
    aqua_value rows = aqua_array_new(a.rt, uint32_t(d.rows));
    for (int r = 0; r < d.rows; r++) {
        const double *p = d.ptr<double>(r);
        if (d.channels() == 1) {
            aqua_array_set(a.rt, rows, uint32_t(r), aqua_array_from_f64(a.rt, p, uint32_t(d.cols), kind));
            continue;
        }
        aqua_value row = aqua_array_new(a.rt, uint32_t(d.cols));
        for (int c = 0; c < d.cols; c++)
            aqua_array_set(a.rt, row, uint32_t(c), aqua_array_from_f64(a.rt, p + c * d.channels(), uint32_t(d.channels()), kind));
        aqua_array_set(a.rt, rows, uint32_t(r), row);
    }
    return rows;
}

cv::Mat filled(const cv::Mat &m, Args &a, unsigned i) {
    if (!a.has(i)) return m;
    std::vector<double> values = a.doubles(i);
    cv::Mat source(int(values.size()), 1, CV_64F, values.data()), result;
    if (int(values.size()) != m.total() * m.channels()) a.fail("expected %d values, got %d", int(m.total() * m.channels()), int(values.size()));
    source.reshape(m.channels(), m.rows).convertTo(result, m.type());
    return result;
}

void register_classes(aqua_library *library) {
    aqua_class *mat = define_class<cv::Mat>(library, "Mat");
    aqua_define_property(mat, "rows|列數", call<[](Args &a) -> aqua_value { return a.out(a.self<cv::Mat>().rows); }>);
    aqua_define_property(mat, "cols|行數", call<[](Args &a) -> aqua_value { return a.out(a.self<cv::Mat>().cols); }>);
    FN(mat, "type|類型", "", { return a.out(a.self<cv::Mat>().type()); });
    FN(mat, "depth|深度", "", { return a.out(a.self<cv::Mat>().depth()); });
    FN(mat, "channels|通道數", "", { return a.out(a.self<cv::Mat>().channels()); });
    FN(mat, "total|元素總數", "", { return a.out(double(a.self<cv::Mat>().total())); });
    FN(mat, "elemSize|元素位元組數", "", { return a.out(double(a.self<cv::Mat>().elemSize())); });
    FN(mat, "empty|是否為空", "", { return a.out(a.self<cv::Mat>().empty()); });
    FN(mat, "size|尺寸", "", { return a.out(a.self<cv::Mat>().size()); });
    FN(mat, "shape|形狀", "", {
        cv::Mat &m = a.self<cv::Mat>();
        if (m.channels() == 1) return a.out(std::vector<int>{m.rows, m.cols});
        return a.out(std::vector<int>{m.rows, m.cols, m.channels()});
    });
    FN(mat, "clone|複製", "", { return a.out(a.self<cv::Mat>().clone()); });
    FN(mat, "copyTo|複製到", "|vv", {
        cv::Mat &m = a.self<cv::Mat>();
        if (!a.has(0)) { cv::Mat result; m.copyTo(result, a.mat(1, cv::Mat())); return a.out(result); }
        m.copyTo(a.target(0), a.mat(1, cv::Mat()));
        return a.arg(0);
    });
    FN(mat, "convertTo|轉換型別", "n|nn", {
        cv::Mat result;
        a.self<cv::Mat>().convertTo(result, a.integer(0), a.number(1, 1), a.number(2, 0));
        return a.out(result);
    });
    FN(mat, "setTo|設定值", "v|v", { a.self<cv::Mat>().setTo(a.scalar(0), a.mat(1, cv::Mat())); return a.self_value(); });
    FN(mat, "roi|區域", "v", { return a.out(a.self<cv::Mat>()(a.rect(0))); });
    FN(mat, "row|列", "n", { return a.out(a.self<cv::Mat>().row(a.integer(0))); });
    FN(mat, "col|行", "n", { return a.out(a.self<cv::Mat>().col(a.integer(0))); });
    FN(mat, "rowRange|列範圍", "nn", { return a.out(a.self<cv::Mat>().rowRange(a.integer(0), a.integer(1))); });
    FN(mat, "colRange|行範圍", "nn", { return a.out(a.self<cv::Mat>().colRange(a.integer(0), a.integer(1))); });
    FN(mat, "reshape|重塑", "n|n", { return a.out(a.self<cv::Mat>().reshape(a.integer(0), a.integer(1, 0))); });
    FN(mat, "t|轉置矩陣", "", { return a.out(cv::Mat(a.self<cv::Mat>().t())); });
    FN(mat, "inv|反矩陣", "|n", { return a.out(cv::Mat(a.self<cv::Mat>().inv(a.integer(0, cv::DECOMP_LU)))); });
    FN(mat, "mul|逐元素相乘", "v|n", { return a.out(cv::Mat(a.self<cv::Mat>().mul(a.mat(0), a.number(1, 1)))); });
    FN(mat, "matmul|矩陣相乘", "v", { return a.out(cv::Mat(a.self<cv::Mat>() * a.mat(0))); });
    FN(mat, "dot|內積", "v", { return a.out(a.self<cv::Mat>().dot(a.mat(0))); });
    FN(mat, "cross|外積", "v", { return a.out(a.self<cv::Mat>().cross(a.mat(0))); });
    FN(mat, "at|取值", "nn", { return element(a, a.self<cv::Mat>(), a.integer(0), a.integer(1)); });
    FN(mat, "set|設值", "nnv", {
        cv::Mat &m = a.self<cv::Mat>();
        int row = a.integer(0), col = a.integer(1);
        if (row < 0 || col < 0 || row >= m.rows || col >= m.cols) a.fail("element (%d, %d) is outside the Mat", row, col);
        m(cv::Rect(col, row, 1, 1)).setTo(a.scalar(2));
        return a.self_value();
    });
    FN(mat, "toArray|轉為陣列", "", { return to_array(a, a.self<cv::Mat>()); });
    FN(mat, "data|資料", "", {
        cv::Mat m = a.self<cv::Mat>();
        if (!m.isContinuous()) m = m.clone();
        if (m.depth() == CV_8U) return aqua_array_from_u8(a.rt, m.data, uint32_t(m.total() * m.channels()));
        cv::Mat d;
        m.convertTo(d, CV_64F);
        aqua_type kind = m.depth() == CV_32F || m.depth() == CV_64F ? AQUA_DOUBLE : AQUA_INT;
        return aqua_array_from_f64(a.rt, d.ptr<double>(), uint32_t(d.total() * d.channels()), kind);
    });
    FN(mat, "toImage|轉為影像", "|v", { return to_image(a, a.self<cv::Mat>(), 0); });
    FN(mat, "delete|刪除", "", { a.self<cv::Mat>().release(); return a.none(); });
    FN(mat, "release|釋放", "", { a.self<cv::Mat>().release(); return a.none(); });
}

void register_constants(aqua_class *cv) {
#define CV(name) aqua_define_int(cv, #name, int(cv::name));
#define RAW(name) aqua_define_int(cv, #name, int(name));
#define NS(name, value) aqua_define_int(cv, #name, int(value));
#include "constants.def"
#undef CV
#undef RAW
#undef NS
    aqua_define_double(cv, "CV_PI", CV_PI);
    // The language has no null literal: cv.None (cv2's None, C++'s noArray()) skips an optional argument.
    aqua_define_property(cv, "None|無", call<[](Args &a) -> aqua_value { return a.none(); }>);
    aqua_define_string(cv, "haarcascades|人臉模型路徑", "/bottles/Aqua_OpenCV/assets/");
    aqua_define_string(cv, "samples|範例路徑", "/bottles/Aqua_OpenCV/assets/");
}

} // namespace

void register_core(aqua_library *library, aqua_class *cv) {
    register_classes(library);
    register_constants(cv);

    // Construction.
    FN(cv, "Mat|矩陣", "|vvvv", {
        if (!a.has(0)) return a.out(cv::Mat());
        if (a.arg(0).type == AQUA_ARRAY || !aqua_is_number(a.arg(0))) return a.out(a.mat(0).clone());
        return a.out(cv::Mat(a.integer(0), a.integer(1), a.integer(2, CV_8UC1), a.scalar(3, cv::Scalar())));
    });
    FN(cv, "zeros|零矩陣", "nn|n", { return a.out(cv::Mat(cv::Mat::zeros(a.integer(0), a.integer(1), a.integer(2, CV_64F)))); });
    FN(cv, "ones|全一矩陣", "nn|n", { return a.out(cv::Mat(cv::Mat::ones(a.integer(0), a.integer(1), a.integer(2, CV_64F)))); });
    FN(cv, "eye|單位矩陣", "nn|n", { return a.out(cv::Mat(cv::Mat::eye(a.integer(0), a.integer(1), a.integer(2, CV_64F)))); });
    FN(cv, "matFromArray|陣列轉矩陣", "nnnv", {
        cv::Mat m(a.integer(0), a.integer(1), a.integer(2));
        return a.out(filled(m, a, 3));
    });
    FN(cv, "CV_MAKETYPE|組合類型", "nn", { return a.out(CV_MAKETYPE(a.integer(0), a.integer(1))); });
    FN(cv, "KeyPoint|特徵點", "nnn|nnnn", {
        return a.out(cv::KeyPoint(float(a.number(0)), float(a.number(1)), float(a.number(2)), float(a.number(3, -1)),
            float(a.number(4, 0)), a.integer(5, 0), a.integer(6, -1)));
    });
    FN(cv, "DMatch|匹配", "nnn|n", {
        if (a.has(3)) return a.out(cv::DMatch(a.integer(0), a.integer(1), a.integer(2), float(a.number(3))));
        return a.out(cv::DMatch(a.integer(0), a.integer(1), float(a.number(2))));
    });
    FN(cv, "KeyPoint_convert|特徵點轉座標", "v", {
        std::vector<cv::Point2f> points;
        cv::KeyPoint::convert(a.keypoints(0), points);
        return a.out(points);
    });

    // Interop with the runtime's image objects (Images, Webcam, Camera, Processing through Window).
    FN(cv, "fromImage|從影像", "v|b", {
        aqua_image *image = aqua_image_of(a.rt, a.arg(0));
        if (!image) a.fail("argument 1 must be an image object");
        cv::Mat result;
        cv::cvtColor(cv::Mat(image->height, image->width, CV_8UC4, image->rgba), result,
            a.flag(1) ? cv::COLOR_RGBA2BGRA : cv::COLOR_RGBA2BGR);
        return a.out(result);
    });
    FN(cv, "toImage|轉為影像", "v|v", { return to_image(a, a.mat(0), 1); });

    // Image files.
    FN(cv, "imread|讀取影像", "s|n", {
        std::vector<uchar> bytes = read_resource(a, a.text(0));
        return a.out(cv::imdecode(bytes, a.integer(1, cv::IMREAD_COLOR)));
    });
    FN(cv, "imdecode|解碼影像", "v|n", { return a.out(cv::imdecode(a.bytes(0), a.integer(1, cv::IMREAD_COLOR))); });
    FN(cv, "imencode|編碼影像", "sv|v", {
        std::vector<uchar> bytes;
        bool ok = cv::imencode(a.text(0), a.mat(1), bytes, a.has(2) ? a.ints(2) : std::vector<int>());
        return a.tuple(ok, bytes);
    });
    FN(cv, "imwrite|寫入影像", "sv|v", {
#if defined(AQUA_OS_WEB)
        a.fail("imwrite needs a file system; on the web use imencode and the Files library");
#else
        std::string path = a.text(0);
        std::vector<uchar> bytes;
        std::string extension = path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.'));
        if (!cv::imencode(extension, a.mat(1), bytes, a.has(2) ? a.ints(2) : std::vector<int>())) return a.out(false);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
        return a.out(bool(file));
#endif
    });
    FN(cv, "haveImageReader|可讀取影像", "s", { return a.out(cv::haveImageReader(a.text(0))); });

    // Per-element arithmetic and logic.
    FN(cv, "add|相加", "vv|vn", { cv::Mat d; cv::add(a.mat(0), a.mat(1), d, a.mat(2, cv::Mat()), a.integer(3, -1)); return a.out(d); });
    FN(cv, "subtract|相減", "vv|vn", { cv::Mat d; cv::subtract(a.mat(0), a.mat(1), d, a.mat(2, cv::Mat()), a.integer(3, -1)); return a.out(d); });
    FN(cv, "multiply|相乘", "vv|nn", { cv::Mat d; cv::multiply(a.mat(0), a.mat(1), d, a.number(2, 1), a.integer(3, -1)); return a.out(d); });
    FN(cv, "divide|相除", "vv|nn", { cv::Mat d; cv::divide(a.mat(0), a.mat(1), d, a.number(2, 1), a.integer(3, -1)); return a.out(d); });
    FN(cv, "absdiff|絕對差", "vv", { cv::Mat d; cv::absdiff(a.mat(0), a.mat(1), d); return a.out(d); });
    FN(cv, "addWeighted|加權相加", "vnvnn|n", {
        cv::Mat d;
        cv::addWeighted(a.mat(0), a.number(1), a.mat(2), a.number(3), a.number(4), d, a.integer(5, -1));
        return a.out(d);
    });
    FN(cv, "scaleAdd|縮放相加", "vnv", { cv::Mat d; cv::scaleAdd(a.mat(0), a.number(1), a.mat(2), d); return a.out(d); });
    FN(cv, "bitwise_and|位元且", "vv|v", { cv::Mat d; cv::bitwise_and(a.mat(0), a.mat(1), d, a.mat(2, cv::Mat())); return a.out(d); });
    FN(cv, "bitwise_or|位元或", "vv|v", { cv::Mat d; cv::bitwise_or(a.mat(0), a.mat(1), d, a.mat(2, cv::Mat())); return a.out(d); });
    FN(cv, "bitwise_xor|位元互斥或", "vv|v", { cv::Mat d; cv::bitwise_xor(a.mat(0), a.mat(1), d, a.mat(2, cv::Mat())); return a.out(d); });
    FN(cv, "bitwise_not|位元反相", "v|v", { cv::Mat d; cv::bitwise_not(a.mat(0), d, a.mat(1, cv::Mat())); return a.out(d); });
    FN(cv, "min|逐元素最小值", "vv", { cv::Mat d; cv::min(a.mat(0), a.mat(1), d); return a.out(d); });
    FN(cv, "max|逐元素最大值", "vv", { cv::Mat d; cv::max(a.mat(0), a.mat(1), d); return a.out(d); });
    FN(cv, "compare|比較", "vvn", { cv::Mat d; cv::compare(a.mat(0), a.mat(1), d, a.integer(2)); return a.out(d); });
    FN(cv, "inRange|範圍遮罩", "vvv", { cv::Mat d; cv::inRange(a.mat(0), a.mat(1), a.mat(2), d); return a.out(d); });
    FN(cv, "pow|次方", "vn", { cv::Mat d; cv::pow(a.mat(0), a.number(1), d); return a.out(d); });
    FN(cv, "sqrt|平方根", "v", { cv::Mat d; cv::sqrt(a.mat(0), d); return a.out(d); });
    FN(cv, "exp|指數", "v", { cv::Mat d; cv::exp(a.mat(0), d); return a.out(d); });
    FN(cv, "log|對數", "v", { cv::Mat d; cv::log(a.mat(0), d); return a.out(d); });
    FN(cv, "magnitude|向量長度", "vv", { cv::Mat d; cv::magnitude(a.mat(0), a.mat(1), d); return a.out(d); });
    FN(cv, "phase|相位", "vv|b", { cv::Mat d; cv::phase(a.mat(0), a.mat(1), d, a.flag(2)); return a.out(d); });
    FN(cv, "cartToPolar|直角轉極座標", "vv|b", {
        cv::Mat m, p;
        cv::cartToPolar(a.mat(0), a.mat(1), m, p, a.flag(2));
        return a.tuple(m, p);
    });
    FN(cv, "polarToCart|極座標轉直角", "vv|b", {
        cv::Mat x, y;
        cv::polarToCart(a.mat(0), a.mat(1), x, y, a.flag(2));
        return a.tuple(x, y);
    });
    FN(cv, "convertScaleAbs|縮放取絕對值", "v|nn", { cv::Mat d; cv::convertScaleAbs(a.mat(0), d, a.number(1, 1), a.number(2, 0)); return a.out(d); });
    FN(cv, "normalize|正規化", "v|nnnnv", {
        cv::Mat d;
        cv::normalize(a.mat(0), d, a.number(1, 1), a.number(2, 0), a.integer(3, cv::NORM_L2), a.integer(4, -1), a.mat(5, cv::Mat()));
        return a.out(d);
    });
    FN(cv, "LUT|查表", "vv", { cv::Mat d; cv::LUT(a.mat(0), a.mat(1), d); return a.out(d); });

    // Channels, layout and borders.
    FN(cv, "split|分離通道", "v", { std::vector<cv::Mat> planes; cv::split(a.mat(0), planes); return a.out(planes); });
    FN(cv, "merge|合併通道", "v", { cv::Mat d; cv::merge(a.mats(0), d); return a.out(d); });
    FN(cv, "extractChannel|擷取通道", "vn", { cv::Mat d; cv::extractChannel(a.mat(0), d, a.integer(1)); return a.out(d); });
    FN(cv, "insertChannel|插入通道", "vvn", { cv::insertChannel(a.mat(0), a.target(1), a.integer(2)); return a.arg(1); });
    FN(cv, "flip|翻轉", "vn", { cv::Mat d; cv::flip(a.mat(0), d, a.integer(1)); return a.out(d); });
    FN(cv, "rotate|旋轉", "vn", { cv::Mat d; cv::rotate(a.mat(0), d, a.integer(1)); return a.out(d); });
    FN(cv, "transpose|轉置", "v", { cv::Mat d; cv::transpose(a.mat(0), d); return a.out(d); });
    FN(cv, "repeat|重複", "vnn", { cv::Mat d; cv::repeat(a.mat(0), a.integer(1), a.integer(2), d); return a.out(d); });
    FN(cv, "hconcat|水平串接", "v", { cv::Mat d; cv::hconcat(a.mats(0), d); return a.out(d); });
    FN(cv, "vconcat|垂直串接", "v", { cv::Mat d; cv::vconcat(a.mats(0), d); return a.out(d); });
    FN(cv, "copyMakeBorder|加邊框", "vnnnnn|v", {
        cv::Mat d;
        cv::copyMakeBorder(a.mat(0), d, a.integer(1), a.integer(2), a.integer(3), a.integer(4), a.integer(5), a.scalar(6, cv::Scalar()));
        return a.out(d);
    });
    FN(cv, "reduce|降維", "vnn|n", { cv::Mat d; cv::reduce(a.mat(0), d, a.integer(1), a.integer(2), a.integer(3, -1)); return a.out(d); });
    FN(cv, "sort|排序", "vn", { cv::Mat d; cv::sort(a.mat(0), d, a.integer(1)); return a.out(d); });
    FN(cv, "sortIdx|排序索引", "vn", { cv::Mat d; cv::sortIdx(a.mat(0), d, a.integer(1)); return a.out(d); });

    // Statistics.
    FN(cv, "countNonZero|非零計數", "v", { return a.out(cv::countNonZero(a.mat(0))); });
    FN(cv, "findNonZero|非零位置", "v", { std::vector<cv::Point> points; cv::findNonZero(a.mat(0), points); return a.out(points); });
    FN(cv, "sum|總和", "v", { return a.out(cv::sum(a.mat(0))); });
    FN(cv, "mean|平均", "v|v", { return a.out(cv::mean(a.mat(0), a.mat(1, cv::Mat()))); });
    FN(cv, "meanStdDev|平均與標準差", "v|v", {
        cv::Scalar mean, deviation;
        cv::meanStdDev(a.mat(0), mean, deviation, a.mat(1, cv::Mat()));
        return a.tuple(mean, deviation);
    });
    FN(cv, "minMaxLoc|最小最大位置", "v|v", {
        double low, high;
        cv::Point lowAt, highAt;
        cv::minMaxLoc(a.mat(0), &low, &high, &lowAt, &highAt, a.mat(1, cv::Mat()));
        return a.tuple(low, high, lowAt, highAt);
    });
    FN(cv, "norm|範數", "v|vn", {
        if (a.has(1) && a.arg(1).type != AQUA_INT && !aqua_is_number(a.arg(1)))
            return a.out(cv::norm(a.mat(0), a.mat(1), a.integer(2, cv::NORM_L2)));
        return a.out(cv::norm(a.mat(0), a.integer(1, cv::NORM_L2)));
    });
    FN(cv, "PSNR|峰值訊噪比", "vv", { return a.out(cv::PSNR(a.mat(0), a.mat(1))); });
    FN(cv, "calcCovarMatrix|共變異數矩陣", "vn|n", {
        cv::Mat covariance, mean;
        cv::calcCovarMatrix(a.mat(0), covariance, mean, a.integer(1), a.integer(2, CV_64F));
        return a.tuple(covariance, mean);
    });
    FN(cv, "PCACompute|主成分分析", "v|n", {
        cv::Mat mean, vectors, values;
        cv::PCACompute(a.mat(0), mean, vectors, values, a.integer(1, 0));
        return a.tuple(mean, vectors, values);
    });

    // Linear algebra and transforms.
    FN(cv, "gemm|矩陣乘加", "vvnvn|n", {
        cv::Mat d;
        cv::gemm(a.mat(0), a.mat(1), a.number(2), a.mat(3), a.number(4), d, a.integer(5, 0));
        return a.out(d);
    });
    FN(cv, "solve|解線性方程", "vv|n", {
        cv::Mat d;
        bool ok = cv::solve(a.mat(0), a.mat(1), d, a.integer(2, cv::DECOMP_LU));
        return a.tuple(ok, d);
    });
    FN(cv, "invert|反矩陣", "v|n", {
        cv::Mat d;
        double result = cv::invert(a.mat(0), d, a.integer(1, cv::DECOMP_LU));
        return a.tuple(result, d);
    });
    FN(cv, "determinant|行列式", "v", { return a.out(cv::determinant(a.mat(0))); });
    FN(cv, "trace|跡", "v", { return a.out(cv::trace(a.mat(0))); });
    FN(cv, "eigen|特徵值", "v", {
        cv::Mat values, vectors;
        bool ok = cv::eigen(a.mat(0), values, vectors);
        return a.tuple(ok, values, vectors);
    });
    FN(cv, "SVDecomp|奇異值分解", "v|n", {
        cv::Mat w, u, vt;
        cv::SVDecomp(a.mat(0), w, u, vt, a.integer(1, 0));
        return a.tuple(w, u, vt);
    });
    FN(cv, "transform|線性轉換", "vv", { cv::Mat d; cv::transform(a.mat(0), d, a.mat(1)); return a.out(d); });
    FN(cv, "perspectiveTransform|透視點轉換", "vv", {
        std::vector<cv::Point2f> points;
        cv::perspectiveTransform(a.points2f(0), points, a.mat(1));
        return a.out(points);
    });
    FN(cv, "dft|離散傅立葉轉換", "v|nn", { cv::Mat d; cv::dft(a.mat(0), d, a.integer(1, 0), a.integer(2, 0)); return a.out(d); });
    FN(cv, "idft|反離散傅立葉轉換", "v|nn", { cv::Mat d; cv::idft(a.mat(0), d, a.integer(1, 0), a.integer(2, 0)); return a.out(d); });
    FN(cv, "dct|離散餘弦轉換", "v|n", { cv::Mat d; cv::dct(a.mat(0), d, a.integer(1, 0)); return a.out(d); });
    FN(cv, "idct|反離散餘弦轉換", "v|n", { cv::Mat d; cv::idct(a.mat(0), d, a.integer(1, 0)); return a.out(d); });
    FN(cv, "mulSpectrums|頻譜相乘", "vvn|b", { cv::Mat d; cv::mulSpectrums(a.mat(0), a.mat(1), d, a.integer(2), a.flag(3)); return a.out(d); });
    FN(cv, "getOptimalDFTSize|最佳DFT尺寸", "n", { return a.out(cv::getOptimalDFTSize(a.integer(0))); });
    FN(cv, "kmeans|K平均分群", "vnvvnn|v", {
        cv::Mat labels = a.mat(2, cv::Mat()), centers;
        double compactness = cv::kmeans(a.mat(0), a.integer(1), labels, a.criteria(3, cv::TermCriteria()), a.integer(4), a.integer(5), centers);
        return a.tuple(compactness, labels, centers);
    });

    // Random numbers.
    FN(cv, "setRNGSeed|設定亂數種子", "n", { cv::setRNGSeed(a.integer(0)); return a.none(); });
    FN(cv, "randu|均勻亂數", "vvv", { cv::randu(a.target(0), a.scalar(1), a.scalar(2)); return a.arg(0); });
    FN(cv, "randn|常態亂數", "vvv", { cv::randn(a.target(0), a.scalar(1), a.scalar(2)); return a.arg(0); });
    FN(cv, "randShuffle|隨機打亂", "v", { cv::randShuffle(a.target(0)); return a.arg(0); });

    // System.
    FN(cv, "getVersionString|版本", "", { return a.out(cv::getVersionString()); });
    FN(cv, "getBuildInformation|建置資訊", "", { return a.out(cv::getBuildInformation()); });
    FN(cv, "getNumThreads|執行緒數", "", { return a.out(cv::getNumThreads()); });
    FN(cv, "setNumThreads|設定執行緒數", "n", { cv::setNumThreads(a.integer(0)); return a.none(); });
    FN(cv, "getNumberOfCPUs|處理器數", "", { return a.out(cv::getNumberOfCPUs()); });
    FN(cv, "useOptimized|已最佳化", "", { return a.out(cv::useOptimized()); });
    FN(cv, "setUseOptimized|設定最佳化", "b", { cv::setUseOptimized(a.flag(0)); return a.none(); });
    FN(cv, "getTickCount|計時刻度", "", { return a.out(double(cv::getTickCount())); });
    FN(cv, "getTickFrequency|計時頻率", "", { return a.out(cv::getTickFrequency()); });
    FN(cv, "getCPUFeaturesLine|處理器特性", "", { return a.out(cv::getCPUFeaturesLine()); });
}

} // namespace acv
