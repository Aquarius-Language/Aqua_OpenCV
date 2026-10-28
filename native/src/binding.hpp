// Shared plumbing of the OpenCV binding: argument conversion, result conversion and native object classes.
// OpenCV 綁定的共用基礎：引數轉換、結果轉換與原生物件類別。
//
// The binding follows OpenCV's own Python (cv2) conventions: the same function and constant names, the same argument
// order, optional trailing arguments, outputs returned instead of written into arguments (several outputs as an
// array), points as [x, y], sizes as [width, height], rectangles as [x, y, width, height] and scalars as a number or
// [b, g, r, a]. Images and matrices are cv::Mat objects owned by the script (released by the collector or delete()).
#pragma once
#include <aqua/aqua.h>
#include <aqua/image.h>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace acv {

// Thrown after aqua_fail: the call already carries its error.
struct Failed {};

// A native class per C++ type held by script objects. Several script classes may hold one C++ type (ORB and SIFT
// both hold cv::Ptr<cv::Feature2D>); Kind<T>::classes lists them all for argument checks.
template <class T> struct Kind {
    static inline aqua_class *cls = nullptr;
    static inline std::vector<aqua_class *> classes;
};
template <class T> void finalize(aqua_runtime *, void *data) { delete static_cast<T *>(data); }
template <class T> aqua_class *define_class(aqua_library *library, const char *name) {
    aqua_class *cls = aqua_define_class(library, name, finalize<T>);
    if (!Kind<T>::cls) Kind<T>::cls = cls;
    Kind<T>::classes.push_back(cls);
    return cls;
}
template <class T> aqua_value make(aqua_runtime *rt, T value, aqua_class *cls = Kind<T>::cls) {
    return aqua_object_new(rt, cls, new T(std::move(value)));
}

// The arguments of one call, with OpenCV-style conversions. Missing or null optional arguments take the default.
class Args {
public:
    explicit Args(aqua_call *call) : call(call), rt(aqua_call_runtime(call)) {}
    aqua_call *const call;
    aqua_runtime *const rt;

    [[noreturn]] void fail(const char *format, ...) const;
    bool has(unsigned i) const { return i < aqua_argc(call) && !aqua_is_nullish(aqua_arg(call, i)); }
    aqua_value value(unsigned i) const { return aqua_arg(call, i); }

    double number(unsigned i) const;
    double number(unsigned i, double fallback) const { return has(i) ? number(i) : fallback; }
    int integer(unsigned i) const;
    int integer(unsigned i, int fallback) const { return has(i) ? integer(i) : fallback; }
    bool flag(unsigned i, bool fallback = false) const;
    std::string text(unsigned i) const;
    std::string text(unsigned i, const std::string &fallback) const { return has(i) ? text(i) : fallback; }

    // A Mat argument: a Mat object, an image object (RGBA becomes BGR), or a (nested) array of numbers.
    cv::Mat mat(unsigned i) const;
    cv::Mat mat(unsigned i, const cv::Mat &fallback) const { return has(i) ? mat(i) : fallback; }
    // The Mat object itself, for functions that draw or write into it (cv2 modifies the array in place).
    cv::Mat &target(unsigned i) const;
    std::vector<cv::Mat> mats(unsigned i) const;

    cv::Size size(unsigned i) const;
    cv::Size size(unsigned i, cv::Size fallback) const { return has(i) ? size(i) : fallback; }
    cv::Point point(unsigned i) const;
    cv::Point2f point2f(unsigned i) const;
    cv::Point2f point2f(unsigned i, cv::Point2f fallback) const { return has(i) ? point2f(i) : fallback; }
    cv::Rect rect(unsigned i) const;
    cv::Scalar scalar(unsigned i) const;
    cv::Scalar scalar(unsigned i, cv::Scalar fallback) const { return has(i) ? scalar(i) : fallback; }
    cv::TermCriteria criteria(unsigned i, cv::TermCriteria fallback) const;
    std::vector<int> ints(unsigned i) const;
    std::vector<float> floats(unsigned i) const;
    std::vector<double> doubles(unsigned i) const;
    std::vector<std::string> texts(unsigned i) const;
    std::vector<cv::Point> points(unsigned i) const;
    std::vector<cv::Point2f> points2f(unsigned i) const;
    std::vector<cv::Point3f> points3f(unsigned i) const;
    std::vector<std::vector<cv::Point>> contours(unsigned i) const;
    std::vector<std::vector<cv::Point2f>> contours2f(unsigned i) const;
    std::vector<std::vector<cv::Point3f>> contours3f(unsigned i) const;
    std::vector<cv::Rect> rects(unsigned i) const;
    std::vector<cv::KeyPoint> keypoints(unsigned i) const;
    std::vector<cv::DMatch> matches(unsigned i) const;
    std::vector<std::vector<cv::DMatch>> knn_matches(unsigned i) const;
    std::vector<uchar> bytes(unsigned i) const;
    // A hash argument's field by "english|中文" names, or null.
    aqua_value field(unsigned i, const char *names) const;

    // The object a method was called on.
    template <class T> T &self() const { return *static_cast<T *>(aqua_self(call)); }
    // A native object argument of one of T's classes.
    template <class T> T &object(unsigned i) const {
        aqua_value v = aqua_arg(call, i);
        aqua_class *cls = aqua_object_class(rt, v);
        for (aqua_class *candidate : Kind<T>::classes)
            if (cls == candidate) return *static_cast<T *>(aqua_object_data(rt, v, cls));
        fail("argument %u has the wrong type", i + 1);
    }

    // Results.
    aqua_value out(const cv::Mat &m) const;
    aqua_value out(double n) const { return aqua_double(n); }
    aqua_value out(int n) const { return aqua_int(n); }
    aqua_value out(bool b) const { return aqua_bool(b); }
    aqua_value out(const std::string &s) const { return aqua_text(rt, s.data(), s.size()); }
    aqua_value out(cv::Point p) const;
    aqua_value out(cv::Point2f p) const;
    aqua_value out(cv::Point3f p) const;
    aqua_value out(cv::Size s) const;
    aqua_value out(cv::Size2f s) const;
    aqua_value out(cv::Rect r) const;
    aqua_value out(cv::Rect2d r) const;
    aqua_value out(const cv::RotatedRect &r) const;
    aqua_value out(const cv::Scalar &s) const;
    aqua_value out(const cv::KeyPoint &k) const;
    aqua_value out(const cv::DMatch &m) const;
    aqua_value out(const cv::Moments &m) const;
    template <class T> aqua_value out(const std::vector<T> &items) const {
        aqua_value list = aqua_array_new(rt, uint32_t(items.size()));
        for (size_t i = 0; i < items.size(); i++) aqua_array_set(rt, list, uint32_t(i), out(items[i]));
        return list;
    }
    aqua_value out(const std::vector<uchar> &items) const { return aqua_array_from_u8(rt, items.data(), uint32_t(items.size())); }
    aqua_value out(const std::vector<int> &items) const { return aqua_array_from_i32(rt, items.data(), uint32_t(items.size())); }
    // Several results, as cv2 returns a tuple: [retval, dst, ...].
    template <class... T> aqua_value tuple(const T &...items) const {
        aqua_value values[] = {out(items)...};
        aqua_value list = aqua_array_new(rt, uint32_t(sizeof...(T)));
        for (uint32_t i = 0; i < sizeof...(T); i++) aqua_array_set(rt, list, i, values[i]);
        return list;
    }
    aqua_value self_value() const { return aqua_self_value(call); }
    aqua_value arg(unsigned i) const { return aqua_arg(call, i); }
    aqua_value none() const { return aqua_null(); }

private:
    std::vector<cv::Point2d> pairs(aqua_value list, unsigned i) const;
};

// Reads a packaged asset, or a file on targets with a file system.
std::vector<uchar> read_resource(const Args &a, const std::string &path);

// Wraps a binding function as a library function: OpenCV and C++ exceptions become script errors.
template <auto F> void call(aqua_call *c) {
    try {
        Args a(c);
        aqua_return(c, F(a));
    } catch (const Failed &) {
    } catch (const cv::Exception &e) {
        aqua_fail(c, "OpenCV: %s", e.what());
    } catch (const std::exception &e) {
        aqua_fail(c, "OpenCV: %s", e.what());
    }
}

// FN(owner, "name|名稱", "signature", { body using Args &a; returns a value }) defines one function or method.
#define FN(owner, names, signature, ...) \
    aqua_define_function(owner, names, ::acv::call<[](::acv::Args &a) -> aqua_value __VA_ARGS__>, signature)

// Module parts, one per OpenCV module (see the .cpp files of the same names).
void register_core(aqua_library *library, aqua_class *cv);
void register_imgproc(aqua_library *library, aqua_class *cv);
void register_features(aqua_library *library, aqua_class *cv);
void register_calib3d(aqua_library *library, aqua_class *cv);
void register_video(aqua_library *library, aqua_class *cv);
void register_objdetect(aqua_library *library, aqua_class *cv);
void register_photo(aqua_library *library, aqua_class *cv);
void register_dnn(aqua_library *library, aqua_class *cv);
void register_ml(aqua_library *library, aqua_class *cv);

} // namespace acv
