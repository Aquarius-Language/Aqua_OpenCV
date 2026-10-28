// Argument and result conversion, KeyPoint and DMatch records, and the library's entry point.
// 引數與結果轉換、KeyPoint／DMatch 紀錄及函式庫進入點。
#include "binding.hpp"
#include <opencv2/imgproc.hpp>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iterator>

namespace acv {

void Args::fail(const char *format, ...) const {
    va_list arguments;
    va_start(arguments, format);
    aqua_vfail(call, format, arguments);
    va_end(arguments);
    throw Failed{};
}

double Args::number(unsigned i) const {
    aqua_value v = aqua_arg(call, i);
    if (v.type == AQUA_BOOL) return v.number;
    if (!aqua_is_number(v)) fail("argument %u must be a number, not %s", i + 1, aqua_type_name(v));
    return v.number;
}

int Args::integer(unsigned i) const {
    double n = number(i);
    if (n < -2147483648.0 || n > 2147483647.0) fail("argument %u is out of the integer range", i + 1);
    return int(n);
}

bool Args::flag(unsigned i, bool fallback) const {
    if (!has(i)) return fallback;
    aqua_value v = aqua_arg(call, i);
    if (v.type == AQUA_BOOL || aqua_is_number(v)) return v.number != 0;
    fail("argument %u must be a boolean", i + 1);
}

std::string Args::text(unsigned i) const {
    aqua_value v = aqua_arg(call, i);
    if (v.type != AQUA_STRING) fail("argument %u must be a string, not %s", i + 1, aqua_type_name(v));
    aqua_str s = aqua_str_of(call, v);
    return std::string(s.data, s.length);
}

aqua_value Args::field(unsigned i, const char *names) const {
    aqua_value result = aqua_null();
    if (!has(i)) return result;
    aqua_value hash = aqua_arg(call, i);
    if (hash.type != AQUA_HASH && hash.type != AQUA_STRUCT) fail("argument %u must be a hash of parameters", i + 1);
    aqua_hash_field(rt, hash, names, &result);
    return result;
}

// Shape of a nested numeric array: up to rows x columns x channels.
static void shape_of(aqua_runtime *rt, aqua_value v, std::vector<uint32_t> &shape) {
    while (v.type == AQUA_ARRAY) {
        uint32_t n = aqua_array_length(rt, v);
        shape.push_back(n);
        if (n == 0) return;
        v = aqua_array_get(rt, v, 0);
    }
}

static bool flatten(aqua_runtime *rt, aqua_value v, size_t depth, const std::vector<uint32_t> &shape, double *&out) {
    if (depth == shape.size()) {
        if (!aqua_is_number(v) && v.type != AQUA_BOOL) return false;
        *out++ = v.number;
        return true;
    }
    if (v.type != AQUA_ARRAY || aqua_array_length(rt, v) != shape[depth]) return false;
    if (depth + 1 == shape.size()) {
        if (!aqua_array_read_f64(rt, v, 0, shape[depth], out)) return false;
        out += shape[depth];
        return true;
    }
    for (uint32_t k = 0; k < shape[depth]; k++)
        if (!flatten(rt, aqua_array_get(rt, v, k), depth + 1, shape, out)) return false;
    return true;
}

// A nested array of numbers as a CV_64F Mat: [n] is n x 1 (as cv2 reads a 1-D array), [r][c] is r x c and [r][c][k]
// is r x c with k channels.
static cv::Mat from_array(const Args &a, aqua_value v, unsigned i) {
    std::vector<uint32_t> shape;
    shape_of(a.rt, v, shape);
    if (shape.empty() || shape.size() > 3) a.fail("argument %u must be a Mat or an array of 1 to 3 dimensions", i + 1);
    for (uint32_t n : shape)
        if (n == 0) return cv::Mat();
    int rows = int(shape[0]), cols = shape.size() > 1 ? int(shape[1]) : 1, channels = shape.size() > 2 ? int(shape[2]) : 1;
    if (channels > CV_CN_MAX) a.fail("argument %u has more than %d channels", i + 1, CV_CN_MAX);
    cv::Mat m(rows, cols, CV_64FC(channels));
    double *cursor = m.ptr<double>();
    if (!flatten(a.rt, v, 0, shape, cursor)) a.fail("argument %u must be a rectangular array of numbers", i + 1);
    return m;
}

cv::Mat Args::mat(unsigned i) const {
    aqua_value v = aqua_arg(call, i);
    if (cv::Mat *m = static_cast<cv::Mat *>(aqua_object_data(rt, v, Kind<cv::Mat>::cls))) return *m;
    if (aqua_image *image = aqua_image_of(rt, v)) {
        cv::Mat bgr;
        cv::cvtColor(cv::Mat(image->height, image->width, CV_8UC4, image->rgba), bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    }
    if (v.type == AQUA_ARRAY) return from_array(*this, v, i);
    if (aqua_is_number(v)) return cv::Mat(1, 1, CV_64F, cv::Scalar(v.number));
    fail("argument %u must be a Mat, an image or an array, not %s", i + 1, aqua_type_name(v));
}

cv::Mat &Args::target(unsigned i) const {
    cv::Mat *m = static_cast<cv::Mat *>(aqua_object_data(rt, aqua_arg(call, i), Kind<cv::Mat>::cls));
    if (!m) fail("argument %u must be a Mat (it is modified in place)", i + 1);
    return *m;
}

std::vector<cv::Mat> Args::mats(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) return {mat(i)};
    std::vector<cv::Mat> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        aqua_value item = aqua_array_get(rt, list, k);
        if (cv::Mat *m = static_cast<cv::Mat *>(aqua_object_data(rt, item, Kind<cv::Mat>::cls))) result.push_back(*m);
        else if (aqua_image *image = aqua_image_of(rt, item)) {
            cv::Mat bgr;
            cv::cvtColor(cv::Mat(image->height, image->width, CV_8UC4, image->rgba), bgr, cv::COLOR_RGBA2BGR);
            result.push_back(bgr);
        } else result.push_back(from_array(*this, item, i));
    }
    return result;
}

std::vector<double> Args::doubles(unsigned i) const {
    aqua_value v = aqua_arg(call, i);
    if (aqua_is_number(v)) return {v.number};
    if (v.type != AQUA_ARRAY) {
        cv::Mat m = mat(i);
        std::vector<double> result;
        m.reshape(1, 1).convertTo(result, CV_64F);
        return result;
    }
    std::vector<double> result(aqua_array_length(rt, v));
    if (!aqua_array_read_f64(rt, v, 0, uint32_t(result.size()), result.data())) fail("argument %u must be an array of numbers", i + 1);
    return result;
}

std::vector<int> Args::ints(unsigned i) const {
    std::vector<int> result;
    for (double d : doubles(i)) result.push_back(cvRound(d));
    return result;
}

std::vector<float> Args::floats(unsigned i) const {
    std::vector<float> result;
    for (double d : doubles(i)) result.push_back(float(d));
    return result;
}

std::vector<std::string> Args::texts(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type == AQUA_STRING) return {text(i)};
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of strings", i + 1);
    std::vector<std::string> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        aqua_value item = aqua_array_get(rt, list, k);
        if (item.type != AQUA_STRING) fail("argument %u must be an array of strings", i + 1);
        aqua_str s = aqua_str_of(call, item);
        result.emplace_back(s.data, s.length);
    }
    return result;
}

cv::Size Args::size(unsigned i) const {
    auto n = doubles(i);
    if (n.size() == 1) return {cvRound(n[0]), cvRound(n[0])};
    if (n.size() != 2) fail("argument %u must be a size [width, height]", i + 1);
    return {cvRound(n[0]), cvRound(n[1])};
}

cv::Point2f Args::point2f(unsigned i) const {
    auto n = doubles(i);
    if (n.size() != 2) fail("argument %u must be a point [x, y]", i + 1);
    return {float(n[0]), float(n[1])};
}

cv::Point Args::point(unsigned i) const {
    cv::Point2f p = point2f(i);
    return {cvRound(p.x), cvRound(p.y)};
}

cv::Rect Args::rect(unsigned i) const {
    auto n = doubles(i);
    if (n.size() != 4) fail("argument %u must be a rectangle [x, y, width, height]", i + 1);
    return {cvRound(n[0]), cvRound(n[1]), cvRound(n[2]), cvRound(n[3])};
}

cv::Scalar Args::scalar(unsigned i) const {
    auto n = doubles(i);
    if (n.empty() || n.size() > 4) fail("argument %u must be a number or [b, g, r, a]", i + 1);
    cv::Scalar s(n[0], n.size() > 1 ? n[1] : 0, n.size() > 2 ? n[2] : 0, n.size() > 3 ? n[3] : 0);
    return s;
}

cv::TermCriteria Args::criteria(unsigned i, cv::TermCriteria fallback) const {
    if (!has(i)) return fallback;
    auto n = doubles(i);
    if (n.size() != 3) fail("argument %u must be criteria [type, maxCount, epsilon]", i + 1);
    return cv::TermCriteria(int(n[0]), int(n[1]), n[2]);
}

// Points from [[x, y], ...] or from a Mat of N rows (N x 2, or N x 1 with two channels).
std::vector<cv::Point2d> Args::pairs(aqua_value list, unsigned i) const {
    std::vector<cv::Point2d> result;
    if (list.type == AQUA_ARRAY) {
        uint32_t n = aqua_array_length(rt, list);
        result.resize(n);
        for (uint32_t k = 0; k < n; k++) {
            double xy[2];
            aqua_value item = aqua_array_get(rt, list, k);
            if (item.type != AQUA_ARRAY || aqua_array_length(rt, item) != 2 || !aqua_array_read_f64(rt, item, 0, 2, xy))
                fail("argument %u must be an array of points [x, y]", i + 1);
            result[k] = {xy[0], xy[1]};
        }
        return result;
    }
    cv::Mat m = static_cast<cv::Mat *>(aqua_object_data(rt, list, Kind<cv::Mat>::cls)) ? mat(i) : cv::Mat();
    if (m.empty()) fail("argument %u must be an array of points [x, y] or a Mat", i + 1);
    cv::Mat d;
    m.reshape(1, int(m.total() * m.channels() / 2)).convertTo(d, CV_64F);
    if (d.cols != 2) fail("argument %u must hold two coordinates per point", i + 1);
    for (int r = 0; r < d.rows; r++) result.emplace_back(d.at<double>(r, 0), d.at<double>(r, 1));
    return result;
}

std::vector<cv::Point2f> Args::points2f(unsigned i) const {
    std::vector<cv::Point2f> result;
    for (auto p : pairs(aqua_arg(call, i), i)) result.emplace_back(float(p.x), float(p.y));
    return result;
}

std::vector<cv::Point> Args::points(unsigned i) const {
    std::vector<cv::Point> result;
    for (auto p : pairs(aqua_arg(call, i), i)) result.emplace_back(cvRound(p.x), cvRound(p.y));
    return result;
}

std::vector<cv::Point3f> Args::points3f(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    std::vector<cv::Point3f> result;
    if (list.type != AQUA_ARRAY) {
        cv::Mat d;
        mat(i).reshape(1, int(mat(i).total() * mat(i).channels() / 3)).convertTo(d, CV_32F);
        for (int r = 0; r < d.rows; r++) result.emplace_back(d.at<float>(r, 0), d.at<float>(r, 1), d.at<float>(r, 2));
        return result;
    }
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        double xyz[3];
        aqua_value item = aqua_array_get(rt, list, k);
        if (item.type != AQUA_ARRAY || aqua_array_length(rt, item) != 3 || !aqua_array_read_f64(rt, item, 0, 3, xyz))
            fail("argument %u must be an array of points [x, y, z]", i + 1);
        result.emplace_back(float(xyz[0]), float(xyz[1]), float(xyz[2]));
    }
    return result;
}

std::vector<std::vector<cv::Point>> Args::contours(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of contours", i + 1);
    std::vector<std::vector<cv::Point>> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        std::vector<cv::Point> contour;
        for (auto p : pairs(aqua_array_get(rt, list, k), i)) contour.emplace_back(cvRound(p.x), cvRound(p.y));
        result.push_back(std::move(contour));
    }
    return result;
}

std::vector<std::vector<cv::Point2f>> Args::contours2f(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of point arrays", i + 1);
    std::vector<std::vector<cv::Point2f>> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        std::vector<cv::Point2f> points;
        for (auto p : pairs(aqua_array_get(rt, list, k), i)) points.emplace_back(float(p.x), float(p.y));
        result.push_back(std::move(points));
    }
    return result;
}

std::vector<std::vector<cv::Point3f>> Args::contours3f(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of point arrays", i + 1);
    std::vector<std::vector<cv::Point3f>> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        aqua_value item = aqua_array_get(rt, list, k);
        std::vector<cv::Point3f> points;
        for (uint32_t j = 0, m = aqua_array_length(rt, item); j < m; j++) {
            double xyz[3];
            aqua_value p = aqua_array_get(rt, item, j);
            if (p.type != AQUA_ARRAY || aqua_array_length(rt, p) != 3 || !aqua_array_read_f64(rt, p, 0, 3, xyz))
                fail("argument %u must hold points [x, y, z]", i + 1);
            points.emplace_back(float(xyz[0]), float(xyz[1]), float(xyz[2]));
        }
        result.push_back(std::move(points));
    }
    return result;
}

std::vector<cv::Rect> Args::rects(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of rectangles", i + 1);
    std::vector<cv::Rect> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        double r[4];
        aqua_value item = aqua_array_get(rt, list, k);
        if (item.type != AQUA_ARRAY || aqua_array_length(rt, item) != 4 || !aqua_array_read_f64(rt, item, 0, 4, r))
            fail("argument %u must be an array of rectangles [x, y, width, height]", i + 1);
        result.emplace_back(cvRound(r[0]), cvRound(r[1]), cvRound(r[2]), cvRound(r[3]));
    }
    return result;
}

// KeyPoint and DMatch are hashes with cv2's attribute names (and Chinese aliases), so thousands per frame cost guest
// memory the collector sees instead of native objects.
static double record_number(const Args &a, aqua_value record, const char *names, unsigned i, double fallback) {
    aqua_value v = aqua_null();
    if (record.type != AQUA_HASH && record.type != AQUA_STRUCT) a.fail("argument %u must hold KeyPoint or DMatch records", i + 1);
    if (!aqua_hash_field(a.rt, record, names, &v) || aqua_is_nullish(v)) return fallback;
    if (!aqua_is_number(v)) a.fail("argument %u: '%s' must be a number", i + 1, names);
    return v.number;
}

static cv::KeyPoint keypoint_of(const Args &a, aqua_value record, unsigned i) {
    aqua_value pt = aqua_null();
    double xy[2];
    if (record.type != AQUA_HASH && record.type != AQUA_STRUCT) a.fail("argument %u must be an array of KeyPoint records", i + 1);
    if (!aqua_hash_field(a.rt, record, "pt|位置", &pt) || pt.type != AQUA_ARRAY || aqua_array_length(a.rt, pt) != 2 || !aqua_array_read_f64(a.rt, pt, 0, 2, xy))
        a.fail("argument %u: a KeyPoint needs pt [x, y]", i + 1);
    return cv::KeyPoint(float(xy[0]), float(xy[1]), float(record_number(a, record, "size|大小", i, 1)), float(record_number(a, record, "angle|角度", i, -1)),
        float(record_number(a, record, "response|響應", i, 0)), int(record_number(a, record, "octave|層級", i, 0)), int(record_number(a, record, "class_id|類別編號", i, -1)));
}

static cv::DMatch match_of(const Args &a, aqua_value record, unsigned i) {
    return cv::DMatch(int(record_number(a, record, "queryIdx|查詢索引", i, -1)), int(record_number(a, record, "trainIdx|訓練索引", i, -1)),
        int(record_number(a, record, "imgIdx|影像索引", i, -1)), float(record_number(a, record, "distance|距離", i, 0)));
}

std::vector<cv::KeyPoint> Args::keypoints(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of KeyPoint records", i + 1);
    std::vector<cv::KeyPoint> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) result.push_back(keypoint_of(*this, aqua_array_get(rt, list, k), i));
    return result;
}

std::vector<cv::DMatch> Args::matches(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of DMatch records", i + 1);
    std::vector<cv::DMatch> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        aqua_value item = aqua_array_get(rt, list, k);
        // knnMatch results nest one level deeper; drawMatches accepts the first match of each.
        if (item.type == AQUA_ARRAY) item = aqua_array_length(rt, item) ? aqua_array_get(rt, item, 0) : aqua_null();
        if (!aqua_is_nullish(item)) result.push_back(match_of(*this, item, i));
    }
    return result;
}

std::vector<std::vector<cv::DMatch>> Args::knn_matches(unsigned i) const {
    aqua_value list = aqua_arg(call, i);
    if (list.type != AQUA_ARRAY) fail("argument %u must be an array of DMatch arrays", i + 1);
    std::vector<std::vector<cv::DMatch>> result;
    for (uint32_t k = 0, n = aqua_array_length(rt, list); k < n; k++) {
        aqua_value row = aqua_array_get(rt, list, k);
        std::vector<cv::DMatch> group;
        for (uint32_t j = 0, m = row.type == AQUA_ARRAY ? aqua_array_length(rt, row) : 0; j < m; j++) group.push_back(match_of(*this, aqua_array_get(rt, row, j), i));
        result.push_back(std::move(group));
    }
    return result;
}

std::vector<uchar> Args::bytes(unsigned i) const {
    aqua_value v = aqua_arg(call, i);
    if (v.type != AQUA_ARRAY) {
        cv::Mat m = mat(i);
        if (m.depth() != CV_8U) fail("argument %u must be bytes", i + 1);
        return std::vector<uchar>(m.datastart, m.dataend);
    }
    uint32_t n = aqua_array_length(rt, v);
    if (n > 512u * 1024 * 1024) fail("argument %u exceeds 512 MiB", i + 1);
    std::vector<uchar> result(n);
    if (!aqua_array_read_u8(rt, v, 0, n, result.data())) fail("argument %u must be bytes 0..255", i + 1);
    return result;
}

aqua_value Args::out(const cv::Mat &m) const { return make(rt, m); }
aqua_value Args::out(cv::Point p) const { double v[] = {double(p.x), double(p.y)}; return aqua_array_from_f64(rt, v, 2, AQUA_INT); }
aqua_value Args::out(cv::Point2f p) const { double v[] = {p.x, p.y}; return aqua_array_from_f64(rt, v, 2, AQUA_DOUBLE); }
aqua_value Args::out(cv::Point3f p) const { double v[] = {p.x, p.y, p.z}; return aqua_array_from_f64(rt, v, 3, AQUA_DOUBLE); }
aqua_value Args::out(cv::Size s) const { double v[] = {double(s.width), double(s.height)}; return aqua_array_from_f64(rt, v, 2, AQUA_INT); }
aqua_value Args::out(cv::Size2f s) const { double v[] = {s.width, s.height}; return aqua_array_from_f64(rt, v, 2, AQUA_DOUBLE); }
aqua_value Args::out(cv::Rect r) const {
    double v[] = {double(r.x), double(r.y), double(r.width), double(r.height)};
    return aqua_array_from_f64(rt, v, 4, AQUA_INT);
}
aqua_value Args::out(cv::Rect2d r) const { double v[] = {r.x, r.y, r.width, r.height}; return aqua_array_from_f64(rt, v, 4, AQUA_DOUBLE); }
aqua_value Args::out(const cv::RotatedRect &r) const { return tuple(r.center, r.size, double(r.angle)); }
aqua_value Args::out(const cv::Scalar &s) const { return aqua_array_from_f64(rt, s.val, 4, AQUA_DOUBLE); }
aqua_value Args::out(const cv::KeyPoint &k) const {
    aqua_value hash = aqua_hash_new(rt, 12);
    aqua_value pt = out(k.pt);
    const std::pair<const char *, aqua_value> fields[] = {{"pt", pt}, {"size", aqua_double(k.size)}, {"angle", aqua_double(k.angle)},
        {"response", aqua_double(k.response)}, {"octave", aqua_int(k.octave)}, {"class_id", aqua_int(k.class_id)}, {"位置", pt},
        {"大小", aqua_double(k.size)}, {"角度", aqua_double(k.angle)}, {"響應", aqua_double(k.response)}, {"層級", aqua_int(k.octave)},
        {"類別編號", aqua_int(k.class_id)}};
    for (auto &f : fields) aqua_hash_set_field(rt, hash, f.first, f.second);
    return hash;
}
aqua_value Args::out(const cv::DMatch &m) const {
    aqua_value hash = aqua_hash_new(rt, 8);
    const std::pair<const char *, aqua_value> fields[] = {{"queryIdx", aqua_int(m.queryIdx)}, {"trainIdx", aqua_int(m.trainIdx)},
        {"imgIdx", aqua_int(m.imgIdx)}, {"distance", aqua_double(m.distance)}, {"查詢索引", aqua_int(m.queryIdx)},
        {"訓練索引", aqua_int(m.trainIdx)}, {"影像索引", aqua_int(m.imgIdx)}, {"距離", aqua_double(m.distance)}};
    for (auto &f : fields) aqua_hash_set_field(rt, hash, f.first, f.second);
    return hash;
}
aqua_value Args::out(const cv::Moments &m) const {
    aqua_value hash = aqua_hash_new(rt, 24);
    const std::pair<const char *, double> fields[] = {
        {"m00", m.m00}, {"m10", m.m10}, {"m01", m.m01}, {"m20", m.m20}, {"m11", m.m11}, {"m02", m.m02}, {"m30", m.m30},
        {"m21", m.m21}, {"m12", m.m12}, {"m03", m.m03}, {"mu20", m.mu20}, {"mu11", m.mu11}, {"mu02", m.mu02},
        {"mu30", m.mu30}, {"mu21", m.mu21}, {"mu12", m.mu12}, {"mu03", m.mu03}, {"nu20", m.nu20}, {"nu11", m.nu11},
        {"nu02", m.nu02}, {"nu30", m.nu30}, {"nu21", m.nu21}, {"nu12", m.nu12}, {"nu03", m.nu03}};
    for (auto &f : fields) aqua_hash_set_field(rt, hash, f.first, aqua_double(f.second));
    return hash;
}

std::vector<uchar> read_resource(const Args &a, const std::string &path) {
    size_t length = 0;
    if (const uint8_t *asset = aqua_asset(a.rt, path.c_str(), &length)) return std::vector<uchar>(asset, asset + length);
#if !defined(AQUA_OS_WEB)
    std::ifstream file(path, std::ios::binary);
    if (file) return std::vector<uchar>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
#endif
    a.fail("cannot read '%s': it is neither a packaged asset nor a readable file", path.c_str());
}

} // namespace acv

using namespace acv;

AQUA_LIBRARY(opencv) {
    aqua_class *cv = aqua_define_module(library, "OpenCV|cv|電腦視覺");
    register_core(library, cv);
    register_imgproc(library, cv);
    register_features(library, cv);
    register_calib3d(library, cv);
    register_video(library, cv);
    register_objdetect(library, cv);
    register_photo(library, cv);
    register_dnn(library, cv);
    register_ml(library, cv);
}
