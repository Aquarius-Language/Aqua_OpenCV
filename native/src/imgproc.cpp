// imgproc: color, geometry, filters, thresholds, morphology, histograms, contours and shapes, Hough transforms,
// corners, segmentation and drawing. Drawing functions draw into their Mat and return it, as cv2 does.
// imgproc：色彩、幾何、濾波、二值化、形態學、直方圖、輪廓與形狀、霍夫轉換、角點、分割及繪圖。
#include "binding.hpp"
#include <opencv2/imgproc.hpp>
#include <climits>
#include <iterator>

namespace acv {
namespace {

// A CLAHE object (cv2.createCLAHE) and a line segment detector (cv2.createLineSegmentDetector).
using ClaheBox = cv::Ptr<cv::CLAHE>;
using LsdBox = cv::Ptr<cv::LineSegmentDetector>;

void register_geometry(aqua_class *cv) {
    FN(cv, "cvtColor|色彩轉換", "vn|n", { cv::Mat d; cv::cvtColor(a.mat(0), d, a.integer(1), a.integer(2, 0)); return a.out(d); });
    FN(cv, "resize|調整大小", "vv|nnn", {
        cv::Mat d;
        cv::Size size = a.has(1) ? a.size(1) : cv::Size();
        cv::resize(a.mat(0), d, size, a.number(2, 0), a.number(3, 0), a.integer(4, cv::INTER_LINEAR));
        return a.out(d);
    });
    FN(cv, "warpAffine|仿射變換", "vvv|nnv", {
        cv::Mat d;
        cv::warpAffine(a.mat(0), d, a.mat(1), a.size(2), a.integer(3, cv::INTER_LINEAR), a.integer(4, cv::BORDER_CONSTANT), a.scalar(5, cv::Scalar()));
        return a.out(d);
    });
    FN(cv, "warpPerspective|透視變換", "vvv|nnv", {
        cv::Mat d;
        cv::warpPerspective(a.mat(0), d, a.mat(1), a.size(2), a.integer(3, cv::INTER_LINEAR), a.integer(4, cv::BORDER_CONSTANT), a.scalar(5, cv::Scalar()));
        return a.out(d);
    });
    FN(cv, "warpPolar|極座標變換", "vvvnn", {
        cv::Mat d;
        cv::warpPolar(a.mat(0), d, a.size(1), a.point2f(2), a.number(3), a.integer(4));
        return a.out(d);
    });
    FN(cv, "remap|重映射", "vvvn|nv", {
        cv::Mat d;
        cv::remap(a.mat(0), d, a.mat(1), a.mat(2), a.integer(3), a.integer(4, cv::BORDER_CONSTANT), a.scalar(5, cv::Scalar()));
        return a.out(d);
    });
    FN(cv, "getRotationMatrix2D|旋轉矩陣", "vnn", { return a.out(cv::getRotationMatrix2D(a.point2f(0), a.number(1), a.number(2))); });
    FN(cv, "getAffineTransform|仿射矩陣", "vv", {
        auto from = a.points2f(0), to = a.points2f(1);
        if (from.size() != 3 || to.size() != 3) a.fail("getAffineTransform needs three points each");
        return a.out(cv::getAffineTransform(from.data(), to.data()));
    });
    FN(cv, "getPerspectiveTransform|透視矩陣", "vv|n", {
        auto from = a.points2f(0), to = a.points2f(1);
        if (from.size() != 4 || to.size() != 4) a.fail("getPerspectiveTransform needs four points each");
        return a.out(cv::getPerspectiveTransform(from.data(), to.data(), a.integer(2, cv::DECOMP_LU)));
    });
    FN(cv, "invertAffineTransform|反仿射矩陣", "v", { cv::Mat d; cv::invertAffineTransform(a.mat(0), d); return a.out(d); });
    FN(cv, "pyrDown|降採樣", "v|vn", {
        cv::Mat d;
        cv::pyrDown(a.mat(0), d, a.size(1, cv::Size()), a.integer(2, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "pyrUp|升採樣", "v|vn", {
        cv::Mat d;
        cv::pyrUp(a.mat(0), d, a.size(1, cv::Size()), a.integer(2, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "getRectSubPix|次像素擷取", "vvv", { cv::Mat d; cv::getRectSubPix(a.mat(0), a.size(1), a.point2f(2), d); return a.out(d); });
}

void register_filters(aqua_class *cv) {
    FN(cv, "blur|均值模糊", "vv|vn", {
        cv::Mat d;
        cv::blur(a.mat(0), d, a.size(1), a.has(2) ? a.point(2) : cv::Point(-1, -1), a.integer(3, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "boxFilter|方框濾波", "vnv|vbn", {
        cv::Mat d;
        cv::boxFilter(a.mat(0), d, a.integer(1), a.size(2), a.has(3) ? a.point(3) : cv::Point(-1, -1), a.flag(4, true), a.integer(5, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "GaussianBlur|高斯模糊", "vvn|nn", {
        cv::Mat d;
        cv::GaussianBlur(a.mat(0), d, a.size(1), a.number(2), a.number(3, 0), a.integer(4, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "medianBlur|中值濾波", "vn", { cv::Mat d; cv::medianBlur(a.mat(0), d, a.integer(1)); return a.out(d); });
    FN(cv, "bilateralFilter|雙邊濾波", "vnnn|n", {
        cv::Mat d;
        cv::bilateralFilter(a.mat(0), d, a.integer(1), a.number(2), a.number(3), a.integer(4, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "stackBlur|堆疊模糊", "vv", { cv::Mat d; cv::stackBlur(a.mat(0), d, a.size(1)); return a.out(d); });
    FN(cv, "filter2D|自訂卷積", "vnv|vnn", {
        cv::Mat d;
        cv::filter2D(a.mat(0), d, a.integer(1), a.mat(2), a.has(3) ? a.point(3) : cv::Point(-1, -1), a.number(4, 0), a.integer(5, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "sepFilter2D|可分離卷積", "vnvv|vnn", {
        cv::Mat d;
        cv::sepFilter2D(a.mat(0), d, a.integer(1), a.mat(2), a.mat(3), a.has(4) ? a.point(4) : cv::Point(-1, -1), a.number(5, 0), a.integer(6, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "Sobel|Sobel梯度", "vnnn|nnnn", {
        cv::Mat d;
        cv::Sobel(a.mat(0), d, a.integer(1), a.integer(2), a.integer(3), a.integer(4, 3), a.number(5, 1), a.number(6, 0), a.integer(7, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "Scharr|Scharr梯度", "vnnn|nnn", {
        cv::Mat d;
        cv::Scharr(a.mat(0), d, a.integer(1), a.integer(2), a.integer(3), a.number(4, 1), a.number(5, 0), a.integer(6, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "Laplacian|拉普拉斯", "vn|nnnn", {
        cv::Mat d;
        cv::Laplacian(a.mat(0), d, a.integer(1), a.integer(2, 1), a.number(3, 1), a.number(4, 0), a.integer(5, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "Canny|Canny邊緣", "vnn|nb", {
        cv::Mat d;
        cv::Canny(a.mat(0), d, a.number(1), a.number(2), a.integer(3, 3), a.flag(4));
        return a.out(d);
    });
    FN(cv, "getGaussianKernel|高斯核心", "nn|n", { return a.out(cv::getGaussianKernel(a.integer(0), a.number(1), a.integer(2, CV_64F))); });
    FN(cv, "getGaborKernel|Gabor核心", "vnnnn|nn", {
        return a.out(cv::getGaborKernel(a.size(0), a.number(1), a.number(2), a.number(3), a.number(4), a.number(5, CV_PI * 0.5), a.integer(6, CV_64F)));
    });
    FN(cv, "getStructuringElement|結構元素", "nv|v", {
        return a.out(cv::getStructuringElement(a.integer(0), a.size(1), a.has(2) ? a.point(2) : cv::Point(-1, -1)));
    });
    FN(cv, "erode|侵蝕", "vv|vnnv", {
        cv::Mat d;
        cv::erode(a.mat(0), d, a.mat(1), a.has(2) ? a.point(2) : cv::Point(-1, -1), a.integer(3, 1), a.integer(4, cv::BORDER_CONSTANT),
            a.has(5) ? a.scalar(5) : cv::morphologyDefaultBorderValue());
        return a.out(d);
    });
    FN(cv, "dilate|膨脹", "vv|vnnv", {
        cv::Mat d;
        cv::dilate(a.mat(0), d, a.mat(1), a.has(2) ? a.point(2) : cv::Point(-1, -1), a.integer(3, 1), a.integer(4, cv::BORDER_CONSTANT),
            a.has(5) ? a.scalar(5) : cv::morphologyDefaultBorderValue());
        return a.out(d);
    });
    FN(cv, "morphologyEx|形態學運算", "vnv|vnnv", {
        cv::Mat d;
        cv::morphologyEx(a.mat(0), d, a.integer(1), a.mat(2), a.has(3) ? a.point(3) : cv::Point(-1, -1), a.integer(4, 1),
            a.integer(5, cv::BORDER_CONSTANT), a.has(6) ? a.scalar(6) : cv::morphologyDefaultBorderValue());
        return a.out(d);
    });
    FN(cv, "threshold|二值化", "vnnn", {
        cv::Mat d;
        double used = cv::threshold(a.mat(0), d, a.number(1), a.number(2), a.integer(3));
        return a.tuple(used, d);
    });
    FN(cv, "adaptiveThreshold|自適應二值化", "vnnnnn", {
        cv::Mat d;
        cv::adaptiveThreshold(a.mat(0), d, a.number(1), a.integer(2), a.integer(3), a.integer(4), a.number(5));
        return a.out(d);
    });
    FN(cv, "integral|積分影像", "v|n", { cv::Mat d; cv::integral(a.mat(0), d, a.integer(1, -1)); return a.out(d); });
    FN(cv, "distanceTransform|距離轉換", "vnn|n", {
        cv::Mat d;
        cv::distanceTransform(a.mat(0), d, a.integer(1), a.integer(2), a.integer(3, CV_32F));
        return a.out(d);
    });
    FN(cv, "applyColorMap|偽彩色映射", "vv", {
        cv::Mat d;
        if (aqua_is_number(a.arg(1))) cv::applyColorMap(a.mat(0), d, a.integer(1));
        else cv::applyColorMap(a.mat(0), d, a.mat(1));
        return a.out(d);
    });
}

void register_histograms(aqua_library *library, aqua_class *cv) {
    FN(cv, "equalizeHist|直方圖等化", "v", { cv::Mat d; cv::equalizeHist(a.mat(0), d); return a.out(d); });
    FN(cv, "calcHist|計算直方圖", "vvvvv|b", {
        cv::Mat hist;
        std::vector<float> ranges = a.floats(4);
        cv::calcHist(a.mats(0), a.ints(1), a.mat(2, cv::Mat()), hist, a.ints(3), ranges, a.flag(5));
        return a.out(hist);
    });
    FN(cv, "calcBackProject|反向投影", "vvvvn", {
        cv::Mat d;
        cv::calcBackProject(a.mats(0), a.ints(1), a.mat(2), d, a.floats(3), a.number(4));
        return a.out(d);
    });
    FN(cv, "compareHist|比較直方圖", "vvn", { return a.out(cv::compareHist(a.mat(0), a.mat(1), a.integer(2))); });
    FN(cv, "matchTemplate|樣板比對", "vvn|v", {
        cv::Mat d;
        cv::matchTemplate(a.mat(0), a.mat(1), d, a.integer(2), a.mat(3, cv::Mat()));
        return a.out(d);
    });

    aqua_class *clahe = define_class<ClaheBox>(library, "CLAHE");
    FN(cv, "createCLAHE|建立CLAHE", "|nv", {
        return make<ClaheBox>(a.rt, cv::createCLAHE(a.number(0, 40), a.size(1, cv::Size(8, 8))));
    });
    FN(clahe, "apply|套用", "v", { cv::Mat d; a.self<ClaheBox>()->apply(a.mat(0), d); return a.out(d); });
    FN(clahe, "setClipLimit|設定對比限制", "n", { a.self<ClaheBox>()->setClipLimit(a.number(0)); return a.none(); });
    FN(clahe, "getClipLimit|對比限制", "", { return a.out(a.self<ClaheBox>()->getClipLimit()); });
    FN(clahe, "setTilesGridSize|設定區塊格數", "v", { a.self<ClaheBox>()->setTilesGridSize(a.size(0)); return a.none(); });
    FN(clahe, "getTilesGridSize|區塊格數", "", { return a.out(a.self<ClaheBox>()->getTilesGridSize()); });
}

void register_shapes(aqua_library *library, aqua_class *cv) {
    FN(cv, "findContours|尋找輪廓", "vnn|v", {
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat hierarchy;
        cv::findContours(a.mat(0), contours, hierarchy, a.integer(1), a.integer(2), a.has(3) ? a.point(3) : cv::Point());
        return a.tuple(contours, hierarchy);
    });
    FN(cv, "drawContours|繪製輪廓", "vvnv|nnvnv", {
        cv::drawContours(a.target(0), a.contours(1), a.integer(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8),
            a.mat(6, cv::Mat()), a.integer(7, INT_MAX), a.has(8) ? a.point(8) : cv::Point());
        return a.arg(0);
    });
    FN(cv, "contourArea|輪廓面積", "v|b", { return a.out(cv::contourArea(a.points2f(0), a.flag(1))); });
    FN(cv, "arcLength|輪廓周長", "vb", { return a.out(cv::arcLength(a.points2f(0), a.flag(1))); });
    FN(cv, "approxPolyDP|多邊形近似", "vnb", {
        std::vector<cv::Point> result;
        cv::approxPolyDP(a.points(0), result, a.number(1), a.flag(2));
        return a.out(result);
    });
    FN(cv, "boundingRect|外接矩形", "v", { return a.out(cv::boundingRect(a.points(0))); });
    FN(cv, "minAreaRect|最小外接矩形", "v", { return a.out(cv::minAreaRect(a.points2f(0))); });
    FN(cv, "boxPoints|矩形頂點", "v", {
        aqua_value box = a.arg(0);
        if (box.type != AQUA_ARRAY || aqua_array_length(a.rt, box) != 3) a.fail("boxPoints needs [[cx, cy], [width, height], angle]");
        double center[2], size[2];
        aqua_array_read_f64(a.rt, aqua_array_get(a.rt, box, 0), 0, 2, center);
        aqua_array_read_f64(a.rt, aqua_array_get(a.rt, box, 1), 0, 2, size);
        cv::RotatedRect r({float(center[0]), float(center[1])}, {float(size[0]), float(size[1])}, float(aqua_array_get(a.rt, box, 2).number));
        cv::Point2f corners[4];
        r.points(corners);
        return a.out(std::vector<cv::Point2f>(corners, corners + 4));
    });
    FN(cv, "minEnclosingCircle|最小外接圓", "v", {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(a.points2f(0), center, radius);
        return a.tuple(center, double(radius));
    });
    FN(cv, "minEnclosingTriangle|最小外接三角形", "v", {
        std::vector<cv::Point2f> triangle;
        double area = cv::minEnclosingTriangle(a.points2f(0), triangle);
        return a.tuple(area, triangle);
    });
    FN(cv, "fitEllipse|擬合橢圓", "v", { return a.out(cv::fitEllipse(a.points2f(0))); });
    FN(cv, "fitLine|擬合直線", "vnnnn", {
        cv::Vec4f line;
        cv::fitLine(a.points2f(0), line, a.integer(1), a.number(2), a.number(3), a.number(4));
        return a.out(std::vector<double>{line[0], line[1], line[2], line[3]});
    });
    FN(cv, "convexHull|凸包", "v|bb", {
        if (a.flag(2, true)) {
            std::vector<cv::Point> hull;
            cv::convexHull(a.points(0), hull, a.flag(1));
            return a.out(hull);
        }
        std::vector<int> hull;
        cv::convexHull(a.points(0), hull, a.flag(1), false);
        return a.out(hull);
    });
    FN(cv, "convexityDefects|凸缺陷", "vv", {
        std::vector<cv::Vec4i> defects;
        cv::convexityDefects(a.points(0), a.ints(1), defects);
        std::vector<std::vector<int>> rows;
        for (auto &d : defects) rows.push_back({d[0], d[1], d[2], d[3]});
        return a.out(rows);
    });
    FN(cv, "isContourConvex|是否凸輪廓", "v", { return a.out(cv::isContourConvex(a.points2f(0))); });
    FN(cv, "moments|矩", "v|b", {
        aqua_value v = a.arg(0);
        bool image = v.type != AQUA_ARRAY;
        return a.out(image ? cv::moments(a.mat(0), a.flag(1)) : cv::moments(a.points2f(0), a.flag(1)));
    });
    FN(cv, "HuMoments|Hu矩", "h", {
        cv::Moments m;
        double *fields[] = {&m.m00, &m.m10, &m.m01, &m.m20, &m.m11, &m.m02, &m.m30, &m.m21, &m.m12, &m.m03,
            &m.mu20, &m.mu11, &m.mu02, &m.mu30, &m.mu21, &m.mu12, &m.mu03, &m.nu20, &m.nu11, &m.nu02, &m.nu30, &m.nu21, &m.nu12, &m.nu03};
        const char *names[] = {"m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03", "mu20", "mu11", "mu02",
            "mu30", "mu21", "mu12", "mu03", "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"};
        for (size_t k = 0; k < std::size(names); k++) {
            aqua_value field = a.field(0, names[k]);
            *fields[k] = aqua_is_number(field) ? field.number : 0;
        }
        double hu[7];
        cv::HuMoments(m, hu);
        return a.out(std::vector<double>(hu, hu + 7));
    });
    FN(cv, "matchShapes|形狀比對", "vvnn", { return a.out(cv::matchShapes(a.points2f(0), a.points2f(1), a.integer(2), a.number(3))); });
    FN(cv, "pointPolygonTest|點多邊形測試", "vvb", { return a.out(cv::pointPolygonTest(a.points2f(0), a.point2f(1), a.flag(2))); });
    FN(cv, "intersectConvexConvex|凸多邊形交集", "vv|b", {
        std::vector<cv::Point2f> intersection;
        float area = cv::intersectConvexConvex(a.points2f(0), a.points2f(1), intersection, a.flag(2, true));
        return a.tuple(double(area), intersection);
    });
    FN(cv, "connectedComponents|連通元件", "v|nn", {
        cv::Mat labels;
        int n = cv::connectedComponents(a.mat(0), labels, a.integer(1, 8), a.integer(2, CV_32S));
        return a.tuple(n, labels);
    });
    FN(cv, "connectedComponentsWithStats|連通元件統計", "v|nn", {
        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(a.mat(0), labels, stats, centroids, a.integer(1, 8), a.integer(2, CV_32S));
        return a.tuple(n, labels, stats, centroids);
    });
    FN(cv, "watershed|分水嶺分割", "vv", { cv::watershed(a.mat(0), a.target(1)); return a.arg(1); });
    FN(cv, "grabCut|GrabCut前景分割", "vvvvvn|n", {
        cv::Mat &mask = a.target(1), &background = a.target(3), &foreground = a.target(4);
        cv::grabCut(a.mat(0), mask, a.has(2) ? a.rect(2) : cv::Rect(), background, foreground, a.integer(5), a.integer(6, cv::GC_EVAL));
        return a.tuple(mask, background, foreground);
    });
    FN(cv, "floodFill|漫水填充", "vvvv|vvn", {
        cv::Rect box;
        cv::Mat mask = a.has(1) ? a.target(1) : cv::Mat();
        int area = cv::floodFill(a.target(0), mask, a.point(2), a.scalar(3), &box, a.scalar(4, cv::Scalar()), a.scalar(5, cv::Scalar()), a.integer(6, 4));
        return a.tuple(area, a.target(0), mask, box);
    });
    FN(cv, "pyrMeanShiftFiltering|均值漂移濾波", "vnn|n", {
        cv::Mat d;
        cv::pyrMeanShiftFiltering(a.mat(0), d, a.number(1), a.number(2), a.integer(3, 1));
        return a.out(d);
    });

    aqua_class *lsd = define_class<LsdBox>(library, "LineSegmentDetector");
    FN(cv, "createLineSegmentDetector|建立線段偵測器", "|n", { return make<LsdBox>(a.rt, cv::createLineSegmentDetector(a.integer(0, cv::LSD_REFINE_STD))); });
    FN(lsd, "detect|偵測", "v", {
        std::vector<cv::Vec4f> lines;
        a.self<LsdBox>()->detect(a.mat(0), lines);
        std::vector<std::vector<double>> rows;
        for (auto &l : lines) rows.push_back({l[0], l[1], l[2], l[3]});
        return a.out(rows);
    });
}

void register_corners(aqua_class *cv) {
    FN(cv, "HoughLines|霍夫直線", "vnnn|nn", {
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(a.mat(0), lines, a.number(1), a.number(2), a.integer(3), a.number(4, 0), a.number(5, 0));
        std::vector<std::vector<double>> rows;
        for (auto &l : lines) rows.push_back({l[0], l[1]});
        return a.out(rows);
    });
    FN(cv, "HoughLinesP|機率霍夫直線", "vnnn|nn", {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(a.mat(0), lines, a.number(1), a.number(2), a.integer(3), a.number(4, 0), a.number(5, 0));
        std::vector<std::vector<int>> rows;
        for (auto &l : lines) rows.push_back({l[0], l[1], l[2], l[3]});
        return a.out(rows);
    });
    FN(cv, "HoughCircles|霍夫圓", "vnnn|nnnn", {
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(a.mat(0), circles, a.integer(1), a.number(2), a.number(3), a.number(4, 100), a.number(5, 100), a.integer(6, 0), a.integer(7, 0));
        std::vector<std::vector<double>> rows;
        for (auto &c : circles) rows.push_back({c[0], c[1], c[2]});
        return a.out(rows);
    });
    FN(cv, "goodFeaturesToTrack|追蹤特徵點", "vnnn|vnbn", {
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(a.mat(0), corners, a.integer(1), a.number(2), a.number(3), a.mat(4, cv::Mat()), a.integer(5, 3), a.flag(6), a.number(7, 0.04));
        return a.out(corners);
    });
    FN(cv, "cornerHarris|哈里斯角點", "vnnn|n", {
        cv::Mat d;
        cv::cornerHarris(a.mat(0), d, a.integer(1), a.integer(2), a.number(3), a.integer(4, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "cornerMinEigenVal|最小特徵值角點", "vn|nn", {
        cv::Mat d;
        cv::cornerMinEigenVal(a.mat(0), d, a.integer(1), a.integer(2, 3), a.integer(3, cv::BORDER_DEFAULT));
        return a.out(d);
    });
    FN(cv, "cornerSubPix|次像素角點", "vvvvv", {
        std::vector<cv::Point2f> corners = a.points2f(1);
        cv::cornerSubPix(a.mat(0), corners, a.size(2), a.size(3), a.criteria(4, cv::TermCriteria()));
        return a.out(corners);
    });
}

void register_drawing(aqua_class *cv) {
    FN(cv, "line|畫線", "vvvv|nnn", {
        cv::line(a.target(0), a.point(1), a.point(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8), a.integer(6, 0));
        return a.arg(0);
    });
    FN(cv, "arrowedLine|畫箭頭", "vvvv|nnnn", {
        cv::arrowedLine(a.target(0), a.point(1), a.point(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8), a.integer(6, 0), a.number(7, 0.1));
        return a.arg(0);
    });
    FN(cv, "rectangle|畫矩形", "vvv|vnnn", {
        // cv2 takes two corners (img, pt1, pt2, color, ...) or a rectangle (img, rec, color, ...).
        aqua_value second = a.arg(1);
        bool corners = second.type == AQUA_ARRAY && aqua_array_length(a.rt, second) == 2;
        if (corners) cv::rectangle(a.target(0), a.point(1), a.point(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8), a.integer(6, 0));
        else cv::rectangle(a.target(0), a.rect(1), a.scalar(2), a.integer(3, 1), a.integer(4, cv::LINE_8), a.integer(5, 0));
        return a.arg(0);
    });
    FN(cv, "circle|畫圓", "vvnv|nnn", {
        cv::circle(a.target(0), a.point(1), a.integer(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8), a.integer(6, 0));
        return a.arg(0);
    });
    FN(cv, "ellipse|畫橢圓", "vvv|nnnvnnn", {
        // (img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift) or (img, box, color, ...).
        if (a.arg(1).type == AQUA_ARRAY && aqua_array_length(a.rt, a.arg(1)) == 3) {
            aqua_value box = a.arg(1);
            double center[2], size[2];
            aqua_array_read_f64(a.rt, aqua_array_get(a.rt, box, 0), 0, 2, center);
            aqua_array_read_f64(a.rt, aqua_array_get(a.rt, box, 1), 0, 2, size);
            cv::RotatedRect r({float(center[0]), float(center[1])}, {float(size[0]), float(size[1])}, float(aqua_array_get(a.rt, box, 2).number));
            cv::ellipse(a.target(0), r, a.scalar(2), a.integer(3, 1), a.integer(4, cv::LINE_8));
            return a.arg(0);
        }
        cv::ellipse(a.target(0), a.point(1), a.size(2), a.number(3), a.number(4), a.number(5), a.scalar(6), a.integer(7, 1),
            a.integer(8, cv::LINE_8), a.integer(9, 0));
        return a.arg(0);
    });
    FN(cv, "polylines|畫折線", "vvbv|nnn", {
        cv::polylines(a.target(0), a.contours(1), a.flag(2), a.scalar(3), a.integer(4, 1), a.integer(5, cv::LINE_8), a.integer(6, 0));
        return a.arg(0);
    });
    FN(cv, "fillPoly|填滿多邊形", "vvv|nnv", {
        cv::fillPoly(a.target(0), a.contours(1), a.scalar(2), a.integer(3, cv::LINE_8), a.integer(4, 0), a.has(5) ? a.point(5) : cv::Point());
        return a.arg(0);
    });
    FN(cv, "fillConvexPoly|填滿凸多邊形", "vvv|nn", {
        cv::fillConvexPoly(a.target(0), a.points(1), a.scalar(2), a.integer(3, cv::LINE_8), a.integer(4, 0));
        return a.arg(0);
    });
    FN(cv, "putText|繪製文字", "vsvnnv|nnb", {
        cv::putText(a.target(0), a.text(1), a.point(2), a.integer(3), a.number(4), a.scalar(5), a.integer(6, 1), a.integer(7, cv::LINE_8), a.flag(8));
        return a.arg(0);
    });
    FN(cv, "getTextSize|文字尺寸", "snnn", {
        int baseline = 0;
        cv::Size size = cv::getTextSize(a.text(0), a.integer(1), a.number(2), a.integer(3), &baseline);
        return a.tuple(size, baseline);
    });
    FN(cv, "drawMarker|繪製標記", "vvv|nnnn", {
        cv::drawMarker(a.target(0), a.point(1), a.scalar(2), a.integer(3, cv::MARKER_CROSS), a.integer(4, 20), a.integer(5, 1), a.integer(6, 8));
        return a.arg(0);
    });
    FN(cv, "clipLine|裁切線段", "vvv", {
        cv::Point p1 = a.point(1), p2 = a.point(2);
        bool inside = cv::clipLine(a.rect(0), p1, p2);
        return a.tuple(inside, p1, p2);
    });
}

} // namespace

void register_imgproc(aqua_library *library, aqua_class *cv) {
    register_geometry(cv);
    register_filters(cv);
    register_histograms(library, cv);
    register_shapes(library, cv);
    register_corners(cv);
    register_drawing(cv);
}

} // namespace acv
