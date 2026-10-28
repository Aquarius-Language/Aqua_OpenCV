// photo: inpainting, denoising, non-photorealistic rendering, seamless cloning, decolorization and HDR.
// photo：影像修補、去噪、非寫實渲染、無縫複製、去色及高動態範圍。
#include "binding.hpp"
#include <opencv2/photo.hpp>

namespace acv {
namespace {

using MergeMertens = cv::Ptr<cv::MergeMertens>;
using Tonemap = cv::Ptr<cv::Tonemap>;
using MergeDebevec = cv::Ptr<cv::MergeDebevec>;
using CalibrateDebevec = cv::Ptr<cv::CalibrateDebevec>;

} // namespace

void register_photo(aqua_library *library, aqua_class *cv) {
    FN(cv, "inpaint|影像修補", "vvnn", { cv::Mat d; cv::inpaint(a.mat(0), a.mat(1), d, a.number(2), a.integer(3)); return a.out(d); });
    FN(cv, "fastNlMeansDenoising|非局部均值去噪", "v|nnn", {
        cv::Mat d;
        cv::fastNlMeansDenoising(a.mat(0), d, float(a.number(1, 3)), a.integer(2, 7), a.integer(3, 21));
        return a.out(d);
    });
    FN(cv, "fastNlMeansDenoisingColored|彩色非局部均值去噪", "v|nnnn", {
        cv::Mat d;
        cv::fastNlMeansDenoisingColored(a.mat(0), d, float(a.number(1, 3)), float(a.number(2, 3)), a.integer(3, 7), a.integer(4, 21));
        return a.out(d);
    });
    FN(cv, "detailEnhance|細節增強", "v|nn", {
        cv::Mat d;
        cv::detailEnhance(a.mat(0), d, float(a.number(1, 10)), float(a.number(2, 0.15)));
        return a.out(d);
    });
    FN(cv, "edgePreservingFilter|保邊濾波", "v|nnn", {
        cv::Mat d;
        cv::edgePreservingFilter(a.mat(0), d, a.integer(1, 1), float(a.number(2, 60)), float(a.number(3, 0.4)));
        return a.out(d);
    });
    FN(cv, "pencilSketch|鉛筆素描", "v|nnn", {
        cv::Mat gray, color;
        cv::pencilSketch(a.mat(0), gray, color, float(a.number(1, 60)), float(a.number(2, 0.07)), float(a.number(3, 0.02)));
        return a.tuple(gray, color);
    });
    FN(cv, "stylization|風格化", "v|nn", {
        cv::Mat d;
        cv::stylization(a.mat(0), d, float(a.number(1, 60)), float(a.number(2, 0.45)));
        return a.out(d);
    });
    FN(cv, "seamlessClone|無縫複製", "vvvvn", {
        cv::Mat d;
        cv::seamlessClone(a.mat(0), a.mat(1), a.mat(2), a.point(3), d, a.integer(4));
        return a.out(d);
    });
    FN(cv, "colorChange|顏色變換", "vv|nnn", {
        cv::Mat d;
        cv::colorChange(a.mat(0), a.mat(1), d, float(a.number(2, 1)), float(a.number(3, 1)), float(a.number(4, 1)));
        return a.out(d);
    });
    FN(cv, "illuminationChange|光照變換", "vv|nn", {
        cv::Mat d;
        cv::illuminationChange(a.mat(0), a.mat(1), d, float(a.number(2, 0.2)), float(a.number(3, 0.4)));
        return a.out(d);
    });
    FN(cv, "textureFlattening|紋理平滑", "vv|nnn", {
        cv::Mat d;
        cv::textureFlattening(a.mat(0), a.mat(1), d, float(a.number(2, 30)), float(a.number(3, 45)), a.integer(4, 3));
        return a.out(d);
    });
    FN(cv, "decolor|去色", "v", {
        cv::Mat gray, boost;
        cv::decolor(a.mat(0), gray, boost);
        return a.tuple(gray, boost);
    });

    aqua_class *mertens = define_class<MergeMertens>(library, "MergeMertens");
    FN(cv, "createMergeMertens|建立曝光融合", "|nnn", {
        return make<MergeMertens>(a.rt, cv::createMergeMertens(float(a.number(0, 1)), float(a.number(1, 1)), float(a.number(2, 0))));
    });
    FN(mertens, "process|處理", "v", { cv::Mat d; a.self<MergeMertens>()->process(a.mats(0), d); return a.out(d); });

    aqua_class *calibrate = define_class<CalibrateDebevec>(library, "CalibrateDebevec");
    FN(cv, "createCalibrateDebevec|建立Debevec響應校正", "|nnb", {
        return make<CalibrateDebevec>(a.rt, cv::createCalibrateDebevec(a.integer(0, 70), float(a.number(1, 10)), a.flag(2)));
    });
    FN(calibrate, "process|處理", "vv", {
        cv::Mat d;
        a.self<CalibrateDebevec>()->process(a.mats(0), d, cv::Mat(a.floats(1), true));
        return a.out(d);
    });
    aqua_class *debevec = define_class<MergeDebevec>(library, "MergeDebevec");
    FN(cv, "createMergeDebevec|建立Debevec合成", "", { return make<MergeDebevec>(a.rt, cv::createMergeDebevec()); });
    FN(debevec, "process|處理", "vv|v", {
        cv::Mat d;
        cv::Mat times(a.floats(1), true);
        if (a.has(2)) a.self<MergeDebevec>()->process(a.mats(0), d, times, a.mat(2));
        else a.self<MergeDebevec>()->process(a.mats(0), d, times);
        return a.out(d);
    });
    aqua_class *tonemap = define_class<Tonemap>(library, "Tonemap");
    FN(cv, "createTonemap|建立色調映射", "|n", { return make<Tonemap>(a.rt, cv::createTonemap(float(a.number(0, 1)))); });
    FN(cv, "createTonemapReinhard|建立Reinhard色調映射", "|nnnn", {
        return make<Tonemap>(a.rt, cv::createTonemapReinhard(float(a.number(0, 1)), float(a.number(1, 0)), float(a.number(2, 1)), float(a.number(3, 0))));
    });
    FN(tonemap, "process|處理", "v", { cv::Mat d; a.self<Tonemap>()->process(a.mat(0), d); return a.out(d); });
}

} // namespace acv
