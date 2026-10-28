// calib3d and stitching: homographies, camera calibration, pose estimation, stereo and panoramas.
// calib3d 與 stitching：單應性、相機校正、姿態估計、立體視覺及全景拼接。
#include "binding.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <cfloat>

namespace acv {
namespace {

using Stereo = cv::Ptr<cv::StereoMatcher>;
using StitcherBox = cv::Ptr<cv::Stitcher>;

void register_geometry(aqua_class *cv) {
    FN(cv, "findHomography|單應性矩陣", "vv|nnvnn", {
        cv::Mat mask;
        cv::Mat h = cv::findHomography(a.points2f(0), a.points2f(1), a.integer(2, 0), a.number(3, 3), mask, a.integer(5, 2000), a.number(6, 0.995));
        return a.tuple(h, mask);
    });
    FN(cv, "estimateAffine2D|估計仿射", "vv|nnnn", {
        cv::Mat inliers;
        cv::Mat m = cv::estimateAffine2D(a.points2f(0), a.points2f(1), inliers, a.integer(2, cv::RANSAC), a.number(3, 3), size_t(a.integer(4, 2000)), a.number(5, 0.99));
        return a.tuple(m, inliers);
    });
    FN(cv, "estimateAffinePartial2D|估計相似變換", "vv|nnnn", {
        cv::Mat inliers;
        cv::Mat m = cv::estimateAffinePartial2D(a.points2f(0), a.points2f(1), inliers, a.integer(2, cv::RANSAC), a.number(3, 3), size_t(a.integer(4, 2000)), a.number(5, 0.99));
        return a.tuple(m, inliers);
    });
    FN(cv, "findFundamentalMat|基礎矩陣", "vv|nnn", {
        cv::Mat mask;
        cv::Mat f = cv::findFundamentalMat(a.points2f(0), a.points2f(1), a.integer(2, cv::FM_RANSAC), a.number(3, 3), a.number(4, 0.99), mask);
        return a.tuple(f, mask);
    });
    FN(cv, "findEssentialMat|本質矩陣", "vvv|nnn", {
        cv::Mat mask;
        cv::Mat e = cv::findEssentialMat(a.points2f(0), a.points2f(1), a.mat(2), a.integer(3, cv::RANSAC), a.number(4, 0.999), a.number(5, 1), mask);
        return a.tuple(e, mask);
    });
    FN(cv, "recoverPose|還原姿態", "vvvv|v", {
        cv::Mat r, t, mask = a.mat(4, cv::Mat()).clone();
        int inliers = cv::recoverPose(a.mat(0), a.points2f(1), a.points2f(2), a.mat(3), r, t, mask);
        return a.tuple(inliers, r, t, mask);
    });
    FN(cv, "triangulatePoints|三角測量", "vvvv", {
        cv::Mat points;
        cv::triangulatePoints(a.mat(0), a.mat(1), a.points2f(2), a.points2f(3), points);
        return a.out(points);
    });
    FN(cv, "convertPointsFromHomogeneous|齊次座標轉換", "v", {
        cv::Mat d;
        cv::convertPointsFromHomogeneous(a.mat(0), d);
        return a.out(d);
    });
    FN(cv, "decomposeHomographyMat|分解單應性", "vv", {
        std::vector<cv::Mat> rotations, translations, normals;
        int n = cv::decomposeHomographyMat(a.mat(0), a.mat(1), rotations, translations, normals);
        return a.tuple(n, rotations, translations, normals);
    });
}

void register_calibration(aqua_class *cv) {
    FN(cv, "findChessboardCorners|尋找棋盤角點", "vv|n", {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(a.mat(0), a.size(1), corners, a.integer(2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE));
        return a.tuple(found, corners);
    });
    FN(cv, "findChessboardCornersSB|尋找棋盤角點SB", "vv|n", {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCornersSB(a.mat(0), a.size(1), corners, a.integer(2, 0));
        return a.tuple(found, corners);
    });
    FN(cv, "findCirclesGrid|尋找圓點陣列", "vv|n", {
        std::vector<cv::Point2f> centers;
        bool found = cv::findCirclesGrid(a.mat(0), a.size(1), centers, a.integer(2, cv::CALIB_CB_SYMMETRIC_GRID));
        return a.tuple(found, centers);
    });
    FN(cv, "drawChessboardCorners|繪製棋盤角點", "vvvb", {
        cv::drawChessboardCorners(a.target(0), a.size(1), a.points2f(2), a.flag(3));
        return a.arg(0);
    });
    FN(cv, "calibrateCamera|相機校正", "vvv|vvnv", {
        cv::Mat camera = a.mat(3, cv::Mat()).clone(), distortion = a.mat(4, cv::Mat()).clone();
        std::vector<cv::Mat> rotations, translations;
        double rms = cv::calibrateCamera(a.contours3f(0), a.contours2f(1), a.size(2), camera, distortion, rotations, translations,
            a.integer(5, 0), a.criteria(6, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON)));
        return a.tuple(rms, camera, distortion, rotations, translations);
    });
    FN(cv, "getOptimalNewCameraMatrix|最佳新相機矩陣", "vvvn|vb", {
        cv::Rect valid;
        cv::Mat m = cv::getOptimalNewCameraMatrix(a.mat(0), a.mat(1), a.size(2), a.number(3), a.size(4, cv::Size()), &valid, a.flag(5));
        return a.tuple(m, valid);
    });
    FN(cv, "undistort|去畸變", "vvv|v", {
        cv::Mat d;
        cv::undistort(a.mat(0), d, a.mat(1), a.mat(2), a.mat(3, cv::Mat()));
        return a.out(d);
    });
    FN(cv, "undistortPoints|點去畸變", "vvv|vv", {
        std::vector<cv::Point2f> points;
        cv::undistortPoints(a.points2f(0), points, a.mat(1), a.mat(2), a.mat(3, cv::Mat()), a.mat(4, cv::Mat()));
        return a.out(points);
    });
    FN(cv, "initUndistortRectifyMap|去畸變映射", "vvvvvn", {
        cv::Mat x, y;
        cv::initUndistortRectifyMap(a.mat(0), a.mat(1), a.mat(2, cv::Mat()), a.mat(3), a.size(4), a.integer(5), x, y);
        return a.tuple(x, y);
    });
    FN(cv, "solvePnP|姿態估計", "vvvv|vvbn", {
        cv::Mat r = a.mat(4, cv::Mat()).clone(), t = a.mat(5, cv::Mat()).clone();
        bool ok = cv::solvePnP(a.points3f(0), a.points2f(1), a.mat(2), a.mat(3, cv::Mat()), r, t, a.flag(6), a.integer(7, cv::SOLVEPNP_ITERATIVE));
        return a.tuple(ok, r, t);
    });
    FN(cv, "solvePnPRansac|穩健姿態估計", "vvvv|vvbnnnn", {
        cv::Mat r = a.mat(4, cv::Mat()).clone(), t = a.mat(5, cv::Mat()).clone();
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(a.points3f(0), a.points2f(1), a.mat(2), a.mat(3, cv::Mat()), r, t, a.flag(6), a.integer(7, 100),
            float(a.number(8, 8)), a.number(9, 0.99), inliers, a.integer(10, cv::SOLVEPNP_ITERATIVE));
        return a.tuple(ok, r, t, inliers);
    });
    FN(cv, "projectPoints|投影點", "vvvvv", {
        std::vector<cv::Point2f> points;
        cv::Mat jacobian;
        cv::projectPoints(a.points3f(0), a.mat(1), a.mat(2), a.mat(3), a.mat(4, cv::Mat()), points, jacobian);
        return a.tuple(points, jacobian);
    });
    FN(cv, "Rodrigues|羅德里格斯轉換", "v", {
        cv::Mat d, jacobian;
        cv::Rodrigues(a.mat(0), d, jacobian);
        return a.tuple(d, jacobian);
    });
    FN(cv, "drawFrameAxes|繪製座標軸", "vvvvvn|n", {
        cv::drawFrameAxes(a.target(0), a.mat(1), a.mat(2, cv::Mat()), a.mat(3), a.mat(4), float(a.number(5)), a.integer(6, 3));
        return a.arg(0);
    });
    FN(cv, "reprojectImageTo3D|重投影為3D", "vv|bn", {
        cv::Mat d;
        cv::reprojectImageTo3D(a.mat(0), d, a.mat(1), a.flag(2), a.integer(3, -1));
        return a.out(d);
    });
    FN(cv, "stereoRectify|立體校正", "vvvvvvv|nnv", {
        cv::Mat r1, r2, p1, p2, q;
        cv::Rect roi1, roi2;
        cv::stereoRectify(a.mat(0), a.mat(1), a.mat(2), a.mat(3), a.size(4), a.mat(5), a.mat(6), r1, r2, p1, p2, q,
            a.integer(7, cv::CALIB_ZERO_DISPARITY), a.number(8, -1), a.size(9, cv::Size()), &roi1, &roi2);
        return a.tuple(r1, r2, p1, p2, q, roi1, roi2);
    });
}

} // namespace

void register_calib3d(aqua_library *library, aqua_class *cv) {
    register_geometry(cv);
    register_calibration(cv);

    aqua_class *bm = define_class<Stereo>(library, "StereoBM");
    aqua_class *sgbm = define_class<Stereo>(library, "StereoSGBM");
    for (aqua_class *cls : {bm, sgbm}) {
        FN(cls, "compute|計算視差", "vv", { cv::Mat d; a.self<Stereo>()->compute(a.mat(0), a.mat(1), d); return a.out(d); });
        FN(cls, "setNumDisparities|設定視差數", "n", { a.self<Stereo>()->setNumDisparities(a.integer(0)); return a.none(); });
        FN(cls, "setBlockSize|設定區塊大小", "n", { a.self<Stereo>()->setBlockSize(a.integer(0)); return a.none(); });
        FN(cls, "setMinDisparity|設定最小視差", "n", { a.self<Stereo>()->setMinDisparity(a.integer(0)); return a.none(); });
        FN(cls, "setSpeckleWindowSize|設定斑點視窗", "n", { a.self<Stereo>()->setSpeckleWindowSize(a.integer(0)); return a.none(); });
        FN(cls, "setSpeckleRange|設定斑點範圍", "n", { a.self<Stereo>()->setSpeckleRange(a.integer(0)); return a.none(); });
    }
    FN(cv, "StereoBM_create|建立StereoBM", "|nn", {
        return make<Stereo>(a.rt, cv::StereoBM::create(a.integer(0, 0), a.integer(1, 21)), Kind<Stereo>::classes[0]);
    });
    FN(cv, "StereoSGBM_create|建立StereoSGBM", "|nnnnnnnnnnn", {
        return make<Stereo>(a.rt, cv::StereoSGBM::create(a.integer(0, 0), a.integer(1, 16), a.integer(2, 3), a.integer(3, 0), a.integer(4, 0),
            a.integer(5, 0), a.integer(6, 0), a.integer(7, 0), a.integer(8, 0), a.integer(9, 0), a.integer(10, cv::StereoSGBM::MODE_SGBM)),
            Kind<Stereo>::classes[1]);
    });

    aqua_class *stitcher = define_class<StitcherBox>(library, "Stitcher");
    FN(cv, "Stitcher_create|建立全景拼接器", "|n", { return make<StitcherBox>(a.rt, cv::Stitcher::create(cv::Stitcher::Mode(a.integer(0, cv::Stitcher::PANORAMA)))); });
    FN(stitcher, "stitch|拼接", "v", {
        cv::Mat pano;
        int status = int(a.self<StitcherBox>()->stitch(a.mats(0), pano));
        return a.tuple(status, pano);
    });
    FN(stitcher, "setPanoConfidenceThresh|設定拼接信心門檻", "n", { a.self<StitcherBox>()->setPanoConfidenceThresh(a.number(0)); return a.none(); });
    FN(stitcher, "setRegistrationResol|設定配準解析度", "n", { a.self<StitcherBox>()->setRegistrationResol(a.number(0)); return a.none(); });
    FN(stitcher, "setWaveCorrection|設定波形校正", "b", { a.self<StitcherBox>()->setWaveCorrection(a.flag(0)); return a.none(); });
}

} // namespace acv
