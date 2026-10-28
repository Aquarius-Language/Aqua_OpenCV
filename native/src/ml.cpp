// ml (cv2.ml.*): k-nearest neighbours, support vector machines, random forests, decision trees, logistic regression,
// neural networks and normal Bayes classifiers sharing the StatModel methods.
// ml：K 近鄰、支援向量機、隨機森林、決策樹、邏輯迴歸、神經網路與常態貝氏分類器，共用 StatModel 方法。
#include "binding.hpp"
#include <opencv2/ml.hpp>

namespace acv {
namespace {

using Model = cv::Ptr<cv::ml::StatModel>;

template <class T> cv::Ptr<T> as(Args &a) {
    cv::Ptr<T> model = a.self<Model>().dynamicCast<T>();
    if (!model) a.fail("this method belongs to another model type");
    return model;
}

cv::Mat samples(Args &a, unsigned i) {
    cv::Mat m;
    a.mat(i).convertTo(m, CV_32F);
    return m;
}

void model_methods(aqua_class *cls) {
    FN(cls, "train|訓練", "vnv", {
        Model &model = a.self<Model>();
        cv::Mat responses = a.mat(2);
        // K-nearest, logistic regression and neural networks take float responses. The others classify integer labels
        // and regress floats: arrays (read as doubles) of whole numbers become labels, as int32 arrays do in cv2.
        bool floats = model.dynamicCast<cv::ml::KNearest>() || model.dynamicCast<cv::ml::LogisticRegression>() || model.dynamicCast<cv::ml::ANN_MLP>();
        if (!floats && responses.depth() != CV_32S) {
            cv::Mat rounded;
            responses.convertTo(rounded, CV_32S);
            cv::Mat back;
            rounded.convertTo(back, responses.depth());
            floats = cv::norm(back, responses, cv::NORM_INF) > 0;
        }
        responses.convertTo(responses, floats ? CV_32F : CV_32S);
        return a.out(model->train(samples(a, 0), a.integer(1), responses));
    });
    FN(cls, "predict|預測", "v|n", {
        cv::Mat results;
        float value = a.self<Model>()->predict(samples(a, 0), results, a.integer(1, 0));
        return a.tuple(double(value), results);
    });
    FN(cls, "isTrained|已訓練", "", { return a.out(a.self<Model>()->isTrained()); });
    FN(cls, "isClassifier|是分類器", "", { return a.out(a.self<Model>()->isClassifier()); });
    FN(cls, "getVarCount|變數數", "", { return a.out(a.self<Model>()->getVarCount()); });
    FN(cls, "calcError|計算誤差", "vnvb", {
        cv::Mat responses = a.mat(2);
        responses.convertTo(responses, CV_32F);
        auto data = cv::ml::TrainData::create(samples(a, 0), a.integer(1), responses);
        cv::Mat predictions;
        return a.out(double(a.self<Model>()->calcError(data, a.flag(3), predictions)));
    });
}

aqua_class *model_class(aqua_library *library, const char *name) {
    aqua_class *cls = define_class<Model>(library, name);
    model_methods(cls);
    return cls;
}

} // namespace

void register_ml(aqua_library *library, aqua_class *cv) {
    // Kind<Model>::classes keeps this order; each factory below creates objects of its class.
    aqua_class *knn = model_class(library, "KNearest");
    aqua_class *svm = model_class(library, "SVM");
    aqua_class *rtrees = model_class(library, "RTrees");
    aqua_class *dtrees = model_class(library, "DTrees");
    aqua_class *logistic = model_class(library, "LogisticRegression");
    aqua_class *mlp = model_class(library, "ANN_MLP");
    model_class(library, "NormalBayesClassifier");

    FN(cv, "KNearest_create|建立K近鄰", "", { return make<Model>(a.rt, cv::ml::KNearest::create(), Kind<Model>::classes[0]); });
    FN(knn, "findNearest|尋找近鄰", "vn", {
        cv::Mat results, neighbours, distances;
        float value = as<cv::ml::KNearest>(a)->findNearest(samples(a, 0), a.integer(1), results, neighbours, distances);
        return a.tuple(double(value), results, neighbours, distances);
    });
    FN(knn, "setDefaultK|設定預設K", "n", { as<cv::ml::KNearest>(a)->setDefaultK(a.integer(0)); return a.none(); });
    FN(knn, "setAlgorithmType|設定演算法", "n", { as<cv::ml::KNearest>(a)->setAlgorithmType(a.integer(0)); return a.none(); });

    FN(cv, "SVM_create|建立支援向量機", "", { return make<Model>(a.rt, cv::ml::SVM::create(), Kind<Model>::classes[1]); });
    FN(svm, "setType|設定類型", "n", { as<cv::ml::SVM>(a)->setType(a.integer(0)); return a.none(); });
    FN(svm, "setKernel|設定核函數", "n", { as<cv::ml::SVM>(a)->setKernel(a.integer(0)); return a.none(); });
    FN(svm, "setC|設定C", "n", { as<cv::ml::SVM>(a)->setC(a.number(0)); return a.none(); });
    FN(svm, "setGamma|設定Gamma", "n", { as<cv::ml::SVM>(a)->setGamma(a.number(0)); return a.none(); });
    FN(svm, "setDegree|設定次數", "n", { as<cv::ml::SVM>(a)->setDegree(a.number(0)); return a.none(); });
    FN(svm, "setNu|設定Nu", "n", { as<cv::ml::SVM>(a)->setNu(a.number(0)); return a.none(); });
    FN(svm, "setTermCriteria|設定終止條件", "v", { as<cv::ml::SVM>(a)->setTermCriteria(a.criteria(0, cv::TermCriteria())); return a.none(); });
    FN(svm, "getSupportVectors|支援向量", "", { return a.out(as<cv::ml::SVM>(a)->getSupportVectors()); });

    FN(cv, "RTrees_create|建立隨機森林", "", { return make<Model>(a.rt, cv::ml::RTrees::create(), Kind<Model>::classes[2]); });
    FN(cv, "DTrees_create|建立決策樹", "", { return make<Model>(a.rt, cv::ml::DTrees::create(), Kind<Model>::classes[3]); });
    for (aqua_class *cls : {rtrees, dtrees}) {
        FN(cls, "setMaxDepth|設定最大深度", "n", { as<cv::ml::DTrees>(a)->setMaxDepth(a.integer(0)); return a.none(); });
        FN(cls, "setMinSampleCount|設定最少樣本數", "n", { as<cv::ml::DTrees>(a)->setMinSampleCount(a.integer(0)); return a.none(); });
        FN(cls, "setCVFolds|設定交叉驗證折數", "n", { as<cv::ml::DTrees>(a)->setCVFolds(a.integer(0)); return a.none(); });
    }
    FN(rtrees, "setTermCriteria|設定終止條件", "v", { as<cv::ml::RTrees>(a)->setTermCriteria(a.criteria(0, cv::TermCriteria())); return a.none(); });

    FN(cv, "LogisticRegression_create|建立邏輯迴歸", "", { return make<Model>(a.rt, cv::ml::LogisticRegression::create(), Kind<Model>::classes[4]); });
    FN(logistic, "setLearningRate|設定學習率", "n", { as<cv::ml::LogisticRegression>(a)->setLearningRate(a.number(0)); return a.none(); });
    FN(logistic, "setIterations|設定迭代次數", "n", { as<cv::ml::LogisticRegression>(a)->setIterations(a.integer(0)); return a.none(); });

    FN(cv, "ANN_MLP_create|建立多層感知器", "", { return make<Model>(a.rt, cv::ml::ANN_MLP::create(), Kind<Model>::classes[5]); });
    FN(mlp, "setLayerSizes|設定層大小", "v", { as<cv::ml::ANN_MLP>(a)->setLayerSizes(cv::Mat(a.ints(0), true)); return a.none(); });
    FN(mlp, "setActivationFunction|設定激活函數", "n|nn", {
        as<cv::ml::ANN_MLP>(a)->setActivationFunction(a.integer(0), a.number(1, 0), a.number(2, 0));
        return a.none();
    });
    FN(mlp, "setTrainMethod|設定訓練方法", "n|nn", { as<cv::ml::ANN_MLP>(a)->setTrainMethod(a.integer(0), a.number(1, 0), a.number(2, 0)); return a.none(); });
    FN(mlp, "setTermCriteria|設定終止條件", "v", { as<cv::ml::ANN_MLP>(a)->setTermCriteria(a.criteria(0, cv::TermCriteria())); return a.none(); });

    FN(cv, "NormalBayesClassifier_create|建立常態貝氏分類器", "", {
        return make<Model>(a.rt, cv::ml::NormalBayesClassifier::create(), Kind<Model>::classes[6]);
    });
}

} // namespace acv
