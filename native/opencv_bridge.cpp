#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/stitching.hpp>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cmath>
using json=nlohmann::json;
#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
namespace {
std::mutex gate;
struct Session { std::unordered_map<int,cv::Mat> images; cv::Ptr<cv::BackgroundSubtractor> background=cv::createBackgroundSubtractorMOG2(); cv::CascadeClassifier cascade; std::vector<uchar> model; };
std::unordered_map<int,Session> sessions;
int nextSession=1,nextImage=1;
constexpr size_t MaxPixels=16777216,MaxImages=128;
cv::Mat& image(Session& s,int id){auto i=s.images.find(id);if(i==s.images.end())throw std::runtime_error("Unknown or released image handle");return i->second;}
int keep(Session& s,cv::Mat m){if(m.empty()||m.total()>MaxPixels||s.images.size()>=MaxImages)throw std::runtime_error("Empty image or image limit exceeded");int id=nextImage++;s.images.emplace(id,std::move(m));return id;}
cv::Mat gray(const cv::Mat& m){cv::Mat out;if(m.channels()==1)return m;cv::cvtColor(m,out,cv::COLOR_BGR2GRAY);return out;}
cv::Mat color(const cv::Mat& m){cv::Mat out;if(m.channels()==3)return m;cv::cvtColor(m,out,cv::COLOR_GRAY2BGR);return out;}
cv::Mat sample(int frame){cv::Mat m(300,480,CV_8UC3);for(int y=0;y<m.rows;y++)for(int x=0;x<m.cols;x++)m.at<cv::Vec3b>(y,x)={(uchar)(40+x/3),(uchar)(30+y/2),(uchar)(45+x/5)};
for(int y=0;y<6;y++)for(int x=0;x<8;x++)cv::rectangle(m,{18+x*20,20+y*20,20,20},(x+y)%2?cv::Scalar(230,230,230):cv::Scalar(20,20,20),-1);
cv::rectangle(m,{210,40,100,85},{30,200,240},-1);cv::circle(m,{370+(frame%25),85},38,{220,80,40},-1);cv::circle(m,{355,210},38,{40,230,70},3);cv::line(m,{25,240},{280,165},{250,240,230},4);cv::putText(m,"OpenCV 4.12",{25,285},cv::FONT_HERSHEY_SIMPLEX,.75,{245,245,245},2);return m;}
json features(){return {"gray","hsv","gaussian","median","bilateral","canny","sobel","laplacian","threshold","adaptive","otsu","erode","dilate","open","close","gradient","equalize","clahe","colormap","resize","rotate","perspective","contours","hough_lines","hough_circles","corners","orb","sift","matching","template","watershed","grabcut","kmeans","dft","inpaint","denoise","optical_flow","background","face","qr","stereo","homography","calibration","stitching","dnn"};}
cv::Mat apply(Session& s,const std::string& op,const cv::Mat& src,const json& p,json& stats){
 cv::Mat dst,g=gray(src),bgr=color(src);int k=p.value("kernel",5);if(k<1||k>99||k%2==0)throw std::runtime_error("kernel must be odd in 1..99");
 double threshold=p.value("threshold",100.0);
 if(op=="gray")dst=g.clone();
 else if(op=="hsv")cv::cvtColor(bgr,dst,cv::COLOR_BGR2HSV);
 else if(op=="gaussian")cv::GaussianBlur(src,dst,{k,k},0);
 else if(op=="median")cv::medianBlur(src,dst,k);
 else if(op=="bilateral")cv::bilateralFilter(src,dst,9,75,75);
 else if(op=="canny")cv::Canny(g,dst,threshold,threshold*2);
 else if(op=="sobel"){cv::Mat f;cv::Sobel(g,f,CV_16S,1,0);cv::convertScaleAbs(f,dst);}
 else if(op=="laplacian"){cv::Mat f;cv::Laplacian(g,f,CV_16S);cv::convertScaleAbs(f,dst);}
 else if(op=="threshold"||op=="otsu")stats["threshold"]=cv::threshold(g,dst,threshold,255,cv::THRESH_BINARY|(op=="otsu"?cv::THRESH_OTSU:0));
 else if(op=="adaptive")cv::adaptiveThreshold(g,dst,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,11,2);
 else if(op=="erode"||op=="dilate"||op=="open"||op=="close"||op=="gradient"){
  int kind=op=="erode"?cv::MORPH_ERODE:op=="dilate"?cv::MORPH_DILATE:op=="open"?cv::MORPH_OPEN:op=="close"?cv::MORPH_CLOSE:cv::MORPH_GRADIENT;cv::morphologyEx(src,dst,kind,cv::getStructuringElement(cv::MORPH_ELLIPSE,{k,k}));
 }
 else if(op=="equalize")cv::equalizeHist(g,dst);
 else if(op=="clahe")cv::createCLAHE(2.0,{8,8})->apply(g,dst);
 else if(op=="colormap")cv::applyColorMap(g,dst,cv::COLORMAP_TURBO);
 else if(op=="resize"){int w=p.value("width",320),h=p.value("height",200);if(w<1||h<1||(int64_t)w*h>MaxPixels)throw std::runtime_error("Invalid resize dimensions");cv::resize(src,dst,{w,h},0,0,cv::INTER_AREA);}
 else if(op=="rotate")cv::warpAffine(src,dst,cv::getRotationMatrix2D({src.cols/2.f,src.rows/2.f},p.value("angle",20.0),1),src.size());
 else if(op=="perspective"){cv::Point2f a[]={{0,0},{(float)src.cols,0},{(float)src.cols,(float)src.rows},{0,(float)src.rows}},b[]={{35,20},{src.cols-30.f,0},{(float)src.cols,src.rows-20.f},{0,(float)src.rows}};cv::warpPerspective(src,dst,cv::getPerspectiveTransform(a,b),src.size());}
 else if(op=="contours"){cv::Mat edges;cv::Canny(g,edges,80,160);std::vector<std::vector<cv::Point>> c;cv::findContours(edges,c,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);dst=bgr.clone();cv::drawContours(dst,c,-1,{40,255,160},2);stats["contours"]=c.size();}
 else if(op=="hough_lines"){cv::Mat edges;cv::Canny(g,edges,70,140);std::vector<cv::Vec4i> lines;cv::HoughLinesP(edges,lines,1,CV_PI/180,40,35,8);dst=bgr.clone();for(auto l:lines)cv::line(dst,{l[0],l[1]},{l[2],l[3]},{20,40,255},2);stats["lines"]=lines.size();}
 else if(op=="hough_circles"){std::vector<cv::Vec3f> circles;cv::Mat blur;cv::GaussianBlur(g,blur,{9,9},2);cv::HoughCircles(blur,circles,cv::HOUGH_GRADIENT,1,30,100,30,10,90);dst=bgr.clone();for(auto c:circles)cv::circle(dst,{cvRound(c[0]),cvRound(c[1])},cvRound(c[2]),{20,255,100},2);stats["circles"]=circles.size();}
 else if(op=="corners"){std::vector<cv::Point2f> points;cv::goodFeaturesToTrack(g,points,150,.01,8);dst=bgr.clone();for(auto a:points)cv::circle(dst,a,3,{30,255,130},-1);stats["corners"]=points.size();}
 else if(op=="orb"||op=="sift"){std::vector<cv::KeyPoint> keys;cv::Ptr<cv::Feature2D> detector=op=="orb"?cv::Ptr<cv::Feature2D>(cv::ORB::create(400)):cv::Ptr<cv::Feature2D>(cv::SIFT::create(400));detector->detect(g,keys);cv::drawKeypoints(bgr,keys,dst,{80,255,120},cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);stats["keypoints"]=keys.size();}
 else if(op=="matching"||op=="homography"){cv::Mat second,transform=(cv::Mat_<double>(2,3)<<1,0,18,0,1,12);cv::warpAffine(bgr,second,transform,src.size());std::vector<cv::KeyPoint>a,b;cv::Mat da,db;auto orb=cv::ORB::create(500);orb->detectAndCompute(g,cv::noArray(),a,da);orb->detectAndCompute(gray(second),cv::noArray(),b,db);std::vector<cv::DMatch> matches;if(!da.empty()&&!db.empty())cv::BFMatcher(cv::NORM_HAMMING,true).match(da,db,matches);std::sort(matches.begin(),matches.end(),[](auto a,auto b){return a.distance<b.distance;});if(matches.size()>40)matches.resize(40);stats["matches"]=matches.size();if(op=="homography"&&matches.size()>=4){std::vector<cv::Point2f>pa,pb;for(auto m:matches){pa.push_back(a[m.queryIdx].pt);pb.push_back(b[m.trainIdx].pt);}cv::Mat mask;auto H=cv::findHomography(pa,pb,cv::RANSAC,3,mask);stats["inliers"]=cv::countNonZero(mask);if(!H.empty())cv::warpPerspective(bgr,dst,H,src.size());else dst=second;}else cv::drawMatches(bgr,a,second,b,matches,dst);}
 else if(op=="template"){cv::Rect roi(src.cols/3,src.rows/4,std::max(8,src.cols/5),std::max(8,src.rows/5));cv::Mat scores;cv::matchTemplate(g,g(roi),scores,cv::TM_CCOEFF_NORMED);double max;cv::Point where;cv::minMaxLoc(scores,nullptr,&max,nullptr,&where);dst=bgr.clone();cv::rectangle(dst,{where,roi.size()},{30,255,150},3);stats["score"]=max;}
 else if(op=="watershed"){cv::Mat markers=cv::Mat::zeros(src.size(),CV_32S);cv::rectangle(markers,{1,1,src.cols-2,src.rows-2},cv::Scalar(1),3);cv::circle(markers,{src.cols/2,src.rows/2},30,cv::Scalar(2),-1);cv::watershed(bgr,markers);dst=bgr.clone();dst.setTo(cv::Scalar(20,30,255),markers==-1);stats["segments"]=2;}
 else if(op=="grabcut"){cv::Mat mask,bg,fg;cv::grabCut(bgr,mask,{10,10,src.cols-20,src.rows-20},bg,fg,3,cv::GC_INIT_WITH_RECT);dst=cv::Mat::zeros(src.size(),CV_8UC3);bgr.copyTo(dst,(mask==cv::GC_FGD)|(mask==cv::GC_PR_FGD));}
 else if(op=="kmeans"){cv::Mat points; bgr.reshape(1,(int)bgr.total()).convertTo(points,CV_32F);cv::Mat labels,centers;cv::kmeans(points,6,labels,{cv::TermCriteria::EPS|cv::TermCriteria::MAX_ITER,20,1},2,cv::KMEANS_PP_CENTERS,centers);dst=bgr.clone();for(int i=0;i<(int)bgr.total();i++)for(int c=0;c<3;c++)dst.data[i*3+c]=cv::saturate_cast<uchar>(centers.at<float>(labels.at<int>(i),c));stats["clusters"]=6;}
 else if(op=="dft"){cv::Mat f,complex,planes[2];g.convertTo(f,CV_32F);cv::dft(f,complex,cv::DFT_COMPLEX_OUTPUT);cv::split(complex,planes);cv::magnitude(planes[0],planes[1],f);f+=1;cv::log(f,f);int cx=f.cols/2,cy=f.rows/2;cv::Mat shifted(f.size(),f.type());for(int y=0;y<f.rows;y++)for(int x=0;x<f.cols;x++)shifted.at<float>((y+cy)%f.rows,(x+cx)%f.cols)=f.at<float>(y,x);cv::normalize(shifted,dst,0,255,cv::NORM_MINMAX,CV_8U);}
 else if(op=="inpaint"){cv::Mat mask=cv::Mat::zeros(src.size(),CV_8U),damaged=bgr.clone();cv::line(mask,{80,20},{260,src.rows-20},cv::Scalar(255),5);damaged.setTo(cv::Scalar(255,255,255),mask);cv::inpaint(damaged,mask,dst,3,cv::INPAINT_TELEA);}
 else if(op=="denoise")cv::fastNlMeansDenoisingColored(bgr,dst,5,5,7,21);
 else if(op=="optical_flow"){cv::Mat next,transform=(cv::Mat_<double>(2,3)<<1,0,8,0,1,5);cv::warpAffine(g,next,transform,g.size());cv::Mat flow;cv::calcOpticalFlowFarneback(g,next,flow,.5,3,15,3,5,1.2,0);dst=bgr.clone();for(int y=8;y<src.rows;y+=16)for(int x=8;x<src.cols;x+=16){auto v=flow.at<cv::Point2f>(y,x);cv::arrowedLine(dst,{x,y},{cvRound(x+v.x*2),cvRound(y+v.y*2)},{30,255,130},1);}stats["pair"]=json::array({8,5});}
 else if(op=="background"){s.background->apply(sample(0),dst,1);s.background->apply(src,dst,.01);}
 else if(op=="face"){if(s.cascade.empty())throw std::runtime_error("Load a face cascade first");std::vector<cv::Rect>faces;s.cascade.detectMultiScale(g,faces,1.1,3);dst=bgr.clone();for(auto box:faces)cv::rectangle(dst,box,{30,255,130},2);stats["faces"]=faces.size();}
 else if(op=="qr"){cv::Mat code;cv::QRCodeEncoder::create()->encode("https://opencv.org",code);cv::copyMakeBorder(code,code,4,4,4,4,cv::BORDER_CONSTANT,cv::Scalar(255));cv::resize(code,dst,{280,280},0,0,cv::INTER_NEAREST);cv::Mat points;stats["text"]=cv::QRCodeDetector().detectAndDecode(dst,points);}
 else if(op=="stereo"){cv::Mat right,transform=(cv::Mat_<double>(2,3)<<1,0,-8,0,1,0),disparity;cv::warpAffine(g,right,transform,g.size());cv::StereoBM::create(64,15)->compute(g,right,disparity);cv::normalize(disparity,dst,0,255,cv::NORM_MINMAX,CV_8U);stats["input"]= "synthetic rectified pair, 8px shift";}
 else if(op=="calibration"){std::vector<cv::Point3f>object;for(int y=0;y<6;y++)for(int x=0;x<9;x++)object.emplace_back((float)x,(float)y,0.f);cv::Mat camera=(cv::Mat_<double>(3,3)<<600,0,240,0,600,150,0,0,1),rv=(cv::Mat_<double>(3,1)<<.2,.1,.05),tv=(cv::Mat_<double>(3,1)<<-4,-2,15);std::vector<cv::Point2f>points;cv::projectPoints(object,rv,tv,camera,cv::noArray(),points);cv::Mat r,t;bool solved=cv::solvePnP(object,points,camera,cv::noArray(),r,t);dst=bgr.clone();cv::drawChessboardCorners(dst,{9,6},points,solved);stats["solved"]=solved;stats["z"]=t.at<double>(2);}
 else if(op=="stitching"){cv::Mat wide;cv::resize(bgr,wide,{src.cols*2,src.rows});cv::RNG rng(42);for(int i=0;i<350;i++)cv::circle(wide,{rng.uniform(8,wide.cols-8),rng.uniform(8,wide.rows-8)},rng.uniform(2,7),cv::Scalar(rng.uniform(20,255),rng.uniform(20,255),rng.uniform(20,255)),-1);std::vector<cv::Mat>views={wide(cv::Rect(0,0,src.cols+160,src.rows)).clone(),wide(cv::Rect(src.cols-160,0,src.cols+160,src.rows)).clone()};auto stitcher=cv::Stitcher::create(cv::Stitcher::SCANS);stitcher->setFeaturesFinder(cv::SIFT::create(2000));stitcher->setFeaturesMatcher(cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false,false,.3f,4));stitcher->setPanoConfidenceThresh(.1);auto status=stitcher->stitch(views,dst);stats["status"]=(int)status;stats["input"]="two overlapping views with deterministic texture";if(status!=cv::Stitcher::OK)throw std::runtime_error("Stitching needs more overlapping texture; status "+std::to_string(status));}
 else if(op=="dnn"){if(s.model.empty())throw std::runtime_error("Load a YuNet ONNX model first");auto detector=cv::FaceDetectorYN::create("onnx",s.model,{},src.size(),.7f,.3f,5000,cv::dnn::DNN_BACKEND_OPENCV,cv::dnn::DNN_TARGET_CPU);cv::Mat faces;detector->detect(bgr,faces);dst=bgr.clone();for(int i=0;i<faces.rows;i++){auto a=faces.ptr<float>(i);cv::rectangle(dst,{(int)a[0],(int)a[1],(int)a[2],(int)a[3]},{30,240,120},2);for(int j=0;j<5;j++)cv::circle(dst,{(int)a[4+j*2],(int)a[5+j*2]},2,{40,90,255},-1);}stats["faces"]=faces.rows;stats["model"]="YuNet";}
 else throw std::runtime_error("Unsupported OpenCV operation: "+op);
 return dst;
}
json command(const json& q){
 std::string op=q.at("op");
 if(op=="open"){int id=nextSession++;if(sessions.size()>=32)throw std::runtime_error("Session limit exceeded");sessions.emplace(id,Session{});cv::setUseOptimized(true);return id;}
 int sid=q.at("session");auto it=sessions.find(sid);if(it==sessions.end())throw std::runtime_error("Unknown or closed OpenCV session");auto& s=it->second;
 if(op=="close"){sessions.erase(it);return nullptr;}
 if(op=="info")return {{"version",CV_VERSION},{"backend","native C++ / SIMD dispatch / parallel CPU"},{"threads",cv::getNumThreads()},{"optimized",cv::useOptimized()},{"openclAvailable",cv::ocl::haveOpenCL()},{"openclEnabled",cv::ocl::useOpenCL()},{"features",features()},{"images",s.images.size()}};
 if(op=="configure"){int threads=q.value("threads",cv::getNumThreads());if(threads<1||threads>256)throw std::runtime_error("threads must be 1..256");cv::setNumThreads(threads);return true;}
 if(op=="sample")return keep(s,sample(q.value("frame",0)));
 if(op=="release"){s.images.erase(q.at("image").get<int>());return nullptr;}
 if(op=="inspect"){auto& m=image(s,q.at("image"));return {{"width",m.cols},{"height",m.rows},{"channels",m.channels()}};}
 if(op=="apply"){auto& m=image(s,q.at("image"));json stats=json::object();auto start=cv::getTickCount();auto out=apply(s,q.at("feature"),m,q.value("parameters",json::object()),stats);double ms=(cv::getTickCount()-start)*1000.0/cv::getTickFrequency();int id=keep(s,std::move(out));return {{"image",id},{"milliseconds",ms},{"stats",stats}};}
 throw std::runtime_error("Unknown OpenCV command");
}
int copy(const std::string& s,uint8_t* output,int capacity){if((int)s.size()>capacity)return -2;std::memcpy(output,s.data(),s.size());return (int)s.size();}
int id32(const uint8_t* p){return p[0]|p[1]<<8|p[2]<<16|p[3]<<24;}
}
EXPORT int32_t aqua_library_abi(){return 1;}
EXPORT int32_t cv_command(const uint8_t* input,int32_t length,uint8_t* output,int32_t capacity){
 std::lock_guard<std::mutex> lock(gate);
 try{return copy(json({{"ok",true},{"value",command(json::parse(input,input+length))}}).dump(),output,capacity);}
 catch(const std::exception& e){return copy(json({{"ok",false},{"error",e.what()}}).dump(),output,capacity);}catch(...){return -1;}
}
EXPORT int32_t cv_decode(const uint8_t* input,int32_t length,uint8_t* output,int32_t capacity){
 std::lock_guard<std::mutex> lock(gate);try{if(length<5||capacity<4)return -1;auto it=sessions.find(id32(input));if(it==sessions.end())return -1;auto m=cv::imdecode(cv::Mat(1,length-4,CV_8U,(void*)(input+4)),cv::IMREAD_COLOR);int id=keep(it->second,m);for(int i=0;i<4;i++)output[i]=(id>>(i*8))&255;return 4;}catch(...){return -1;}
}
EXPORT int32_t cv_upload(const uint8_t* input,int32_t length,uint8_t* output,int32_t capacity){
 std::lock_guard<std::mutex> lock(gate);try{if(length<6||capacity<1)return -1;auto it=sessions.find(id32(input));if(it==sessions.end())return -1;auto& s=it->second;if(input[4]==1){cv::FileStorage fs(std::string((const char*)input+5,length-5),cv::FileStorage::READ|cv::FileStorage::MEMORY);if(!s.cascade.read(fs.getFirstTopLevelNode()))return -1;}else if(input[4]==2)s.model.assign(input+5,input+length);else return -1;output[0]=1;return 1;}catch(...){return -1;}
}
EXPORT int32_t cv_png(const uint8_t* input,int32_t length,uint8_t* output,int32_t capacity){
 std::lock_guard<std::mutex> lock(gate);try{if(length!=8)return -1;auto it=sessions.find(id32(input));if(it==sessions.end())return -1;auto& m=image(it->second,id32(input+4));std::vector<uchar> bytes;cv::imencode(".png",m,bytes,{cv::IMWRITE_PNG_COMPRESSION,1});if(bytes.size()>(size_t)capacity)return -2;std::memcpy(output,bytes.data(),bytes.size());return (int)bytes.size();}catch(...){return -1;}
}
