import initial from '../vendor/opencv.cjs';
let cv=initial;
const ready=new Promise((resolve,reject)=>{if(cv.Mat)resolve();else if(typeof initial.then==='function')initial.then(instance=>{cv=instance;resolve();},reject);else cv.onRuntimeInitialized=resolve;});
const encoder=new TextEncoder(),decoder=new TextDecoder();
export function aqua_library_abi(){return 1;}
const supported=['gray','hsv','gaussian','median','bilateral','canny','sobel','laplacian','threshold','adaptive','otsu','erode','dilate','open','close','gradient','equalize','clahe','colormap','resize','rotate','perspective','contours','hough_lines','hough_circles','corners','orb','template','watershed','grabcut','kmeans','dft','optical_flow','background','face','homography'];
export function aqua_create_library(){
 let nextSession=1,nextImage=1,closed=false;const sessions=new Map();
 const session=id=>{const s=sessions.get(id);if(!s)throw Error('Unknown or closed OpenCV session');return s;};
 const image=(s,id)=>{const m=s.images.get(id);if(!m)throw Error('Unknown or released image handle');return m;};
 const keep=(s,m)=>{if(m.empty()||m.total()>16777216||s.images.size>=128){m.delete();throw Error('Image limit exceeded');}const id=nextImage++;s.images.set(id,m);return id;};
 const remove=s=>{for(const m of s.images.values())m.delete();s.images.clear();s.background?.delete();s.cascade?.delete();if(s.cascadePath)cv.FS_unlink(s.cascadePath);};
 const u32=(b,p=0)=>new DataView(b.buffer,b.byteOffset,b.byteLength).getUint32(p,true);
 const encodedId=id=>{const b=new Uint8Array(4);new DataView(b.buffer).setUint32(0,id,true);return b;};
 const sample=frame=>{
  const m=new cv.Mat(300,480,cv.CV_8UC3);for(let y=0;y<300;y++)for(let x=0;x<480;x++){let i=(y*480+x)*3;m.data[i]=40+x/3;m.data[i+1]=30+y/2;m.data[i+2]=45+x/5;}
  for(let y=0;y<6;y++)for(let x=0;x<8;x++)cv.rectangle(m,new cv.Point(18+x*20,20+y*20),new cv.Point(38+x*20,40+y*20),(x+y)%2?new cv.Scalar(230,230,230):new cv.Scalar(20,20,20),-1);
  cv.rectangle(m,new cv.Point(210,40),new cv.Point(310,125),new cv.Scalar(30,200,240),-1);cv.circle(m,new cv.Point(370+frame%25,85),38,new cv.Scalar(220,80,40),-1);cv.circle(m,new cv.Point(355,210),38,new cv.Scalar(40,230,70),3);cv.line(m,new cv.Point(25,240),new cv.Point(280,165),new cv.Scalar(250,240,230),4);cv.putText(m,'OpenCV',new cv.Point(25,285),cv.FONT_HERSHEY_SIMPLEX,.75,new cv.Scalar(245,245,245),2);return m;
 };
 function apply(s,src,feature,p){
  if(!supported.includes(feature))throw Error('This OpenCV.js build does not expose '+feature+'; use the native build');
  let kept;const resources=[],own=o=>(resources.push(o),o);const dst=own(new cv.Mat()),g=own(new cv.Mat()),bgr=own(new cv.Mat());const stats={};
  const kernel=p.kernel??5;if(!Number.isInteger(kernel)||kernel<1||kernel>99||kernel%2!==1)throw Error('kernel must be odd in 1..99');
  try{
   if(src.channels()===1){src.copyTo(g);cv.cvtColor(src,bgr,cv.COLOR_GRAY2BGR);}else{cv.cvtColor(src,g,cv.COLOR_BGR2GRAY);src.copyTo(bgr);}
   const threshold=p.threshold??100,size=new cv.Size(kernel,kernel),anchor=new cv.Point(-1,-1);
   if(feature==='gray')g.copyTo(dst);
   else if(feature==='hsv')cv.cvtColor(bgr,dst,cv.COLOR_BGR2HSV);
   else if(feature==='gaussian')cv.GaussianBlur(src,dst,size,0);
   else if(feature==='median')cv.medianBlur(src,dst,kernel);
   else if(feature==='bilateral')cv.bilateralFilter(src,dst,9,75,75);
   else if(feature==='canny')cv.Canny(g,dst,threshold,threshold*2);
   else if(feature==='sobel'||feature==='laplacian'){const f=own(new cv.Mat());if(feature==='sobel')cv.Sobel(g,f,cv.CV_16S,1,0);else cv.Laplacian(g,f,cv.CV_16S);cv.convertScaleAbs(f,dst);}
   else if(feature==='threshold'||feature==='otsu')stats.threshold=cv.threshold(g,dst,threshold,255,cv.THRESH_BINARY|(feature==='otsu'?cv.THRESH_OTSU:0));
   else if(feature==='adaptive')cv.adaptiveThreshold(g,dst,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2);
   else if(['erode','dilate','open','close','gradient'].includes(feature)){const kinds={erode:cv.MORPH_ERODE,dilate:cv.MORPH_DILATE,open:cv.MORPH_OPEN,close:cv.MORPH_CLOSE,gradient:cv.MORPH_GRADIENT};const k=own(cv.getStructuringElement(cv.MORPH_ELLIPSE,size));cv.morphologyEx(src,dst,kinds[feature],k);}
   else if(feature==='equalize')cv.equalizeHist(g,dst);
   else if(feature==='clahe'){const clahe=own(new cv.CLAHE(2,new cv.Size(8,8)));clahe.apply(g,dst);}
   else if(feature==='colormap')cv.applyColorMap(g,dst,cv.COLORMAP_TURBO);
   else if(feature==='resize'){const w=p.width??320,h=p.height??200;if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||w*h>16777216)throw Error('Invalid resize dimensions');cv.resize(src,dst,new cv.Size(w,h),0,0,cv.INTER_AREA);}
   else if(feature==='rotate'){const matrix=own(cv.getRotationMatrix2D(new cv.Point(src.cols/2,src.rows/2),p.angle??20,1));cv.warpAffine(src,dst,matrix,src.size());}
   else if(feature==='perspective'||feature==='homography'){const a=own(cv.matFromArray(4,1,cv.CV_32FC2,[0,0,src.cols,0,src.cols,src.rows,0,src.rows])),b=own(cv.matFromArray(4,1,cv.CV_32FC2,[35,20,src.cols-30,0,src.cols,src.rows-20,0,src.rows]));const matrix=own(feature==='homography'?cv.findHomography(a,b):cv.getPerspectiveTransform(a,b));cv.warpPerspective(src,dst,matrix,src.size());stats.correspondences=4;}
   else if(feature==='contours'){const edges=own(new cv.Mat()),contours=own(new cv.MatVector()),hierarchy=own(new cv.Mat());cv.Canny(g,edges,80,160);cv.findContours(edges,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);bgr.copyTo(dst);cv.drawContours(dst,contours,-1,new cv.Scalar(40,255,160),2);stats.contours=contours.size();}
   else if(feature==='hough_lines'){const edges=own(new cv.Mat()),lines=own(new cv.Mat());cv.Canny(g,edges,70,140);cv.HoughLinesP(edges,lines,1,Math.PI/180,40,35,8);bgr.copyTo(dst);for(let i=0;i<lines.rows;i++){const l=lines.data32S.subarray(i*4,i*4+4);cv.line(dst,new cv.Point(l[0],l[1]),new cv.Point(l[2],l[3]),new cv.Scalar(20,40,255),2);}stats.lines=lines.rows;}
   else if(feature==='hough_circles'){const blur=own(new cv.Mat()),circles=own(new cv.Mat());cv.GaussianBlur(g,blur,new cv.Size(9,9),2);cv.HoughCircles(blur,circles,cv.HOUGH_GRADIENT,1,30,100,30,10,90);bgr.copyTo(dst);for(let i=0;i<circles.cols;i++){const a=circles.data32F.subarray(i*3,i*3+3);cv.circle(dst,new cv.Point(Math.round(a[0]),Math.round(a[1])),Math.round(a[2]),new cv.Scalar(20,255,100),2);}stats.circles=circles.cols;}
   else if(feature==='corners'){const points=own(new cv.Mat());cv.goodFeaturesToTrack(g,points,150,.01,8);bgr.copyTo(dst);for(let i=0;i<points.rows;i++)cv.circle(dst,new cv.Point(points.data32F[i*2],points.data32F[i*2+1]),3,new cv.Scalar(30,255,130),-1);stats.corners=points.rows;}
   else if(feature==='orb'){const detector=own(new cv.ORB()),keys=own(new cv.KeyPointVector()),mask=own(new cv.Mat());detector.detect(g,keys,mask);cv.drawKeypoints(bgr,keys,dst,new cv.Scalar(80,255,120));stats.keypoints=keys.size();}
   else if(feature==='template'){const roi=new cv.Rect(Math.floor(src.cols/3),Math.floor(src.rows/4),Math.floor(src.cols/5),Math.floor(src.rows/5)),part=own(g.roi(roi)),scores=own(new cv.Mat());cv.matchTemplate(g,part,scores,cv.TM_CCOEFF_NORMED);const result=cv.minMaxLoc(scores);bgr.copyTo(dst);cv.rectangle(dst,result.maxLoc,new cv.Point(result.maxLoc.x+roi.width,result.maxLoc.y+roi.height),new cv.Scalar(30,255,150),3);stats.score=result.maxVal;}
   else if(feature==='watershed'){const markers=own(cv.Mat.zeros(src.rows,src.cols,cv.CV_32S));cv.rectangle(markers,new cv.Point(1,1),new cv.Point(src.cols-2,src.rows-2),new cv.Scalar(1),3);cv.circle(markers,new cv.Point(src.cols/2,src.rows/2),30,new cv.Scalar(2),-1);cv.watershed(bgr,markers);bgr.copyTo(dst);for(let i=0;i<markers.data32S.length;i++)if(markers.data32S[i]===-1)dst.data.set([20,30,255],i*3);stats.segments=2;}
   else if(feature==='grabcut'){const mask=own(new cv.Mat()),bg=own(new cv.Mat()),fg=own(new cv.Mat());cv.grabCut(bgr,mask,new cv.Rect(10,10,src.cols-20,src.rows-20),bg,fg,3,cv.GC_INIT_WITH_RECT);bgr.copyTo(dst);for(let i=0;i<mask.data.length;i++)if(mask.data[i]!==cv.GC_FGD&&mask.data[i]!==cv.GC_PR_FGD)dst.data.fill(0,i*3,i*3+3);}
   else if(feature==='kmeans'){const points=own(new cv.Mat(bgr.rows*bgr.cols,3,cv.CV_32F)),labels=own(new cv.Mat()),centers=own(new cv.Mat());points.data32F.set(bgr.data);cv.kmeans(points,6,labels,new cv.TermCriteria(cv.TermCriteria_EPS|cv.TermCriteria_MAX_ITER,20,1),2,cv.KMEANS_PP_CENTERS,centers);bgr.copyTo(dst);for(let i=0;i<labels.rows;i++)for(let c=0;c<3;c++)dst.data[i*3+c]=centers.data32F[labels.data32S[i]*3+c];stats.clusters=6;}
   else if(feature==='dft'){const f=own(new cv.Mat()),complex=own(new cv.Mat()),planes=own(new cv.MatVector()),magnitude=own(new cv.Mat());g.convertTo(f,cv.CV_32F);cv.dft(f,complex,cv.DFT_COMPLEX_OUTPUT);cv.split(complex,planes);const x=own(planes.get(0)),y=own(planes.get(1));cv.magnitude(x,y,magnitude);for(let i=0;i<magnitude.data32F.length;i++)magnitude.data32F[i]=Math.log(1+magnitude.data32F[i]);cv.normalize(magnitude,dst,0,255,cv.NORM_MINMAX,cv.CV_8U);}
   else if(feature==='optical_flow'){const next=own(new cv.Mat()),matrix=own(cv.matFromArray(2,3,cv.CV_64F,[1,0,8,0,1,5])),flow=own(new cv.Mat());cv.warpAffine(g,next,matrix,g.size());cv.calcOpticalFlowFarneback(g,next,flow,.5,3,15,3,5,1.2,0);bgr.copyTo(dst);for(let y=8;y<src.rows;y+=16)for(let x=8;x<src.cols;x+=16){const i=(y*src.cols+x)*2;cv.line(dst,new cv.Point(x,y),new cv.Point(Math.round(x+flow.data32F[i]*2),Math.round(y+flow.data32F[i+1]*2)),new cv.Scalar(30,255,130),1);}stats.pair=[8,5];}
   else if(feature==='background'){s.background??=new cv.BackgroundSubtractorMOG2(500,16,true);const base=own(sample(0));s.background.apply(base,dst,1);s.background.apply(src,dst,.01);}
   else if(feature==='face'){if(!s.cascade)throw Error('Load face cascade first');const faces=own(new cv.RectVector());s.cascade.detectMultiScale(g,faces,1.1,3);bgr.copyTo(dst);for(let i=0;i<faces.size();i++){const f=faces.get(i);cv.rectangle(dst,new cv.Point(f.x,f.y),new cv.Point(f.x+f.width,f.y+f.height),new cv.Scalar(30,255,130),2);}stats.faces=faces.size();}
   kept=dst;return {mat:dst,stats};
  }finally{for(const o of resources.reverse())if(o!==kept)o.delete();}
 }
 async function command(q){
  await ready;if(closed)throw Error('OpenCV library closed');
  if(q.op==='open'){if(sessions.size>=32)throw Error('Session limit exceeded');const id=nextSession++;sessions.set(id,{images:new Map()});return id;}
  const s=session(q.session);
  if(q.op==='close'){remove(s);sessions.delete(q.session);return null;}
  if(q.op==='info')return {version:cv.getBuildInformation().match(/OpenCV ([\d.]+)/)?.[1],backend:'OpenCV.js / WebAssembly SIMD128',optimized:true,simd:true,threads:1,features:supported,images:s.images.size};
  if(q.op==='sample')return keep(s,sample(q.frame??0));
  if(q.op==='rgba'){if(!Number.isInteger(q.width)||!Number.isInteger(q.height)||q.width<1||q.height<1||q.width*q.height>16777216)throw Error('Invalid image dimensions');return keep(s,cv.Mat.zeros(q.height,q.width,cv.CV_8UC3));}
  if(q.op==='release'){const m=s.images.get(q.image);if(m){m.delete();s.images.delete(q.image);}return null;}
  if(q.op==='inspect'){const m=image(s,q.image);return {width:m.cols,height:m.rows,channels:m.channels()};}
  if(q.op==='apply'){const start=performance.now(),result=apply(s,image(s,q.image),q.feature,q.parameters??{});return {image:keep(s,result.mat),milliseconds:performance.now()-start,stats:result.stats};}
  if(q.op==='configure'){if(q.threads!==1)throw Error('This browser build is single-threaded');return true;}
  throw Error('Unknown OpenCV command');
 }
 return {
  async cv_command(bytes){try{return encoder.encode(JSON.stringify({ok:true,value:await command(JSON.parse(decoder.decode(bytes)))}));}catch(e){return encoder.encode(JSON.stringify({ok:false,error:String(e.message??e)}));}},
  async cv_write(bytes){await ready;const m=image(session(u32(bytes)),u32(bytes,4)),offset=u32(bytes,8),data=bytes.subarray(12);if(offset%4||data.length%4||offset+data.length>m.total()*4)throw Error('Invalid RGBA chunk');for(let i=0;i<data.length;i+=4){const j=(offset+i)/4*3;m.data[j]=data[i+2];m.data[j+1]=data[i+1];m.data[j+2]=data[i];}return Uint8Array.of(1);},
  async cv_pixels(bytes){await ready;const m=image(session(u32(bytes)),u32(bytes,4)),offset=u32(bytes,8),count=u32(bytes,12);if(offset%4||count%4||offset+count>m.total()*4)throw Error('Invalid RGBA chunk');const out=new Uint8Array(count);for(let i=0;i<count;i+=4){const j=(offset+i)/4*m.channels();out[i]=m.data[j+(m.channels()===1?0:2)];out[i+1]=m.data[j+(m.channels()===1?0:1)];out[i+2]=m.data[j];out[i+3]=255;}return out;},
  async cv_decode(bytes){await ready;const s=session(u32(bytes));let data;if(typeof native!=='undefined'){const decoded=JSON.parse(native.image('decode',JSON.stringify({bytes:encode64(bytes.subarray(4))})));data={width:decoded.width,height:decoded.height,data:decode64(decoded.pixels)};}else if(typeof createImageBitmap==='function'){const bitmap=await createImageBitmap(new Blob([bytes.subarray(4)]));if(bitmap.width*bitmap.height>16777216){bitmap.close();throw Error('Image limit exceeded');}const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();data=ctx.getImageData(0,0,canvas.width,canvas.height);}else{throw Error('Use ImportRGBA for host-decoded images on this platform');}const rgba=cv.matFromImageData(data),m=new cv.Mat();try{cv.cvtColor(rgba,m,cv.COLOR_RGBA2BGR);return encodedId(keep(s,m));}finally{rgba.delete();}},
  async cv_png(bytes){await ready;const m=image(session(u32(bytes)),u32(bytes,4)),rgba=new cv.Mat();try{cv.cvtColor(m,rgba,m.channels()===1?cv.COLOR_GRAY2RGBA:cv.COLOR_BGR2RGBA);if(typeof native!=='undefined'){const r=JSON.parse(native.image('encode',JSON.stringify({width:m.cols,height:m.rows,pixels:encode64(rgba.data),format:'Png'})));return decode64(r.bytes);}const canvas=new OffscreenCanvas(m.cols,m.rows),ctx=canvas.getContext('2d');ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba.data),m.cols,m.rows),0,0);return new Uint8Array(await(await canvas.convertToBlob({type:'image/png'})).arrayBuffer());}finally{rgba.delete();}},
  async cv_upload(bytes){await ready;const s=session(u32(bytes));if(bytes[4]!==1)throw Error('This OpenCV.js adapter accepts cascade XML only');s.cascade?.delete();if(s.cascadePath)cv.FS_unlink(s.cascadePath);s.cascadePath='/cascade-'+Math.random().toString(16).slice(2)+'.xml';cv.FS_createDataFile('/',s.cascadePath.slice(1),bytes.subarray(5),true,false,false);s.cascade=new cv.CascadeClassifier();if(!s.cascade.load(s.cascadePath))throw Error('Invalid cascade');return Uint8Array.of(1);},
  aqua_dispose(){if(closed)return;closed=true;for(const s of sessions.values())remove(s);sessions.clear();}
 };
}
