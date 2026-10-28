# Aqua_OpenCV 3.0.0

OpenCV 4.12 for AquariusLang (星泉), with OpenCV's own API: `Mat` objects, the function and constant names of
OpenCV's Python binding (cv2), the same argument order, and a 繁體中文 alias for every function and method. One C++
binding serves Windows, Linux, Android and the browser.

在星泉中使用 OpenCV 4.12：`Mat` 物件、與 OpenCV Python 綁定（cv2）相同的函式與常數名稱及引數順序，每個函式與方法都有繁體中文別名。
同一份 C++ 綁定支援 Windows、Linux、Android 與瀏覽器。

![Aqua × OpenCV Vision Lab](example/screenshots/vision-lab.png)

**See it run: [`example/`](example/) is the Vision Lab**, an interactive showcase with 26 live labs (color, filters, edges,
contours, Hough, features and homographies, optical flow, background subtraction, Kalman filtering, tracking, DNN
faces, QR and ArUco, AR pose, k-means, GrabCut, Fourier, computational photography, stereo, panoramas and machine
learning) on a live scene, sample photos or your camera, in English and 繁體中文.

**範例：[`example/`](example/) 視覺實驗室**，26 個可互動的即時實驗室，涵蓋 OpenCV 各模組，可切換英文與繁體中文介面。

![All 26 labs](example/screenshots/all-labs.jpg)

## Install · 安裝

```powershell
aqua bottle install ../Aqua_OpenCV --project .     # or a Git URL
aqua run .
aqua build . --target web -o web                   # windows, linux, android, web
```

## Quick start

```aquarius
var cv = import("cv");                                      # also import("OpenCV") or import("電腦視覺")
var img = cv.imread("assets/photo.jpg");
var gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
var edges = cv.Canny(cv.GaussianBlur(gray, [5, 5], 0), 80, 160);
var found = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
cv.drawContours(img, found[0], -1, [0, 255, 0], 2);
var png = cv.imencode(".png", img)[1];                      # bytes for Files, Network, ...
print(cv.getVersionString(), len(found[0]), "contours");
```

```aquarius
變數 視覺 = 匯入("電腦視覺");
變數 影像 = 視覺.讀取影像("assets/photo.jpg");
變數 灰階 = 視覺.色彩轉換(影像, 視覺.COLOR_BGR2GRAY);
變數 邊緣 = 視覺.Canny邊緣(視覺.高斯模糊(灰階, [5, 5], 0), 80, 160);
變數 結果 = 視覺.尋找輪廓(邊緣, 視覺.RETR_EXTERNAL, 視覺.CHAIN_APPROX_SIMPLE);
視覺.繪製輪廓(影像, 結果[0], -1, [0, 255, 0], 2);
印出(視覺.版本(), 長度(結果[0]));
```

## How OpenCV maps to AquariusLang · 對應規則

| OpenCV (C++ / cv2) | AquariusLang |
| --- | --- |
| `cv::Mat` / numpy array | `Mat` object: `rows`, `cols`, `type()`, `channels()`, `size()`, `shape()`, `at(r, c)`, `set(r, c, v)`, `roi(rect)`, `clone()`, `copyTo(dst, mask)`, `convertTo(type, alpha, beta)`, `setTo(v, mask)`, `reshape`, `t`, `inv`, `mul`, `matmul`, `toArray()`, `data()`, `delete()` |
| Function arguments | C++ order with the pure output arrays left out; they are returned. In-out arrays stay (the `mask` of `grabCut`, the image of drawing functions, `bestLabels` of `kmeans`). |
| Several outputs (cv2 tuples) | an array: `var r = cv.threshold(g, 0, 255, cv.THRESH_OTSU); r[0]` is the threshold, `r[1]` the image |
| Point, Size, Rect, Scalar | `[x, y]`, `[width, height]`, `[x, y, width, height]`, a number or `[b, g, r, a]` |
| RotatedRect | `[[cx, cy], [width, height], angle]` |
| `None` / `noArray()` | `cv.None` (`視覺.無`): skips an optional argument |
| Point lists and contours | arrays of `[x, y]` (a Mat of N x 2 is accepted as input too) |
| `KeyPoint`, `DMatch`, `Moments` | records (hashes): `kp["pt"]`, `kp["size"]`, `m["distance"]`, `m["queryIdx"]`, `mo["m00"]` (Chinese keys too: `kp["位置"]`) |
| Algorithms (`ORB`, `BFMatcher`, `CLAHE`, `KalmanFilter`, `Net`, `SVM`, ...) | objects with OpenCV's methods: `cv.ORB_create(500).detectAndCompute(gray)` |
| Submodules `cv2.ml`, `cv2.aruco`, `cv2.dnn` | flattened: `cv.SVM_create()`, `cv.SVM_RBF`, `cv.getPredefinedDictionary(cv.DICT_4X4_50)`, `cv.readNet(path)` |
| `cv2.data.haarcascades` | `cv.haarcascades` (`視覺.人臉模型路徑`): the bundled cascades and the YuNet face model |
| `cv.imshow` | `Window.ToProcessingImage(mat.toImage())` draws a Mat with Processing (`mat.toImage(image)` refills an image of the same size); `cv.fromImage(frame.影像)` takes camera frames |
| Errors (`cv::Exception`) | normal Aquarius runtime errors that `Network.Try` can catch |

Every name is `English|中文`: `cv.GaussianBlur` / `視覺.高斯模糊`, `orb.detectAndCompute` / `偵測器.偵測並計算`,
`mat.rows` / `矩陣.列數`. Constants keep OpenCV's names (`cv.COLOR_BGR2GRAY`). [API.md](API.md) lists every function,
method and alias.

**Memory.** Mats are freed when the collector reclaims them. The collector only sees the language's own memory, so
code that makes many large Mats per frame should call `delete()` on temporaries, exactly as OpenCV.js asks, and draw
with `mat.toImage(image)` into one reused image; the example collects each frame's Mats in a list and deletes them
after drawing.

**記憶體。** Mat 會由垃圾回收釋放；每個影格產生大量暫存 Mat 時，請如 OpenCV.js 一樣呼叫 `delete()` / `刪除()`。

## What is included · 涵蓋範圍

411 functions and methods and 430 constants across OpenCV's main modules:

- **core**: Mat creation and access, arithmetic and logic, channels, borders, statistics, `normalize`, `LUT`,
  linear algebra (`solve`, `invert`, `eigen`, `SVDecomp`, `PCACompute`), `dft`/`dct`, `kmeans`, random numbers.
- **imgcodecs**: `imread` (packaged assets, and files on native targets), `imdecode`, `imencode`, `imwrite` (native).
- **imgproc**: color conversion, geometric transforms, filters, gradients, `Canny`, morphology, thresholds,
  histograms and CLAHE, template matching, contours and shape analysis, Hough lines and circles, corners,
  connected components, watershed, GrabCut, flood fill, line segments and all drawing functions.
- **features2d**: ORB, SIFT, AKAZE, KAZE, BRISK, FAST, AGAST, GFTT, SimpleBlobDetector, MSER; brute-force and FLANN
  matchers; `drawKeypoints`, `drawMatches`.
- **calib3d**: homographies, affine and fundamental/essential matrices, chessboard and circle grids, camera
  calibration, undistortion, `solvePnP`, `projectPoints`, `Rodrigues`, `drawFrameAxes`, stereo rectification,
  StereoBM and StereoSGBM.
- **video**: Lucas-Kanade, Farneback and DIS optical flow, MOG2 and KNN background subtraction, mean shift,
  CamShift, ECC alignment, `KalmanFilter`, `TrackerMIL`.
- **objdetect**: Haar cascades, HOG people detection, QR codes (encode and decode), barcodes, ArUco markers,
  YuNet face detection and SFace face recognition.
- **photo**: inpainting, non-local means denoising, `stylization`, `pencilSketch`, `detailEnhance`,
  `edgePreservingFilter`, seamless cloning, `decolor`, HDR (Debevec, Mertens, tone mapping).
- **dnn**: ONNX, TensorFlow, Caffe, Darknet and TFLite networks, `blobFromImage`, `NMSBoxes`.
- **ml**: KNearest, SVM, RTrees, DTrees, LogisticRegression, ANN_MLP, NormalBayesClassifier.
- **stitching**: `Stitcher` panoramas and scans.

Not included: highgui windows (use Processing), videoio (use the Webcam and Camera modules), and opencv_contrib.

## Platforms and build · 平台與建置

Windows x64, Linux x64, Android (arm64-v8a, x86_64) and the web. The same `native/src` builds for every target with
the SDK's pinned Clang toolchains; no system OpenCV, Python or MSVC runtime is needed. In the browser OpenCV runs as a
single-threaded WebAssembly side module with SIMD; JPEG and PNG error handling uses WebAssembly exceptions.

```powershell
aqua library build . --targets windows-x64,web,android-arm64-v8a,android-x86_64   # Linux targets build on Linux/WSL
```

CMake verifies the pinned OpenCV archive, applies `native/vendor/opencv-wasi.patch` and builds the modules above
statically into one library with private symbols. Sources, hashes and licenses: [native/vendor/README.md](native/vendor/README.md).

## Tests · 測試

`tests/api` runs 127 checks over every module (core, imgproc, photo, features2d, calib3d, video, objdetect, dnn, ml,
stitching and the Chinese aliases): `cd tests/api; aqua run .`. The example's `check.toml` runs all 26 labs without a
window: `aqua build check.toml -o check.wasm; aqua run check.wasm`.

## Assets and licenses · 資產與授權

`assets/` holds OpenCV's Haar cascades (frontal face, eye, smile; Intel License Agreement, in OpenCV's data folder),
the YuNet face model `face_detection_yunet_2023mar.onnx` (MIT, `licenses/YUNET-LICENSE`) and `portrait.jpg`, a test
portrait reused from the local MediaPipe/ONNX examples (`man_smiling.jpg` of yakhyo/mediapipe-face-mesh-onnx at
add50e0f; upstream image rights retained). The library is Apache-2.0 like OpenCV; OpenCV and its bundled third-party
licenses are in `licenses/`.
