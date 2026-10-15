# Aqua_OpenCV

Bottle package 1.1.1. Import `/bottles/Aqua_OpenCV/main.aqua`.

```powershell
aqua bottle install C:/OfficialProjects/Aqua_OpenCV --global
aqua bottle install C:/OfficialProjects/Aqua_OpenCV --project .
aqua build . --platform windows -o dist/app.wasm
aqua run dist/app.wasm
aqua build . --target web -o web
```

`Sample(frame)` and `Load(assetPath)` return opaque image IDs. `Apply(image,feature,parameters)` returns `{ok:true,value:{image,milliseconds,stats}}` or `{ok:false,error}`. `Inspect`, `EncodePNG`, `Preview`, `Release` and `Dispose` manage output and ownership. `Capabilities().features` is authoritative for the current backend. `LoadCascade` and `LoadYuNet` load the bundled model assets. Always release old output images and dispose the session.

Images remain in OpenCV memory between calls. Requests carry small handles; full frames cross the language boundary only on input and display/export. Windows and Android ship a Release C++ bridge against official OpenCV 4.12.0 (optimized CPU dispatch, parallel kernels, IPP where enabled by that binary). `Configure(threads)` adjusts the native thread count. Small image operations use CPU; GPU/OpenCL/CUDA speedups are workload-dependent and are not advertised without measurement. OpenCL availability is reported separately from actual use; this bridge uses `cv::Mat`, not `UMat`.

The self-contained portable ES module uses a locally compiled OpenCV.js 4.12.0 Release Wasm SIMD128 build (Emscripten 3.1.74) (SHA-256 in `vendor/SHA256SUMS`). Browsers and Linux/macOS can use it without installing Python or spawning a process. This build uses `-O3 -msimd128` and OpenCV universal intrinsics, with one thread for ordinary static hosting. Wasm SIMD support is required. `vendor/build-info.json` records the build provenance. SIFT, stitching, stereo, QR generation and some photo/DNN APIs are available in the native build only. The showcase displays actual support.

## 繁體中文 / English API

所有公開影像操作皆提供中英文別名。45 個功能名稱可用英文代碼或繁體中文名稱；`FeatureNames / 功能名稱` 列出對照。`Apply / 處理(影像, 功能, 參數)` 及下表的便利函式 `(影像, 參數)` 使用相同實作，無參數時傳 `{}`。

```aqua
變數 視覺 = 匯入("/bottles/Aqua_OpenCV/main.aqua");
變數 原圖 = 視覺.範例影像(12);
變數 回應 = 視覺.Canny邊緣(原圖, {"閾值":100,"核心大小":5});
如果(回應["成功"]){
    變數 結果 = 回應["結果"];
    印出(結果["毫秒"], 結果["統計"]);
    視覺.釋放影像(結果["影像"]);
}否則{印出(回應["錯誤"],回應["診斷"]);}
視覺.釋放影像(原圖);
視覺.釋放();
```

中文參數：`閾值 / threshold`、`核心大小 / kernel`（1–99 奇數）、`角度 / angle`、`寬度 / width`、`高度 / height`。混用時中文欄位優先，不修改呼叫者的參數。零與 false 值均保留。

結果同時保留 `{ok,value:{image,milliseconds,stats}}` 與 `{成功,結果:{影像,毫秒,統計}}`。`統計` 使用中文欄位；`stats` 保留原始英文。錯誤包含中文 `錯誤`、原始 `診斷 / error`。`Inspect / 檢查` 提供 `寬度、 高度、通道數` 及英文欄位。`Capabilities / 功能支援` 的 `功能` 為中文清單，`features` 為英文清單；`後端、版本、執行緒數、已最佳化、影像數` 與英文欄位並存。

其他雙語函式：`Configure / 設定`、`Sample / 範例影像`、`Load / 載入`、`Decode / 解碼`、`EncodePNG / 編碼PNG`、`Preview / 預覽`、`LoadCascade / 載入人臉分類器`、`LoadYuNet / 載入神經網路`、`Release / 釋放影像`、`Dispose / 釋放`。輸入與輸出是影像代碼，連續運算不會複製完整影像；請釋放不用的影像及工作階段。

| 英文代碼 | English function | 繁體中文功能 / 函式 |
| --- | --- | --- |
| `gray` | `Grayscale` | `灰階` |
| `hsv` | `HSV` | `HSV色彩空間` |
| `gaussian` | `GaussianBlur` | `高斯模糊` |
| `median` | `MedianBlur` | `中值濾波` |
| `bilateral` | `BilateralFilter` | `雙邊濾波` |
| `canny` | `Canny` | `Canny邊緣` |
| `sobel` | `Sobel` | `Sobel梯度` |
| `laplacian` | `Laplacian` | `Laplacian邊緣` |
| `threshold` | `Threshold` | `二值化` |
| `adaptive` | `AdaptiveThreshold` | `自適應二值化` |
| `otsu` | `OtsuThreshold` | `Otsu二值化` |
| `erode` | `Erode` | `侵蝕` |
| `dilate` | `Dilate` | `膨脹` |
| `open` | `MorphologicalOpen` | `開運算` |
| `close` | `MorphologicalClose` | `閉運算` |
| `gradient` | `MorphologicalGradient` | `形態學梯度` |
| `equalize` | `EqualizeHistogram` | `直方圖等化` |
| `clahe` | `CLAHE` | `局部對比增強` |
| `colormap` | `ColorMap` | `偽彩色映射` |
| `resize` | `Resize` | `調整影像大小` |
| `rotate` | `Rotate` | `仿射旋轉` |
| `perspective` | `Perspective` | `透視轉換` |
| `contours` | `Contours` | `輪廓偵測` |
| `hough_lines` | `HoughLines` | `霍夫直線` |
| `hough_circles` | `HoughCircles` | `霍夫圓形` |
| `corners` | `Corners` | `角點偵測` |
| `orb` | `ORB` | `ORB特徵` |
| `sift` | `SIFT` | `SIFT特徵` |
| `matching` | `MatchDescriptors` | `特徵匹配` |
| `template` | `MatchTemplate` | `樣板匹配` |
| `watershed` | `Watershed` | `分水嶺分割` |
| `grabcut` | `GrabCut` | `GrabCut前景` |
| `kmeans` | `KMeans` | `K平均量化` |
| `dft` | `FourierSpectrum` | `傅立葉頻譜` |
| `inpaint` | `Inpaint` | `影像修補` |
| `denoise` | `Denoise` | `非局部均值去噪` |
| `optical_flow` | `OpticalFlow` | `稠密光流` |
| `background` | `BackgroundSubtraction` | `背景相減` |
| `face` | `DetectFaces` | `人臉偵測` |
| `qr` | `QRCode` | `QR碼編解碼` |
| `stereo` | `StereoDisparity` | `立體視差` |
| `homography` | `Homography` | `單應性估計` |
| `calibration` | `CameraPose` | `相機姿態估計` |
| `stitching` | `StitchPanorama` | `全景拼接` |
| `dnn` | `DetectFacesDNN` | `神經網路人臉` |

Windows 與 Android 內附最佳化 C++ 原生庫；網頁、Linux 與 macOS 使用 Release SIMD128 Wasm。後兩者尚未內附原生二進位檔，不能宣稱已測得原生效能。網頁提供 36 項、原生提供 45 項，以 `功能支援()` 為準。這套示範涵蓋主要 OpenCV 類別，不包含所有 API、contrib 模組或任意 DNN 模型。YuNet 介面專用於人臉模型。

## Native platform builds

Build OpenCV 4.12 with `CMAKE_BUILD_TYPE=Release`, CPU dispatch appropriate to the deployment fleet (AVX2 on supported x86, NEON on ARM), and a parallel backend such as TBB/pthreads. Keep a portable baseline; avoid `-march=native` for distributed packages. Build this bridge with `cmake -S . -B build -DOpenCV_DIR=...; cmake --build build --config Release; cmake --install build --prefix runtimes/PLATFORM`. Ship OpenCV shared dependencies beside the bridge and preserve their licenses. Select the resulting path in `platforms.aqua`, and add an asset table filtered to that platform, removing that platform from the portable asset filter. Bump the Bottle version before reinstalling changed artifacts.

For Android use the NDK CMake toolchain, `ANDROID_ABI=arm64-v8a` or `x86_64`, `ANDROID_PLATFORM=android-26`, `ANDROID_STL=c++_static`, and the matching OpenCV Android SDK. Put bridge and dependencies under `native/android/ABI/` and declare application-root targets with the same path. Select `libaqua_opencv.so` in `platforms.aqua`; the Android loader uses its packaged linker namespace. Native Android ARM64 and x86_64 binaries are bundled, statically linking OpenCV and libc++, with NEON/x86 CPU dispatch and 16 KB ELF alignment. All 45 operations passed on the x86_64 emulator. Native Linux/macOS artifacts are not bundled in this snapshot; those hosts use the optimized SIMD Wasm implementation. Build the C++ bridge on those platforms before claiming native performance there.

The bundled browser build already enables SIMD. For deployments that can serve isolated pthread workers, build with OpenCV's `platforms/js/build_js.py --simd --threads`, and serve the pthread worker assets with COOP/COEP headers. Replace the vendor snapshot, rebuild the self-contained bundle and validate its feature list and performance on the target device. See https://docs.opencv.org/4.12.0/d4/da1/tutorial_js_setup.html and https://docs.opencv.org/4.12.0/db/d05/tutorial_config_reference.html.

Native operations: color conversion, Gaussian/median/bilateral filtering, Canny/Sobel/Laplacian, global/adaptive/Otsu thresholding, morphology, histogram equalization/CLAHE, color maps, resize/rotation/perspective, contours, Hough lines/circles, corners, ORB/SIFT, descriptor matching, templates, watershed, GrabCut, k-means, Fourier spectrum, inpainting, denoising, optical flow, MOG2 subtraction, Haar faces, QR encoding/decoding, stereo disparity, homography, pose estimation/calibration primitives, stitching, and YuNet DNN face detection. Deterministic synthetic inputs make algorithms runnable without camera hardware. These demonstrations cover major OpenCV families, not every API or contrib module.

Limits: 32 sessions, 128 images/session, 16 megapixels/image, 1 MiB per encoded transfer (Aquarius ABI). Request errors are returned as data; invalid handles and decode/encode failures become runtime errors. Native module state is shared by OS loads; callers must dispose sessions. Backend unloading releases its remaining state when the last OS reference closes.

OpenCV and YuNet use the included licenses; nlohmann/json 3.12.0 uses MIT. Sample portrait and Haar cascade come from the official OpenCV 4.12 source tree. No automatic install/build hooks execute through Bottle.


## Rebuilding the shipped SIMD artifact

Extract the official OpenCV 4.12.0 sources and activate Emscripten 3.1.74. Run `python platforms/js/build_js.py build_simd --simd --build_wasm --config_only --cmake_option=-GNinja --cmake_option=-DENABLE_CCACHE=OFF --cmake_option=-DOPENCV_PYTHON3_VERSION=3.12`, then `cmake --build build_simd --target opencv.js --parallel 8`. Copy `build_simd/bin/opencv.js` to `vendor/opencv.original.js`; `npm ci && npm run build:web` applies the strict-module wrapper and bundles the factory. `wasm-opt --enable-simd --print-features` confirms the Wasm SIMD requirement. The script does not build or execute during Bottle installation. The worker-free default works with ordinary localhost/HTTPS serving.

Validation: `python tests/native.py` executes all 45 native operations with real image/face/QR/pose assertions and ownership checks; `npm test` executes every portable operation plus chunked RGBA transport and cleanup. `Aqua_OpenCV_Showcase/功能驗證.aqua` additionally verifies the actual compiled Aquarius bindings, codecs and platform loader. Preview creates a bounded thumbnail inside OpenCV before encoding so display stays within the ABI transfer limit; full-resolution encoded transfers still observe the 1 MiB ABI limit.
