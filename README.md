# Aqua_OpenCV 2.0.0

OpenCV 4.12.0 for the current AquariusLang SDK. Import `OpenCV` or `電腦視覺` after declaring a local Bottle dependency.
The same C++ implementation supplies all 45 algorithms on Windows, Linux, Android and the browser.

```powershell
aqua bottle install ../Aqua_OpenCV --project .
aqua build . --target web -o web
aqua run .
```

The native module registers through `AQUA_LIBRARY` and the SDK function table. RGBA pixels and model bytes cross as
checked byte arrays; image handles remain in OpenCV. The public Images SDK decodes input images and encodes PNGs.
Sessions belong to each Aquarius runtime and are destroyed with it. `Release` frees images; `Dispose` closes the session.
Limits: 32 sessions, 128 images per session, 16 megapixels per image and 128 MiB per typed byte input.

`Apply` returns `{ok,value}` or `{ok:false,error}`; typed input errors raise normal Aquarius runtime errors.
`Capabilities().features` lists all 45 implemented feature codes. This binding does not expose arbitrary OpenCV or contrib APIs.
Native builds use optimized portable CPU kernels; browser execution is single-threaded C++ Wasm with SIMD. OpenCL availability
is reported separately and is not an acceleration promise. No system OpenCV, Python, Node runtime or MSVC runtime is required.

Build with `aqua library build .` on the host, or pass `--targets` with SDK target names. Linux builds run on Linux/WSL.
CMake verifies the vendored OpenCV archive, applies the documented portability patch, and uses the SDK's pinned Clang toolchains.
The browser module includes a private LLVM exception runtime so OpenCV exceptions become API errors.
Pinned sources, hashes and license provenance are in `native/vendor/README.md`. Generated source and build directories are ignored.
The current SDK does not provide macOS binaries, so this version does not advertise macOS support.

Tests under `tests/` exercise all 45 algorithms, real Haar/YuNet face detection, QR decoding, PNG round trips, bilingual
results, byte validation, ownership, closed sessions and OpenCV exception recovery. Executed on Windows, Linux/WSL,
Chromium and Android x86_64; Android arm64 is cross-compiled and still needs a physical-device run.

## 繁體中文 / English API

所有公開影像操作皆提供中英文別名。45 個功能名稱可用英文代碼或繁體中文名稱；`FeatureNames / 功能名稱` 列出對照。`Apply / 處理(影像, 功能, 參數)` 及下表的便利函式 `(影像, 參數)` 使用相同實作，無參數時傳 `{}`。

```aqua
變數 視覺 = 匯入("OpenCV");
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
