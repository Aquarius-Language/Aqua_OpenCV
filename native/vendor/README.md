OpenCV 4.12.0: https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
SHA-256: 44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d
The archive includes OpenCV and its bundled third-party licenses. CMake verifies it, extracts an ignored source tree,
and applies opencv-wasi.patch: explicit iterator includes for current libc++; enable the exposed objdetect/DNN
algorithms without background threads on WASI; no-op cv::Mutex in that single-threaded build.
DetectionBasedTracker is excluded on the web because its background worker is outside this binding's API.

LLVM 23.1.3: https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-23.1.3.tar.gz
Source archive SHA-256: 791b560dee8f65fdf7b19ad5db0b4ac627467824d8b687957d8a437b806103f0
llvm-project-23.1.3 contains the libc++abi/libunwind sources and libc++ internal headers needed by
wasm-exceptions.cmake. Their LICENSE.TXT files are retained. The web build also enables WebAssembly setjmp/longjmp
(libjpeg and libpng error handling) and links the pinned WASI sysroot's libsetjmp privately. The exception runtime is private to this side module;
it does not replace AquariusLang's shared runtime or its libc++.
