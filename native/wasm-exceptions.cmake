set(llvm "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llvm-project-23.1.3")
target_link_options(aqua_opencv PRIVATE -Wl,--Bsymbolic)
set(abi_sources cxa_aux_runtime cxa_default_handlers cxa_demangle cxa_exception_storage cxa_guard cxa_handlers
    cxa_vector cxa_virtual stdlib_exception stdlib_stdexcept stdlib_typeinfo abort_message fallback_malloc
    private_typeinfo cxa_exception cxa_personality)
foreach(name IN LISTS abi_sources)
    target_sources(aqua_opencv PRIVATE "${llvm}/libcxxabi/src/${name}.cpp")
endforeach()
target_sources(aqua_opencv PRIVATE "${llvm}/libunwind/src/Unwind-wasm.c")
target_include_directories(aqua_opencv PRIVATE "${llvm}/libcxxabi/include" "${llvm}/libcxx/src"
    "${llvm}/libunwind/include" "${llvm}/libunwind/src")
target_compile_definitions(aqua_opencv PRIVATE _LIBCXXABI_BUILDING_LIBRARY _LIBCXXABI_HAS_NO_THREADS _LIBCPP_BUILDING_LIBRARY _LIBUNWIND_HIDE_SYMBOLS _LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS)
# The setjmp/longjmp runtime (__wasm_setjmp, __wasm_longjmp) of the pinned WASI sysroot, private to this module.
target_link_libraries(aqua_opencv PRIVATE "${AQUA_WASI_SYSROOT}/lib/wasm32-wasip1/libsetjmp.a")
