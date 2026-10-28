include(FetchContent)
set(opencv_archive "${CMAKE_CURRENT_LIST_DIR}/vendor/opencv-4.12.0.tar.gz")
set(opencv_sha 44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d)
file(SHA256 "${opencv_archive}" actual_sha)
if(NOT actual_sha STREQUAL opencv_sha)
    message(FATAL_ERROR "OpenCV source archive SHA-256 mismatch")
endif()
set(opencv_source "${CMAKE_CURRENT_LIST_DIR}/vendor/opencv-4.12.0")
if(NOT EXISTS "${opencv_source}/CMakeLists.txt")
    FetchContent_Declare(opencv_source URL "${opencv_archive}" URL_HASH "SHA256=${opencv_sha}"
        SOURCE_DIR "${opencv_source}")
    FetchContent_Populate(opencv_source)
endif()
find_package(Git REQUIRED)
set(opencv_patch "${CMAKE_CURRENT_LIST_DIR}/vendor/opencv-wasi.patch")
# The ceiling stops git from treating the source as part of an enclosing repository, where it skips the patch silently;
# whitespace is ignored so a tree patched on Windows (CRLF) is recognized on Linux and the other way round.
set(git_apply "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_LIST_DIR}/vendor" "${GIT_EXECUTABLE}" -c core.autocrlf=false
    apply --ignore-whitespace)
execute_process(COMMAND ${git_apply} --reverse --check "${opencv_patch}"
    WORKING_DIRECTORY "${opencv_source}" RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
if(NOT applied EQUAL 0)
    execute_process(COMMAND ${git_apply} "${opencv_patch}"
        WORKING_DIRECTORY "${opencv_source}" COMMAND_ERROR_IS_FATAL ANY)
endif()
