# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build

# Include any dependencies generated for this target.
include CMakeFiles/tensorrt_inference.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tensorrt_inference.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tensorrt_inference.dir/flags.make

CMakeFiles/tensorrt_inference.dir/main.cpp.o: CMakeFiles/tensorrt_inference.dir/flags.make
CMakeFiles/tensorrt_inference.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tensorrt_inference.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tensorrt_inference.dir/main.cpp.o -c /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/main.cpp

CMakeFiles/tensorrt_inference.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tensorrt_inference.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/main.cpp > CMakeFiles/tensorrt_inference.dir/main.cpp.i

CMakeFiles/tensorrt_inference.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tensorrt_inference.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/main.cpp -o CMakeFiles/tensorrt_inference.dir/main.cpp.s

CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o: CMakeFiles/tensorrt_inference.dir/flags.make
CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o: /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o -c /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp

CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp > CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.i

CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp -o CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.s

# Object files for target tensorrt_inference
tensorrt_inference_OBJECTS = \
"CMakeFiles/tensorrt_inference.dir/main.cpp.o" \
"CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o"

# External object files for target tensorrt_inference
tensorrt_inference_EXTERNAL_OBJECTS =

/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: CMakeFiles/tensorrt_inference.dir/main.cpp.o
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: CMakeFiles/tensorrt_inference.dir/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/code/src/build.cpp.o
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: CMakeFiles/tensorrt_inference.dir/build.make
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/libresnet.so
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/libyolov5_cu.so
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/libyolov8.so
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/local/cuda-12.1/lib64/libcudart_static.a
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: /usr/lib/x86_64-linux-gnu/librt.so
/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference: CMakeFiles/tensorrt_inference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tensorrt_inference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tensorrt_inference.dir/build: /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/bin/tensorrt_inference

.PHONY : CMakeFiles/tensorrt_inference.dir/build

CMakeFiles/tensorrt_inference.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tensorrt_inference.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tensorrt_inference.dir/clean

CMakeFiles/tensorrt_inference.dir/depend:
	cd /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build /home/linkdome/mg/jx/yolov8-seg/tensorrt_inference_yolov8pose/project/build/CMakeFiles/tensorrt_inference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tensorrt_inference.dir/depend
