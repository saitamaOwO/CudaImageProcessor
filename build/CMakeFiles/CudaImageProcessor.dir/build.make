# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yash/Desktop/Project/CudaImageProcessor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yash/Desktop/Project/CudaImageProcessor/build

# Include any dependencies generated for this target.
include CMakeFiles/CudaImageProcessor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CudaImageProcessor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CudaImageProcessor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CudaImageProcessor.dir/flags.make

CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o: CMakeFiles/CudaImageProcessor.dir/flags.make
CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o: /home/yash/Desktop/Project/CudaImageProcessor/src/main.cpp
CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o: CMakeFiles/CudaImageProcessor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yash/Desktop/Project/CudaImageProcessor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o -MF CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o.d -o CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o -c /home/yash/Desktop/Project/CudaImageProcessor/src/main.cpp

CMakeFiles/CudaImageProcessor.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CudaImageProcessor.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yash/Desktop/Project/CudaImageProcessor/src/main.cpp > CMakeFiles/CudaImageProcessor.dir/src/main.cpp.i

CMakeFiles/CudaImageProcessor.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CudaImageProcessor.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yash/Desktop/Project/CudaImageProcessor/src/main.cpp -o CMakeFiles/CudaImageProcessor.dir/src/main.cpp.s

CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o: CMakeFiles/CudaImageProcessor.dir/flags.make
CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o: CMakeFiles/CudaImageProcessor.dir/includes_CUDA.rsp
CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o: /home/yash/Desktop/Project/CudaImageProcessor/src/cudaOperations.cu
CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o: CMakeFiles/CudaImageProcessor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yash/Desktop/Project/CudaImageProcessor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o -MF CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o.d -x cu -c /home/yash/Desktop/Project/CudaImageProcessor/src/cudaOperations.cu -o CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o

CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target CudaImageProcessor
CudaImageProcessor_OBJECTS = \
"CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o" \
"CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o"

# External object files for target CudaImageProcessor
CudaImageProcessor_EXTERNAL_OBJECTS =

CudaImageProcessor: CMakeFiles/CudaImageProcessor.dir/src/main.cpp.o
CudaImageProcessor: CMakeFiles/CudaImageProcessor.dir/src/cudaOperations.cu.o
CudaImageProcessor: CMakeFiles/CudaImageProcessor.dir/build.make
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_cvv.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.6.0
CudaImageProcessor: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.6.0
CudaImageProcessor: CMakeFiles/CudaImageProcessor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yash/Desktop/Project/CudaImageProcessor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CudaImageProcessor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CudaImageProcessor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CudaImageProcessor.dir/build: CudaImageProcessor
.PHONY : CMakeFiles/CudaImageProcessor.dir/build

CMakeFiles/CudaImageProcessor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CudaImageProcessor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CudaImageProcessor.dir/clean

CMakeFiles/CudaImageProcessor.dir/depend:
	cd /home/yash/Desktop/Project/CudaImageProcessor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yash/Desktop/Project/CudaImageProcessor /home/yash/Desktop/Project/CudaImageProcessor /home/yash/Desktop/Project/CudaImageProcessor/build /home/yash/Desktop/Project/CudaImageProcessor/build /home/yash/Desktop/Project/CudaImageProcessor/build/CMakeFiles/CudaImageProcessor.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/CudaImageProcessor.dir/depend

