// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file makefile_opencl_compiler.cxx
/// \author David Rohr

#define _CRT_SECURE_NO_WARNINGS
#include "CL/opencl.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "opencl_compiler_structs.h"

#define quit(arg)              \
  {                            \
    fprintf(stderr, arg "\n"); \
    return (1);                \
  }
#define DEFAULT_OPENCL_COMPILER_OPTIONS ""
#define DEFAULT_OUTPUT_FILE "opencl.out"

int main(int argc, char** argv)
{
  const char* output_file = DEFAULT_OUTPUT_FILE;
  std::string compiler_options = DEFAULT_OPENCL_COMPILER_OPTIONS;
  std::vector<char*> files;

  printf("Passing command line options:\n");
  bool add_option = false;
  for (int i = 1; i < argc; i++) {
    if (add_option) {
      compiler_options += " ";
      compiler_options += argv[i];
    } else if (strcmp(argv[i], "--") == 0) {
      add_option = true;
    } else if (strcmp(argv[i], "-output-file") == 0) {
      if (++i >= argc) {
        quit("Output file name missing");
      }
      output_file = argv[i];
    } else {
      fprintf(stderr, "%s\n", argv[i]);
      files.push_back(argv[i]);
    }
  }

  cl_int ocl_error;
  cl_uint num_platforms;
  if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS) {
    quit("Error getting OpenCL Platform Count");
  }
  if (num_platforms == 0) {
    quit("No OpenCL Platform found");
  }
  printf("%d OpenCL Platforms found\n", num_platforms);

  // Query platforms
  cl_platform_id* platforms = new cl_platform_id[num_platforms];
  if (platforms == nullptr) {
    quit("Memory allocation error");
  }
  if (clGetPlatformIDs(num_platforms, platforms, nullptr) != CL_SUCCESS) {
    quit("Error getting OpenCL Platforms");
  }

  cl_platform_id platform;
  bool found = false;

  _makefiles_opencl_platform_info pinfo;
  for (unsigned int i_platform = 0; i_platform < num_platforms; i_platform++) {
    clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_PROFILE, 64, pinfo.platform_profile, nullptr);
    clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VERSION, 64, pinfo.platform_version, nullptr);
    clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_NAME, 64, pinfo.platform_name, nullptr);
    clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VENDOR, 64, pinfo.platform_vendor, nullptr);
    printf("Available Platform %u: (%s %s) %s %s\n", i_platform, pinfo.platform_profile, pinfo.platform_version, pinfo.platform_vendor, pinfo.platform_name);
    if (strcmp(pinfo.platform_vendor, "Advanced Micro Devices, Inc.") == 0 && strcmp(pinfo.platform_version, "OpenCL 2.0 AMD-APP (1800.8)") == 0) {
      found = true;
      printf("AMD OpenCL Platform found (%u)\n", i_platform);
      platform = platforms[i_platform];
      break;
    }
  }
  if (found == false) {
    quit("Did not find AMD OpenCL Platform");
  }

  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &pinfo.count) != CL_SUCCESS) {
    quit("Error getting OPENCL Device Count");
  }

  // Query devices
  cl_device_id* devices = new cl_device_id[pinfo.count];
  if (devices == nullptr) {
    quit("Memory allocation error");
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, pinfo.count, devices, nullptr) != CL_SUCCESS) {
    quit("Error getting OpenCL devices");
  }

  _makefiles_opencl_device_info dinfo;
  cl_device_type device_type;
  cl_uint freq, shaders;

  printf("Available OPENCL devices:\n");
  for (unsigned int i = 0; i < pinfo.count; i++) {
    printf("Examining device %u\n", i);

    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, dinfo.device_name, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, dinfo.device_vendor, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(dinfo.nbits), &dinfo.nbits, nullptr);
    printf("Found Device %u : %s %s (Frequency %d, Shaders %d, %d bit)\n", i, dinfo.device_vendor, dinfo.device_name, (int)freq, (int)shaders, (int)dinfo.nbits);
  }

  if (files.size() == 0) {
    quit("Syntax: opencl [-output-file OUTPUT_FILE] FILE1 [FILE2] ... [FILEn] [-- COMPILER_OPTION_1] [COMPILER_OPTION_2] ... [COMPILER_OPTION_N]");
  }

  char** buffers = (char**)malloc(files.size() * sizeof(char*));
  if (buffers == nullptr) {
    quit("Memory allocation error\n");
  }
  for (unsigned int i = 0; i < files.size(); i++) {
    printf("Reading source file %s\n", files[i]);
    FILE* fp = fopen(files[i], "rb");
    if (fp == nullptr) {
      printf("Cannot open %s\n", files[i]);
      free(buffers);
      return (1);
    }
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    buffers[i] = (char*)malloc(file_size + 1);
    if (buffers[i] == nullptr) {
      quit("Memory allocation error");
    }
    if (fread(buffers[i], 1, file_size, fp) != file_size) {
      quit("Error reading file");
    }
    buffers[i][file_size] = 0;
    fclose(fp);
  }

  printf("Creating OpenCL Context\n");
  // Create OpenCL context
  cl_context context = clCreateContext(nullptr, pinfo.count, devices, nullptr, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    quit("Error creating OpenCL context");
  }

  printf("Creating OpenCL Program Object\n");
  // Create OpenCL program object
  cl_program program = clCreateProgramWithSource(context, (cl_uint)files.size(), (const char**)buffers, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    quit("Error creating program object");
  }

  printf("Compiling OpenCL Program\n");
  // Compile program
  ocl_error = clBuildProgram(program, pinfo.count, devices, compiler_options.c_str(), nullptr, nullptr);
  if (ocl_error != CL_SUCCESS) {
    fprintf(stderr, "OpenCL Error while building program: %d (Compiler options: %s)\n", ocl_error, compiler_options.c_str());
    fprintf(stderr, "OpenCL Kernel:\n\n");
    for (unsigned int i = 0; i < files.size(); i++) {
      printf("%s\n\n", buffers[i]);
    }

    for (unsigned int i = 0; i < pinfo.count; i++) {
      cl_build_status status;
      clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr);
      if (status == CL_BUILD_ERROR) {
        size_t log_size;
        clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        char* build_log = (char*)malloc(log_size + 1);
        if (build_log == nullptr) {
          quit("Memory allocation error");
        }
        clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, log_size, build_log, nullptr);
        fprintf(stderr, "Build Log (device %d):\n\n%s\n\n", i, build_log);
        free(build_log);
      }
    }
  }
  for (unsigned int i = 0; i < files.size(); i++) {
    free(buffers[i]);
  }
  free(buffers);
  if (ocl_error != CL_SUCCESS) {
    return (1);
  }

  printf("Obtaining program binaries\n");
  size_t* binary_sizes = (size_t*)malloc(pinfo.count * sizeof(size_t));
  if (binary_sizes == nullptr) {
    quit("Memory allocation error");
  }
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, pinfo.count * sizeof(size_t), binary_sizes, nullptr);
  char** binary_buffers = (char**)malloc(pinfo.count * sizeof(char*));
  if (binary_buffers == nullptr) {
    quit("Memory allocation error");
  }
  for (unsigned int i = 0; i < pinfo.count; i++) {
    printf("Binary size for device %d: %d\n", i, (int)binary_sizes[i]);
    binary_buffers[i] = (char*)malloc(binary_sizes[i]);
    memset(binary_buffers[i], 0, binary_sizes[i]);
    if (binary_buffers[i] == nullptr) {
      quit("Memory allocation error");
    }
  }
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, pinfo.count * sizeof(char*), binary_buffers, nullptr);

  printf("Programs obtained successfully, cleaning up opencl\n");
  clReleaseProgram(program);
  clReleaseContext(context);

  printf("Writing binaries to file (%s)\n", output_file);
  FILE* fp;
  fp = fopen(output_file, "w+b");
  if (fp == nullptr) {
    quit("Error opening output file\n");
  }
  const char* magic_bytes = "QOCLPB";
  fwrite(magic_bytes, 1, strlen(magic_bytes) + 1, fp);
  fwrite(&pinfo, 1, sizeof(pinfo), fp);
  for (unsigned int i = 0; i < pinfo.count; i++) {
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, dinfo.device_name, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, dinfo.device_vendor, nullptr);
    dinfo.binary_size = binary_sizes[i];
    fwrite(&dinfo, 1, sizeof(dinfo), fp);
    fwrite(binary_buffers[i], 1, binary_sizes[i], fp);
  }
  fclose(fp);

  printf("All done, cleaning up remaining buffers\n");
  for (unsigned int i = 0; i < pinfo.count; i++) {
    free(binary_buffers[i]);
  }
  free(binary_sizes);
  free(binary_buffers);

  return (0);
}
