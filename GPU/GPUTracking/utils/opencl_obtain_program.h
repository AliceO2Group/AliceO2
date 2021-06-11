// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file opencl_obtain_program.h
/// \author David Rohr

#ifndef MAKEFILES_OPENCL_OBTAIN_PROGRAMH
#define MAKEFILES_OPENCL_OBTAIN_PROGRAMH

#include <CL/opencl.h>
#include <vector>
#include "opencl_compiler_structs.h"

static int _makefiles_opencl_obtain_program_helper(cl_context context, cl_uint num_devices, cl_device_id* devices, cl_program* program, char* binaries)
{
  const char* magic_bytes = "QOCLPB";
  if (strncmp(magic_bytes, binaries, strlen(magic_bytes)) != 0) {
    printf("Internal error accessing opencl program\n");
    return (1);
  }
  char* current_ptr = binaries + strlen(magic_bytes) + 1;
  _makefiles_opencl_platform_info* pinfo = (_makefiles_opencl_platform_info*)current_ptr;
  current_ptr += sizeof(_makefiles_opencl_platform_info);

  if (num_devices != pinfo->count) {
    printf("Number of devices differs from number of devices in opencl program\n");
    return (1);
  }
  // printf("Obtaining program for OpenCL Platform: (%s %s) %s %s\n", pinfo->platform_profile, pinfo->platform_version, pinfo->platform_vendor, pinfo->platform_name);

  std::vector<size_t> program_sizes(pinfo->count);
  std::vector<char*> program_binaries(pinfo->count);

  for (unsigned int i = 0; i < pinfo->count; i++) {
    char device_name[64], device_vendor[64];
    cl_uint nbits;
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64, device_name, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 64, device_vendor, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, nullptr);
    _makefiles_opencl_device_info* dinfo = (_makefiles_opencl_device_info*)current_ptr;
    if (strcmp(device_name, dinfo->device_name) != 0 || strcmp(device_vendor, dinfo->device_vendor) != 0) {
      printf("Device list is different to device list from opencl program (Device %d: '%s - %s' != '%s - %s')\n", i, device_vendor, device_name, dinfo->device_vendor, dinfo->device_name);
      return (1);
    }
    if (nbits != dinfo->nbits) {
      printf("Pointer size of device and stored device binary differs\n");
      return (1);
    }
    current_ptr += sizeof(_makefiles_opencl_device_info);
    // printf("Device %d: %s %s (size %lld)\n", i, dinfo->device_vendor, dinfo->device_name, (long long int) dinfo->binary_size);
    program_sizes[i] = dinfo->binary_size;
    program_binaries[i] = current_ptr;
    current_ptr += dinfo->binary_size;
  }

  cl_int return_status[pinfo->count];
  cl_int ocl_error;
  *program = clCreateProgramWithBinary(context, num_devices, devices, program_sizes.data(), (const unsigned char**)program_binaries.data(), return_status, &ocl_error);

  if (ocl_error != CL_SUCCESS) {
    printf("Error loading program\n");
    return (1);
  }

  for (unsigned int i = 0; i < pinfo->count; i++) {
    if (return_status[i] != CL_SUCCESS) {
      printf("Error loading program for device %d\n", i);
      clReleaseProgram(*program);
      return (1);
    }
  }

  ocl_error = clBuildProgram(*program, num_devices, devices, "", nullptr, nullptr);
  if (ocl_error != CL_SUCCESS) {
    printf("Error building program\n");
    clReleaseProgram(*program);
    return (1);
  }

  return (0);
}

#endif
