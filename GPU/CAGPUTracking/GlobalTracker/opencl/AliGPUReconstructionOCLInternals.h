//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliGPUTPCGPUTrackerOpenCL.h
//  @author David Rohr, Sergey Gorbunov
//  @date
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note


#ifndef ALIHLTTPCCAGPUTRACKEROPENCLINTERNALS_H
#define ALIHLTTPCCAGPUTRACKEROPENCLINTERNALS_H

#include <CL/opencl.h>
#include <CL/cl_ext.h>

static const char* opencl_error_string(int errorcode)
{
	switch (errorcode)
	{
		case CL_SUCCESS:                            return "Success!";
		case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
		case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES:                   return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
		case CL_MAP_FAILURE:                        return "Map failure";
		case CL_INVALID_VALUE:                      return "Invalid value";
		case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
		case CL_INVALID_PLATFORM:                   return "Invalid platform";
		case CL_INVALID_DEVICE:                     return "Invalid device";
		case CL_INVALID_CONTEXT:                    return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
		case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
		case CL_INVALID_SAMPLER:                    return "Invalid sampler";
		case CL_INVALID_BINARY:                     return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
		case CL_INVALID_PROGRAM:                    return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
		case CL_INVALID_KERNEL:                     return "Invalid kernel";
		case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
		case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
		case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
		case CL_INVALID_EVENT:                      return "Invalid event";
		case CL_INVALID_OPERATION:                  return "Invalid operation";
		case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
		default: return "Unknown Errorcode";
	}
}

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)

static int GPUFailedMsgA(int error, const char* file, int line)
{
	//Check for OPENCL Error and in the case of an error display the corresponding error string
	if (error == CL_SUCCESS) return(0);
	printf("OCL Error: %d / %s (%s:%d)\n", error, opencl_error_string(error), file, line);
	return(1);
}

static inline int OCLsetKernelParameters_helper(cl_kernel &k,int i) {return 0;}
 
template<typename T, typename... Args> static inline int OCLsetKernelParameters_helper(cl_kernel &kernel, int i, const T &firstParameter, const Args& ...restOfParameters)
{
	GPUFailedMsg(clSetKernelArg(kernel, i, sizeof(T), &firstParameter));
	return OCLsetKernelParameters_helper(kernel, i + 1, restOfParameters...);
}
 
template<typename... Args> static inline int OCLsetKernelParameters(cl_kernel &kernel, const Args& ...args)
{
	return OCLsetKernelParameters_helper(kernel, 0, args...);
}

static inline int clExecuteKernelA(cl_command_queue queue, cl_kernel krnl, size_t local_size, size_t global_size, cl_event* pEvent, cl_event* wait = NULL, cl_int nWaitEvents = 1)
{
	GPUFailedMsg(clEnqueueNDRangeKernel(queue, krnl, 1, NULL, &global_size, &local_size, wait == NULL ? 0 : nWaitEvents, wait, pEvent));
	return 0;
}

struct AliGPUReconstructionOCLInternals
{
	cl_device_id device;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue command_queue[36];
	char cl_queue_event_done[36];
	cl_mem mem_gpu;
	cl_mem mem_constant;
	cl_mem mem_host;
	cl_event* selector_events;
	cl_program program;

	cl_kernel kernel_neighbours_finder, kernel_neighbours_cleaner, kernel_start_hits_finder, kernel_start_hits_sorter, kernel_tracklet_constructor, kernel_tracklet_selector, kernel_row_blocks;
};

static_assert(std::is_convertible<cl_event, void*>::value, "OpenCL event type incompatible to deviceEvent");

#endif
