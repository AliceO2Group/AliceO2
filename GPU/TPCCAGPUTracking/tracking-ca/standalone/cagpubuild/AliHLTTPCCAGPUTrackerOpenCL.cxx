// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//					David Rohr <drohr@kip.uni-heidelberg.de>				*
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#define __OPENCL__
#define RADEON
#define HLTCA_HOSTCODE

#include <string.h>
#include "AliHLTTPCCAGPUTrackerOpenCL.h"
#include "AliHLTTPCCAGPUTrackerOpenCLInternals.h"
#include "AliHLTTPCCAGPUTrackerCommon.h"

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCATrack.h" 

#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATracker.h"

#include "AliHLTTPCCAProcess.h"

#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCAStartHitsSorter.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "AliHLTTPCCAClusterData.h"

#include "../makefiles/opencl_obtain_program.h"
extern "C" char _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl[];

ClassImp( AliHLTTPCCAGPUTrackerOpenCL )

AliHLTTPCCAGPUTrackerOpenCL::AliHLTTPCCAGPUTrackerOpenCL() : ocl(NULL)
{
	ocl = new AliHLTTPCCAGPUTrackerOpenCLInternals;
	if (ocl == NULL)
	{
		HLTError("Memory Allocation Error");
	}
	ocl->mem_host_ptr = NULL;
	ocl->selector_events = NULL;
	ocl->devices = NULL;
};

AliHLTTPCCAGPUTrackerOpenCL::~AliHLTTPCCAGPUTrackerOpenCL()
{
	delete[] ocl;
};

#define quit(msg) {HLTError(msg);return(1);} 

int AliHLTTPCCAGPUTrackerOpenCL::InitGPU_Runtime(int sliceCount, int forceDeviceID)
{
	//Find best OPENCL device, initialize and allocate memory

	cl_int ocl_error;
	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) quit("Error getting OpenCL Platform Count");
	if (num_platforms == 0) quit("No OpenCL Platform found");
	if (fDebugLevel >= 2) HLTInfo("%d OpenCL Platforms found", num_platforms);
	
	//Query platforms
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == NULL) quit("Memory allocation error");
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) quit("Error getting OpenCL Platforms");

	cl_platform_id platform;
	bool found = false;
	for (unsigned int i_platform = 0;i_platform < num_platforms;i_platform++)
	{
		char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_PROFILE, 64, platform_profile, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VERSION, 64, platform_version, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_NAME, 64, platform_name, NULL);
		clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL);
		if (fDebugLevel >= 2) {HLTDebug("Available Platform %d: (%s %s) %s %s\n", i_platform, platform_profile, platform_version, platform_vendor, platform_name);}
		if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0)
		{
			found = true;
			if (fDebugLevel >= 2) HLTInfo("AMD OpenCL Platform found");
			platform = platforms[i_platform];
			break;
		}
	}
	if (found == false)
	{
		HLTError("Did not find AMD OpenCL Platform");
		return(1);
	}

	cl_uint count, bestDevice = (cl_uint) -1;
	long long int bestDeviceSpeed = 0, deviceSpeed;
	if (GPUFailedMsg(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &count)))
	{
		HLTError("Error getting OPENCL Device Count");
		return(1);
	}

	//Query devices
	ocl->devices = new cl_device_id[count];
	if (ocl->devices == NULL) quit("Memory allocation error");
	if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, ocl->devices, NULL) != CL_SUCCESS) quit("Error getting OpenCL devices"); 

	char device_vendor[64], device_name[64];
	cl_device_type device_type;
	cl_uint freq, shaders;

	if (fDebugLevel >= 2) HLTInfo("Available OPENCL devices:");
	for (unsigned int i = 0;i < count;i++)
	{
		if (fDebugLevel >= 3) {HLTDebug("Examining device %d\n", i);}
		cl_uint nbits;

		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, 64, device_name, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, NULL);
		//if (device_type & CL_DEVICE_TYPE_CPU) continue;
		//if (!(device_type & CL_DEVICE_TYPE_GPU)) continue;
		if (nbits / 8 != sizeof(void*)) continue;

		deviceSpeed = (long long int) freq * (long long int) shaders;
		if (device_type & CL_DEVICE_TYPE_GPU) deviceSpeed *= 10;
		if (fDebugLevel >= 2) {HLTDebug("Found Device %d: %s %s (Frequency %d, Shaders %d, %d bit) (Speed Value: %lld)\n", i, device_vendor, device_name, (int) freq, (int) shaders, (int) nbits, (long long int) deviceSpeed);}

		if (deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
	}
	if (bestDevice == (cl_uint) -1)
	{
		HLTWarning("No %sOPENCL Device available, aborting OPENCL Initialisation", count ? "appropriate " : "");
		return(1);
	}

	if (forceDeviceID > -1 && forceDeviceID < (signed) count) bestDevice = forceDeviceID;
	ocl->device = ocl->devices[bestDevice];

	clGetDeviceInfo(ocl->device, CL_DEVICE_NAME, 64, device_name, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_VENDOR, 64, device_vendor, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, NULL);
	if (fDebugLevel >= 2) {HLTDebug("Using OpenCL device %d: %s %s (Frequency %d, Shaders %d)\n", bestDevice, device_vendor, device_name, (int) freq, (int) shaders);}

	cl_uint compute_units;
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
	
	fConstructorBlockCount = compute_units * HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
	selectorBlockCount = compute_units * HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;

	ocl->context = clCreateContext(NULL, count, ocl->devices, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		HLTError("Could not create OPENCL Device Context!");
		return(1);
	}

	//Workaround to compile CL kernel during tracker initialization
	/*{
		char* file = "cagpubuild/AliHLTTPCCAGPUTrackerOpenCL.cl";
		HLTDebug("Reading source file %s\n", file);
		FILE* fp = fopen(file, "rb");
		if (fp == NULL)
		{
			HLTDebug("Cannot open %s\n", file);
			return(1);
		}
		fseek(fp, 0, SEEK_END);
		size_t file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		char* buffer = (char*) malloc(file_size + 1);
		if (buffer == NULL)
		{
			quit("Memory allocation error");
		}
		if (fread(buffer, 1, file_size, fp) != file_size)
		{
			quit("Error reading file");
		}
		buffer[file_size] = 0;
		fclose(fp);

		HLTDebug("Creating OpenCL Program Object\n");
		//Create OpenCL program object
		ocl->program = clCreateProgramWithSource(ocl->context, (cl_uint) 1, (const char**) &buffer, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating program object");

		HLTDebug("Compiling OpenCL Program\n");
		//Compile program
		ocl_error = clBuildProgram(ocl->program, count, ocl->devices, "-I. -Iinclude -Icode -Ibase -Imerger-ca -Icagpubuild -I/home/qon/AMD-APP-SDK-v2.8.1.0-RC-lnx64/include -I/usr/local/cuda/include -DHLTCA_STANDALONE -DBUILD_GPU -D_64BIT  -x clc++", NULL, NULL);
		if (ocl_error != CL_SUCCESS)
		{
			HLTDebug("OpenCL Error while building program: %d (Compiler options: %s)\n", ocl_error, "");

			for (unsigned int i = 0;i < count;i++)
			{
				cl_build_status status;
				clGetProgramBuildInfo(ocl->program, ocl->devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
				if (status == CL_BUILD_ERROR)
				{
					size_t log_size;
					clGetProgramBuildInfo(ocl->program, ocl->devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
					char* build_log = (char*) malloc(log_size + 1);
					if (build_log == NULL) quit("Memory allocation error");
					clGetProgramBuildInfo(ocl->program, ocl->devices[i], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
					HLTDebug("Build Log (device %d):\n\n%s\n\n", i, build_log);
					free(build_log);
				}
			}
		}
	}*/

	if (_makefiles_opencl_obtain_program_helper(ocl->context, count, ocl->devices, &ocl->program, _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl))
	{
		clReleaseContext(ocl->context);
		HLTError("Could not obtain OpenCL progarm");
		return(1);
	}
	if (fDebugLevel >= 2) HLTInfo("OpenCL program loaded successfully");

	ocl->kernel_row_blocks = clCreateKernel(ocl->program, "PreInitRowBlocks", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 1");return(1);}
	ocl->kernel_neighbours_finder = clCreateKernel(ocl->program, "AliHLTTPCCAProcess_AliHLTTPCCANeighboursFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 1");return(1);}
	ocl->kernel_neighbours_cleaner = clCreateKernel(ocl->program, "AliHLTTPCCAProcess_AliHLTTPCCANeighboursCleaner", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 2");return(1);}
	ocl->kernel_start_hits_finder = clCreateKernel(ocl->program, "AliHLTTPCCAProcess_AliHLTTPCCAStartHitsFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 3");return(1);}
	ocl->kernel_start_hits_sorter = clCreateKernel(ocl->program, "AliHLTTPCCAProcess_AliHLTTPCCAStartHitsSorter", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 4");return(1);}
	ocl->kernel_tracklet_selector = clCreateKernel(ocl->program, "AliHLTTPCCAProcessMulti_AliHLTTPCCATrackletSelector", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 5");return(1);}
	ocl->kernel_tracklet_constructor = clCreateKernel(ocl->program, "AliHLTTPCCATrackletConstructorGPU", &ocl_error); if (ocl_error != CL_SUCCESS) {HLTError("OPENCL Kernel Error 6");return(1);}
	if (fDebugLevel >= 2) HLTInfo("OpenCL kernels created successfully");

	ocl->mem_gpu = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, fGPUMemSize, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		HLTError("OPENCL Memory Allocation Error");
		clReleaseContext(ocl->context);
		return(1);
	}

	ocl->mem_constant = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, HLTCA_GPU_TRACKER_CONSTANT_MEM, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		HLTError("OPENCL Constant Memory Allocation Error");
		clReleaseMemObject(ocl->mem_gpu);
		clReleaseContext(ocl->context);
		return(1);
	}

	int nStreams = CAMath::Max(3, fSliceCount);
	if (nStreams > 36)
	{
		HLTError("Uhhh, more than 36 command queues requested, cannot do this. Did the TPC become larger?");
		return(1);
	}
	for (int i = 0;i < nStreams;i++)
	{
#ifdef CL_VERSION_2_0
		ocl->command_queue[i] = clCreateCommandQueueWithProperties(ocl->context, ocl->device, NULL, &ocl_error);
#else
		ocl->command_queue[i] = clCreateCommandQueue(ocl->context, ocl->device, 0, &ocl_error);
#endif
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
	}
	if (clEnqueueMigrateMemObjects(ocl->command_queue[0], 1, &ocl->mem_gpu, 0, 0, NULL, NULL) != CL_SUCCESS) quit("Error migrating buffer");

	if (fDebugLevel >= 1) HLTInfo("GPU Memory used: %d", (int) fGPUMemSize);
	int hostMemSize = HLTCA_GPU_ROWS_MEMORY + HLTCA_GPU_COMMON_MEMORY + sliceCount * (HLTCA_GPU_SLICE_DATA_MEMORY + HLTCA_GPU_TRACKS_MEMORY) + HLTCA_GPU_TRACKER_OBJECT_MEMORY;

	ocl->mem_host = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, hostMemSize, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error allocating pinned host memory");

	const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) *host_mem = (size_t) gpu_mem;}";
	cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char**) &krnlGetPtr, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error creating program object");
	ocl_error = clBuildProgram(program, 1, &ocl->device, "", NULL, NULL);
	if (ocl_error != CL_SUCCESS)
	{
		char build_log[16384];
		clGetProgramBuildInfo(program, ocl->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
		HLTImportant("Build Log:\n\n%s\n\n", build_log);
		quit("Error compiling program");
	}
	cl_kernel kernel = clCreateKernel(program, "krnlGetPtr", &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error creating kernel");
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &ocl->mem_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &ocl->mem_host);
	size_t local_size = 16, global_size = 16;
	if (clEnqueueNDRangeKernel(ocl->command_queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL) != CL_SUCCESS) quit("Error executing kernel");
	clFinish(ocl->command_queue[0]);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	if (fDebugLevel >= 2) HLTInfo("Mapping hostmemory");
	ocl->mem_host_ptr = clEnqueueMapBuffer(ocl->command_queue[0], ocl->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, hostMemSize, 0, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		HLTError("Error allocating Page Locked Host Memory");
		return(1);
	}
	fHostLockedMemory = ocl->mem_host_ptr;
	if (fDebugLevel >= 1) HLTInfo("Host Memory used: %d", hostMemSize);
	fGPUMergerHostMemory = ((char*) fHostLockedMemory) + hostMemSize - fGPUMergerMaxMemory;

	if (fDebugLevel >= 2) HLTInfo("Obtained Pointer to GPU Memory: %p", *((void**) ocl->mem_host_ptr));
	fGPUMemory = *((void**) ocl->mem_host_ptr);
	fGPUMergerMemory = ((char*) fGPUMemory) + fGPUMemSize - fGPUMergerMaxMemory;

	if (fDebugLevel >= 1)
	{
		memset(ocl->mem_host_ptr, 0, hostMemSize);
	}

	ocl->selector_events = new cl_event[fSliceCount];

	HLTInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d) Thread %d, Max slices: %d, %d bytes used)", bestDevice, device_vendor, device_name, (int) freq, (int) shaders, fThreadId, fSliceCount, fGPUMemSize);

	return(0);
}

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
 

bool AliHLTTPCCAGPUTrackerOpenCL::GPUFailedMsgA(int error, const char* file, int line)
{
	//Check for OPENCL Error and in the case of an error display the corresponding error string
	if (error == CL_SUCCESS) return(false);
	HLTWarning("OCL Error: %d / %s (%s:%d)", error, opencl_error_string(error), file, line);
	return(true);
}

int AliHLTTPCCAGPUTrackerOpenCL::GPUSync(const char* state, int stream, int slice)
{
	//Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards

	if (fDebugLevel == 0) return(0);
	for (int i = 0;i < fSliceCount;i++)
	{
		if (stream != -1) i = stream;
		clFinish(ocl->command_queue[i]);
		if (stream != -1) break;
	}
	if (fDebugLevel >= 3) HLTInfo("OPENCL Sync Done");
	return(0);
}

template <class T> static inline cl_int clSetKernelArgA(cl_kernel krnl, cl_uint num, T arg)
{
	return(clSetKernelArg(krnl, num, sizeof(T), &arg));
}

static inline cl_int clExecuteKernelA(cl_command_queue queue, cl_kernel krnl, size_t local_size, size_t global_size, cl_event* pEvent, cl_event* wait = NULL, cl_int nWaitEvents = 1)
{
	return(clEnqueueNDRangeKernel(queue, krnl, 1, NULL, &global_size, &local_size, wait == NULL ? 0 : nWaitEvents, wait, pEvent));
}

int AliHLTTPCCAGPUTrackerOpenCL::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function

	if (Reconstruct_Base_Init(pOutput, pClusterData, firstSlice, sliceCountLocal)) return(1);

	//Copy Tracker Object to GPU Memory
	if (fDebugLevel >= 3) HLTInfo("Copying Tracker objects to GPU");

	cl_event initEvent;
	GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[0], ocl->mem_constant, CL_FALSE, 0, sizeof(AliHLTTPCCATracker) * sliceCountLocal, fGpuTracker, 0, NULL, &initEvent));
	ocl->cl_queue_event_done[0] = true;
	for (int i = 1;i < 2;i++) //2 queues for first phase
	{
		ocl->cl_queue_event_done[i] = false;
	}

	if (GPUSync("Initialization (1)", 0, firstSlice) RANDOM_ERROR)
	{
		ResetHelperThreads(0);
		return(1);
	}
	
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(pClusterData, iSlice, firstSlice)) return(1);

		//Initialize temporary memory where needed
		if (fDebugLevel >= 3) HLTInfo("Copying Slice Data to GPU and initializing temporary memory");
		clSetKernelArgA(ocl->kernel_row_blocks, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_row_blocks, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_row_blocks, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[2], ocl->kernel_row_blocks, HLTCA_GPU_THREAD_COUNT, HLTCA_GPU_THREAD_COUNT * fConstructorBlockCount, NULL, &initEvent);
		if (GPUSync("Initialization (2)", 2, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].CommonMemory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), ocl->cl_queue_event_done[iSlice & 1] ? 0 : 1, ocl->cl_queue_event_done[iSlice & 1] ? NULL : &initEvent, NULL));
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].Data().Memory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), 0, NULL, NULL));
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].SliceDataRows() - (char*) fGPUMemory, (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), 0, NULL, NULL));
		ocl->cl_queue_event_done[iSlice & 1] = true;

		if (fDebugLevel >= 4)
		{
			if (fDebugLevel >= 5) HLTInfo("Allocating Debug Output Memory");
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100]), pClusterData[iSlice].NumberOfClusters() );
		}

		if (GPUSync("Initialization (3)", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		StandalonePerfTime(firstSlice + iSlice, 1);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_neighbours_finder, HLTCA_GPU_THREAD_COUNT_FINDER, HLTCA_GPU_THREAD_COUNT_FINDER * fSlaveTrackers[firstSlice + iSlice].Param().NRows(), NULL);

		if (GPUSync("Neighbours finder", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 2);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].Data().Memory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), 0, NULL, NULL));
			if (fDebugMask & 2) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, sliceCountLocal);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_neighbours_cleaner, HLTCA_GPU_THREAD_COUNT, HLTCA_GPU_THREAD_COUNT * (fSlaveTrackers[firstSlice + iSlice].Param().NRows() - 2), NULL);
		if (GPUSync("Neighbours Cleaner", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 3);

		if (fDebugLevel >= 4)
		{
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].Data().Memory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), 0, NULL, NULL));
			if (fDebugMask & 4) fSlaveTrackers[firstSlice + iSlice].DumpLinks(*fOutFile);
		}

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_start_hits_finder, HLTCA_GPU_THREAD_COUNT, HLTCA_GPU_THREAD_COUNT * (fSlaveTrackers[firstSlice + iSlice].Param().NRows() - 6), NULL);

		if (GPUSync("Start Hits Finder", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 4);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, sliceCountLocal);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_start_hits_sorter, HLTCA_GPU_THREAD_COUNT, HLTCA_GPU_THREAD_COUNT * fConstructorBlockCount, NULL);
		if (GPUSync("Start Hits Sorter", iSlice & 1, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 5);

		if (fDebugLevel >= 2)
		{
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].CommonMemory() - (char*) fGPUMemory, fGpuTracker[iSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), 0, NULL, NULL) RANDOM_ERROR);
			if (fDebugLevel >= 3) HLTInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *fSlaveTrackers[firstSlice + iSlice].NTracklets(), iSlice);
			if (*fSlaveTrackers[firstSlice + iSlice].NTracklets() > HLTCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				HLTError("HLTCA_GPU_MAX_TRACKLETS constant insuffisant");
				ResetHelperThreads(1);
				return(1);
			}
		}

		if (fDebugLevel >= 4 && *fSlaveTrackers[firstSlice + iSlice].NTracklets())
		{
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].TrackletTmpStartHits() - (char*) fGPUMemory, pClusterData[iSlice].NumberOfClusters() * sizeof(AliHLTTPCCAHitId), fSlaveTrackers[firstSlice + iSlice].TrackletStartHits(), 0, NULL, NULL));
			if (fDebugMask & 8)
			{
				*fOutFile << "Temporary ";
				fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
			}
			uint3* tmpMemory = (uint3*) malloc(sizeof(uint3) * fSlaveTrackers[firstSlice + iSlice].Param().NRows());
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].RowStartHitCountOffset() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].Param().NRows() * sizeof(uint3), tmpMemory, 0, NULL, NULL));
			if (fDebugMask & 16)
			{
				*fOutFile << "Start Hits Sort Vector:" << std::endl;
				for (int i = 1;i < fSlaveTrackers[firstSlice + iSlice].Param().NRows() - 5;i++)
				{
					*fOutFile << "Row: " << i << ", Len: " << tmpMemory[i].x << ", Offset: " << tmpMemory[i].y << ", New Offset: " << tmpMemory[i].z << std::endl;
				}
			}
			free(tmpMemory);
#endif

			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].HitMemory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].HitMemorySize(), fSlaveTrackers[firstSlice + iSlice].HitMemory(), 0, NULL, NULL));
			if (fDebugMask & 32) fSlaveTrackers[firstSlice + iSlice].DumpStartHits(*fOutFile);
		}

		StandalonePerfTime(firstSlice + iSlice, 6);

		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
	}
	clReleaseEvent(initEvent);

	for (int i = 0;i < fNHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	StandalonePerfTime(firstSlice, 7);

	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");

	cl_event initEvents2[3];
	for (int i = 0;i < 3;i++)
	{
		clEnqueueMarkerWithWaitList(ocl->command_queue[i], 0, NULL, &initEvents2[i]);
		//clFinish(ocl->command_queue[i]);
	}

	clSetKernelArgA(ocl->kernel_tracklet_constructor, 0, ocl->mem_gpu);
	clSetKernelArgA(ocl->kernel_tracklet_constructor, 1, ocl->mem_constant);
	clExecuteKernelA(ocl->command_queue[0], ocl->kernel_tracklet_constructor, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR * fConstructorBlockCount, NULL, initEvents2, 3);
	for (int i = 0;i < 3;i++)
	{
		clReleaseEvent(initEvents2[i]);
	}
	if (GPUSync("Tracklet Constructor", 0, firstSlice) RANDOM_ERROR)
	{
		SynchronizeGPU();
		return(1);
	}
	clFinish(ocl->command_queue[0]);

	StandalonePerfTime(firstSlice, 8);

	if (fDebugLevel >= 4)
	{
		for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
		{
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[0], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].CommonMemory() - (char*) fGPUMemory, fGpuTracker[iSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), 0, NULL, NULL));
			if (fDebugLevel >= 5)
			{
				HLTInfo("Obtained %d tracklets", *fSlaveTrackers[firstSlice + iSlice].NTracklets());
			}
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[0], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].TrackletMemory() - (char*) fGPUMemory, fGpuTracker[iSlice].TrackletMemorySize(), fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), 0, NULL, NULL));
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[0], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].HitMemory() - (char*) fGPUMemory, fGpuTracker[iSlice].HitMemorySize(), fSlaveTrackers[firstSlice + iSlice].HitMemory(), 0, NULL, NULL));
			if (fDebugMask & 128) fSlaveTrackers[firstSlice + iSlice].DumpTrackletHits(*fOutFile);
		}
	}

	int runSlices = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += runSlices)
	{
		if (runSlices < HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		if (fDebugLevel >= 3) HLTInfo("Running HLT Tracklet selector (Slice %d to %d)", iSlice, iSlice + runSlices);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 2, iSlice);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 3, (int) CAMath::Min(runSlices, sliceCountLocal - iSlice));
		clExecuteKernelA(ocl->command_queue[iSlice], ocl->kernel_tracklet_selector, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR * fConstructorBlockCount, NULL);
		if (GPUSync("Tracklet Selector", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			SynchronizeGPU();
			return(1);
		}
	}
	StandalonePerfTime(firstSlice, 9);
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		clEnqueueMarkerWithWaitList(ocl->command_queue[iSlice], 0, NULL, &ocl->selector_events[iSlice]);
	}

	char *tmpMemoryGlobalTracking = NULL;
	fSliceOutputReady = 0;
	
	if (Reconstruct_Base_StartGlobal(pOutput, tmpMemoryGlobalTracking)) return(1);

	int tmpSlice = 0, tmpSlice2 = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice++)
	{
		if (fDebugLevel >= 3) HLTInfo("Transfering Tracks from GPU to Host");
		cl_int eventdone;

		if (tmpSlice < sliceCountLocal) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		while(tmpSlice < sliceCountLocal && (tmpSlice == iSlice || eventdone == CL_COMPLETE))
		{
			clReleaseEvent(ocl->selector_events[tmpSlice]);
			if (GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice].CommonMemory() - (char*) fGPUMemory, fGpuTracker[tmpSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + tmpSlice].CommonMemory(), 0, NULL, &ocl->selector_events[tmpSlice]) RANDOM_ERROR))
			{
				HLTImportant("Error transferring tracks from GPU to host");
				ResetHelperThreads(1);
				ActivateThreadContext();
				return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
			}
			tmpSlice++;
			if (tmpSlice < sliceCountLocal) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		}

		if (tmpSlice2 < tmpSlice) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice2], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		while (tmpSlice2 < tmpSlice && (tmpSlice2 == iSlice ? (clFinish(ocl->command_queue[tmpSlice2]) == CL_SUCCESS) : (eventdone == CL_COMPLETE)))
		{
			if (*fSlaveTrackers[firstSlice + tmpSlice2].NTracks() > 0)
			{
	 			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice2], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice2].Tracks() - (char*) fGPUMemory, sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + tmpSlice2].NTracks(), fSlaveTrackers[firstSlice + tmpSlice2].Tracks(), 0, NULL, NULL));
				GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice2], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice2].TrackHits() - (char*) fGPUMemory, sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + tmpSlice2].NTrackHits(), fSlaveTrackers[firstSlice + tmpSlice2].TrackHits(), 0, NULL, NULL));
			}
			tmpSlice2++;
			if (tmpSlice2 < tmpSlice) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice2], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		}

		if (GPUFailedMsg(clFinish(ocl->command_queue[iSlice])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
		}

		if (fDebugLevel >= 4)
		{
			SynchronizeGPU();
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
			//GPUFailedMsg(cudaMemcpy(fSlaveTrackers[firstSlice + iSlice].Data().HitWeights(), fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign() * sizeof(int), cudaMemcpyDeviceToHost));
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[0], ocl->mem_gpu, CL_TRUE, (char*) fGpuTracker[iSlice].TrackletMemory() - (char*) fGPUMemory, fGpuTracker[iSlice].TrackletMemorySize(), fSlaveTrackers[firstSlice + iSlice].TrackletMemory(), 0, NULL, NULL));
			if (fDebugMask & 256) fSlaveTrackers[firstSlice + iSlice].DumpHitWeights(*fOutFile);
#endif
			if (fDebugMask & 512) fSlaveTrackers[firstSlice + iSlice].DumpTrackHits(*fOutFile);
		}


		if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			HLTError("GPU Tracker returned Error Code %d in slice %d (clusters %d)", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError, firstSlice + iSlice, fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHits());
			ResetHelperThreads(1);
			for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(pOutput, iSlice, firstSlice)) return(1);
	}
	for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);

	if (Reconstruct_Base_Finalize(pOutput, tmpMemoryGlobalTracking, firstSlice)) return(1);

	return(0);
}

int AliHLTTPCCAGPUTrackerOpenCL::ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	HLTFatal("Not implemented in OpenCL (ReconstructPP)");
	return(1);
}

int AliHLTTPCCAGPUTrackerOpenCL::ExitGPU_Runtime()
{
	//Uninitialize OPENCL

	const int nStreams = CAMath::Max(3, fSliceCount);
	for (int i = 0;i < nStreams;i++) clFinish(ocl->command_queue[i]);

	if (fGPUMemory)
	{
		clReleaseMemObject(ocl->mem_gpu);
		clReleaseMemObject(ocl->mem_constant);
		fGPUMemory = NULL;

		clReleaseKernel(ocl->kernel_neighbours_finder);
		clReleaseKernel(ocl->kernel_neighbours_cleaner);
		clReleaseKernel(ocl->kernel_start_hits_finder);
		clReleaseKernel(ocl->kernel_start_hits_sorter);
		clReleaseKernel(ocl->kernel_tracklet_constructor);
		clReleaseKernel(ocl->kernel_tracklet_selector);
		clReleaseKernel(ocl->kernel_row_blocks);
	}
	if (fHostLockedMemory)
	{
		clEnqueueUnmapMemObject(ocl->command_queue[0], ocl->mem_host, ocl->mem_host_ptr, 0, NULL, NULL);
		ocl->mem_host_ptr = NULL;
		for (int i = 0;i < nStreams;i++)
		{
			clReleaseCommandQueue(ocl->command_queue[i]);
		}
		clReleaseMemObject(ocl->mem_host);
		fGpuTracker = NULL;
		fHostLockedMemory = NULL;
	}

	if (ocl->selector_events)
	{
		delete[] ocl->selector_events;
		ocl->selector_events = NULL;
	}
	if (ocl->devices)
	{
		delete[] ocl->devices;
		ocl->devices = NULL;
	}

	clReleaseProgram(ocl->program);
	clReleaseContext(ocl->context);

	HLTInfo("OPENCL Uninitialized");
	fCudaInitialized = 0;
	return(0);
}

int AliHLTTPCCAGPUTrackerOpenCL::RefitMergedTracks(AliHLTTPCGMMerger* Merger)
{
	HLTFatal("Not implemented in OpenCL (Merger)");
	return(1);
}

void AliHLTTPCCAGPUTrackerOpenCL::ActivateThreadContext()
{
}

void AliHLTTPCCAGPUTrackerOpenCL::ReleaseThreadContext()
{
}

void AliHLTTPCCAGPUTrackerOpenCL::SynchronizeGPU()
{
	const int nStreams = CAMath::Max(3, fSliceCount);
	for (int i = 0;i < nStreams;i++) clFinish(ocl->command_queue[i]);
}

AliHLTTPCCAGPUTracker* AliHLTTPCCAGPUTrackerNVCCCreate()
{
	return new AliHLTTPCCAGPUTrackerOpenCL;
}

void AliHLTTPCCAGPUTrackerNVCCDestroy(AliHLTTPCCAGPUTracker* ptr)
{
	delete ptr;
}
