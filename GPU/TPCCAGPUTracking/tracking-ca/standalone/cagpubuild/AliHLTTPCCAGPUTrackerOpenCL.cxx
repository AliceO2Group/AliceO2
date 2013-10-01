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
#define HLTCA_HOSTCODE

#include <string.h>
#include "AliHLTTPCCAGPUTrackerOpenCL.h"
#include "AliHLTTPCCAGPUTrackerOpenCLInternals.h"

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
	HLTInfo("%d OpenCL Platforms found", num_platforms);
	
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
		printf("Available Platform %d: (%s %s) %s %s\n", i_platform, platform_profile, platform_version, platform_vendor, platform_name);
		if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0)
		{
			found = true;
			HLTInfo("AMD OpenCL Platform found");
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
		if (fDebugLevel >= 4) printf("Examining device %d\n", i);
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
		printf("Found Device %d: %s %s (Frequency %d, Shaders %d, %d bit)\n", i, device_vendor, device_name, (int) freq, (int) shaders, (int) nbits);

		deviceSpeed = (long long int) freq * (long long int) shaders;
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

	if (_makefiles_opencl_obtain_program_helper(ocl->context, count, ocl->devices, &ocl->program, _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl))
	{
		clReleaseContext(ocl->context);
		HLTError("Could not obtain OpenCL progarm");
		return(1);
	}
	HLTInfo("OpenCL program loaded successfully");

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
		ocl->command_queue[i] = clCreateCommandQueue(ocl->context, ocl->device, 0, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
	}
	if (clEnqueueMigrateMemObjects(ocl->command_queue[0], 1, &ocl->mem_gpu, 0, 0, NULL, NULL) != CL_SUCCESS) quit("Error migrating buffer");

	fGPUMergerMemory = ((char*) fGPUMemory) + fGPUMemSize - fGPUMergerMaxMemory;
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

	HLTInfo("Mapping hostmemory");
	ocl->mem_host_ptr = clEnqueueMapBuffer(ocl->command_queue[0], ocl->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, hostMemSize, 0, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		HLTError("Error allocating Page Locked Host Memory");
		return(1);
	}
	if (fDebugLevel >= 1) HLTInfo("Host Memory used: %d", hostMemSize);
	fGPUMergerHostMemory = ((char*) fHostLockedMemory) + hostMemSize - fGPUMergerMaxMemory;

	HLTInfo("Obtained Pointer to GPU Memory: %p", *((void**) ocl->mem_host_ptr));
	fGPUMemory = *((void**) ocl->mem_host_ptr);

	if (fDebugLevel >= 1)
	{
		memset(ocl->mem_host_ptr, 0, hostMemSize);
	}

	ocl->selector_events = new cl_event[fSliceCount];

	HLTImportant("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d) Thread %d, Max slices: %d)", bestDevice, device_vendor, device_name, (int) freq, (int) shaders, fThreadId, fSliceCount);

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

int AliHLTTPCCAGPUTrackerOpenCL::GPUSync(char* state, int sliceLocal, int slice)
{
	//Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards

	if (fDebugLevel == 0) return(0);
	for (int i = 0;i < fSliceCount;i++)
	{
		clFinish(ocl->command_queue[i]);
	}
	if (fDebugLevel >= 3) HLTInfo("OPENCL Sync Done");
	return(0);
}

int AliHLTTPCCAGPUTrackerOpenCL::Reconstruct(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	//Primary reconstruction function

	if (Reconstruct_Base_Init(pOutput, pClusterData, firstSlice, sliceCountLocal)) return(1);

	//Copy Tracker Object to GPU Memory
	if (fDebugLevel >= 3) HLTInfo("Copying Tracker objects to GPU");

	GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[0], ocl->mem_constant, CL_FALSE, 0, sizeof(AliHLTTPCCATracker) * sliceCountLocal, fGpuTracker, 0, NULL, NULL));

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
		//PreInitRowBlocks<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[2]>>>(fGpuTracker[iSlice].RowBlockPos(), fGpuTracker[iSlice].RowBlockTracklets(), fGpuTracker[iSlice].Data().HitWeights(), fSlaveTrackers[firstSlice + iSlice].Data().NumberOfHitsPlusAlign());
		if (GPUSync("Initialization (2)", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].CommonMemory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + iSlice].CommonMemory(), 0, NULL, NULL));
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].Data().Memory() - (char*) fGPUMemory, fSlaveTrackers[firstSlice + iSlice].Data().GpuMemorySize(), fSlaveTrackers[firstSlice + iSlice].Data().Memory(), 0, NULL, NULL));
		GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[iSlice & 1], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[iSlice].SliceDataRows() - (char*) fGPUMemory, (HLTCA_ROW_COUNT + 1) * sizeof(AliHLTTPCCARow), fSlaveTrackers[firstSlice + iSlice].SliceDataRows(), 0, NULL, NULL));

		if (fDebugLevel >= 4)
		{
			if (fDebugLevel >= 5) HLTInfo("Allocating Debug Output Memory");
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTrackletsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].TrackletMemorySize()/sizeof( uint4 ) + 100] ), HLTCA_GPU_MAX_TRACKLETS, fConstructorBlockCount);
			fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerHitsMemory(reinterpret_cast<char*> ( new uint4 [ fGpuTracker[iSlice].HitMemorySize()/sizeof( uint4 ) + 100]), pClusterData[iSlice].NumberOfClusters() );
		}

		if (GPUSync("Initialization (3)", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		StandalonePerfTime(firstSlice + iSlice, 1);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		//AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows(), HLTCA_GPU_THREAD_COUNT_FINDER, 0, cudaStreams[iSlice & 1]>>>(iSlice);

		if (GPUSync("Neighbours finder", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 2);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, sliceCountLocal);
		//AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-2, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Neighbours Cleaner", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 3);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, sliceCountLocal);
		//AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder> <<<fSlaveTrackers[firstSlice + iSlice].Param().NRows()-6, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Start Hits Finder", iSlice, iSlice + firstSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		StandalonePerfTime(firstSlice + iSlice, 4);

		if (fDebugLevel >= 3) HLTInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, sliceCountLocal);
		//AliHLTTPCCAProcess<AliHLTTPCCAStartHitsSorter> <<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT, 0, cudaStreams[iSlice & 1]>>>(iSlice);
		if (GPUSync("Start Hits Sorter", iSlice, iSlice + firstSlice) RANDOM_ERROR)
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

		StandalonePerfTime(firstSlice + iSlice, 6);

		fSlaveTrackers[firstSlice + iSlice].SetGPUTrackerTracksMemory((char*) TracksMemory(fHostLockedMemory, iSlice), HLTCA_GPU_MAX_TRACKS, pClusterData[iSlice].NumberOfClusters());
	}

	for (int i = 0;i < fNHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	StandalonePerfTime(firstSlice, 7);

	if (fDebugLevel >= 3) HLTInfo("Running GPU Tracklet Constructor");
	//AliHLTTPCCATrackletConstructorGPU<<<fConstructorBlockCount, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR>>>();
	if (GPUSync("Tracklet Constructor", 0, firstSlice) RANDOM_ERROR)
	{
		SynchronizeGPU();
		return(1);
	}

	StandalonePerfTime(firstSlice, 8);

	int runSlices = 0;
	for (int iSlice = 0;iSlice < sliceCountLocal;iSlice += runSlices)
	{
		if (runSlices < HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		if (fDebugLevel >= 3) HLTInfo("Running HLT Tracklet selector (Slice %d to %d)", iSlice, iSlice + runSlices);
		//AliHLTTPCCAProcessMulti<AliHLTTPCCATrackletSelector><<<selectorBlockCount, HLTCA_GPU_THREAD_COUNT_SELECTOR, 0, cudaStreams[iSlice]>>>(iSlice, CAMath::Min(runSlices, sliceCountLocal - iSlice));
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

		GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		while(tmpSlice < sliceCountLocal && (tmpSlice == iSlice || eventdone == CL_COMPLETE))
		{
			clReleaseEvent(ocl->selector_events[tmpSlice]);
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice].CommonMemory() - (char*) fGPUMemory, fGpuTracker[tmpSlice].CommonMemorySize(), fSlaveTrackers[firstSlice + tmpSlice].CommonMemory(), 0, NULL, &ocl->selector_events[tmpSlice]) RANDOM_ERROR);
			{
				ResetHelperThreads(1);
				ActivateThreadContext();
				return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
			}
			tmpSlice++;
		}

		GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		while (tmpSlice2 < tmpSlice && (tmpSlice2 == iSlice ? (clFinish(ocl->command_queue[tmpSlice2]) == CL_SUCCESS) : (eventdone == CL_COMPLETE)))
		{
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice2], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice2].Tracks() - (char*) fGPUMemory, sizeof(AliHLTTPCCATrack) * *fSlaveTrackers[firstSlice + tmpSlice2].NTracks(), fSlaveTrackers[firstSlice + tmpSlice2].Tracks(), 0, NULL, NULL));
			GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[tmpSlice2], ocl->mem_gpu, CL_FALSE, (char*) fGpuTracker[tmpSlice2].TrackHits() - (char*) fGPUMemory, sizeof(AliHLTTPCCAHitId) * *fSlaveTrackers[firstSlice + tmpSlice2].NTrackHits(), fSlaveTrackers[firstSlice + tmpSlice2].TrackHits(), 0, NULL, NULL));
			tmpSlice2++;
		}

		if (GPUFailedMsg(clFinish(ocl->command_queue[iSlice])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(SelfHealReconstruct(pOutput, pClusterData, firstSlice, sliceCountLocal));
		}

		if (fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			HLTError("GPU Tracker returned Error Code %d in slice %d", fSlaveTrackers[firstSlice + iSlice].GPUParameters()->fGPUError, firstSlice + iSlice);
			ResetHelperThreads(1);
			for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(1);
		}
		if (fDebugLevel >= 3) HLTInfo("Tracks Transfered: %d / %d", *fSlaveTrackers[firstSlice + iSlice].NTracks(), *fSlaveTrackers[firstSlice + iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(pOutput, iSlice, firstSlice)) return(1);
	}
	for (int iSlice2 = 0;iSlice2 < sliceCountLocal;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);

	if (Reconstruct_Base_Finalize(pOutput, tmpMemoryGlobalTracking, firstSlice)) return(1);

	return(1);
}

int AliHLTTPCCAGPUTrackerOpenCL::ReconstructPP(AliHLTTPCCASliceOutput** pOutput, AliHLTTPCCAClusterData* pClusterData, int firstSlice, int sliceCountLocal)
{
	HLTFatal("Not implemented in OpenCL");
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
	HLTFatal("Not implemented in OpenCL");
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

