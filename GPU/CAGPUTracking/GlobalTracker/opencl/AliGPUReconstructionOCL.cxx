#define GPUCA_GPUTYPE_RADEON

#include "AliGPUReconstructionOCL.h"
#include "AliGPUReconstructionOCL.h"

#ifdef HAVE_O2HEADERS
#include "ITStracking/TrackerTraitsCPU.h"
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsCPU : public TrackerTraits {}; }}
#endif

#include "AliGPUCADataTypes.h"
#include "AliCAGPULogging.h"

#include <string.h>
#include "AliGPUReconstructionOCLInternals.h"
#include "AliGPUTPCGPUTrackerCommon.h"

#include "AliGPUTPCTrackParam.h"
#include "AliGPUTPCTrack.h"

#include "AliGPUTPCHitArea.h"
#include "AliGPUTPCGrid.h"
#include "AliGPUTPCRow.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCTracker.h"

#include "AliGPUTPCProcess.h"

#include "AliGPUTPCTrackletSelector.h"
#include "AliGPUTPCNeighboursFinder.h"
#include "AliGPUTPCNeighboursCleaner.h"
#include "AliGPUTPCStartHitsFinder.h"
#include "AliGPUTPCStartHitsSorter.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCClusterData.h"

#include "AliGPUCADataTypes.h"

#include <unistd.h>

#include "../makefiles/opencl_obtain_program.h"
extern "C" char _makefile_opencl_program_GlobalTracker_opencl_AliGPUReconstructionOCL_cl[];

#define quit(msg) {CAGPUError(msg);return(1);}

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

AliGPUReconstructionOCL::AliGPUReconstructionOCL(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mProcessingSettings.deviceType = OCL;
	mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsCPU);

	ocl = new AliGPUReconstructionOCLInternals;
	if (ocl == NULL)
	{
		CAGPUError("Memory Allocation Error");
	}
	ocl->mem_host_ptr = NULL;
	ocl->selector_events = NULL;
	ocl->devices = NULL;
}

AliGPUReconstructionOCL::~AliGPUReconstructionOCL()
{
	delete ocl;
}

AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg)
{
	return new AliGPUReconstructionOCL(cfg);
}

int AliGPUReconstructionOCL::InitDevice_Runtime()
{
	//Find best OPENCL device, initialize and allocate memory

	cl_int ocl_error;
	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) quit("Error getting OpenCL Platform Count");
	if (num_platforms == 0) quit("No OpenCL Platform found");
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("%d OpenCL Platforms found", num_platforms);
	
	//Query platforms
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == NULL) quit("Memory allocation error");
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) quit("Error getting OpenCL Platforms");

	cl_platform_id platform;
	bool found = false;
	if (mDeviceProcessingSettings.platformNum >= 0)
	{
		if (mDeviceProcessingSettings.platformNum >= (int) num_platforms)
		{
			CAGPUError("Invalid platform specified");
			return(1);
		}
		platform = platforms[mDeviceProcessingSettings.platformNum];
		found = true;
	}
	else
	{
		for (unsigned int i_platform = 0;i_platform < num_platforms;i_platform++)
		{
			char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_PROFILE, 64, platform_profile, NULL);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VERSION, 64, platform_version, NULL);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_NAME, 64, platform_name, NULL);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VENDOR, 64, platform_vendor, NULL);
			if (mDeviceProcessingSettings.debugLevel >= 2) {CAGPUDebug("Available Platform %d: (%s %s) %s %s\n", i_platform, platform_profile, platform_version, platform_vendor, platform_name);}
			if (strcmp(platform_vendor, "Advanced Micro Devices, Inc.") == 0)
			{
				found = true;
				if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("AMD OpenCL Platform found");
				platform = platforms[i_platform];
				break;
			}
		}
	}
	delete[] platforms;
	if (found == false)
	{
		CAGPUError("Did not find AMD OpenCL Platform");
		return(1);
	}

	cl_uint count, bestDevice = (cl_uint) -1;
	double bestDeviceSpeed = 0, deviceSpeed;
	if (GPUFailedMsg(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &count)))
	{
		CAGPUError("Error getting OPENCL Device Count");
		return(1);
	}

	//Query devices
	ocl->devices = new cl_device_id[count];
	if (ocl->devices == NULL) quit("Memory allocation error");
	if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, ocl->devices, NULL) != CL_SUCCESS) quit("Error getting OpenCL devices");

	char device_vendor[64], device_name[64];
	cl_device_type device_type;
	cl_uint freq, shaders;

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Available OPENCL devices:");
	for (unsigned int i = 0;i < count;i++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUInfo("Examining device %d\n", i);}
		cl_uint nbits;

		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, 64, device_name, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_VENDOR, 64, device_vendor, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, NULL);
		clGetDeviceInfo(ocl->devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, NULL);
		if (mDeviceProcessingSettings.gpuDeviceOnly && ((device_type & CL_DEVICE_TYPE_CPU) || !(device_type & CL_DEVICE_TYPE_GPU))) continue;
		if (nbits / 8 != sizeof(void*)) continue;

		deviceSpeed = (double) freq * (double) shaders;
		if (device_type & CL_DEVICE_TYPE_GPU) deviceSpeed *= 10;
		if (mDeviceProcessingSettings.debugLevel >= 2) {CAGPUInfo("Found Device %d: %s %s (Frequency %d, Shaders %d, %d bit) (Speed Value: %lld)\n", i, device_vendor, device_name, (int) freq, (int) shaders, (int) nbits, (long long int) deviceSpeed);}

		if (deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
	}
	if (bestDevice == (cl_uint) -1)
	{
		CAGPUWarning("No %sOPENCL Device available, aborting OPENCL Initialisation", count ? "appropriate " : "");
		return(1);
	}

	if (mDeviceProcessingSettings.deviceNum > -1)
	{
		if (mDeviceProcessingSettings.deviceNum < (signed) count)
		{
			bestDevice = mDeviceProcessingSettings.deviceNum;
		}
		else
		{
			CAGPUWarning("Requested device ID %d non existend, falling back to default device id %d", mDeviceProcessingSettings.deviceNum, bestDevice);
		}
	}
	ocl->device = ocl->devices[bestDevice];

	clGetDeviceInfo(ocl->device, CL_DEVICE_NAME, 64, device_name, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_VENDOR, 64, device_vendor, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, NULL);
	if (mDeviceProcessingSettings.debugLevel >= 2) {CAGPUDebug("Using OpenCL device %d: %s %s (Frequency %d, Shaders %d)\n", bestDevice, device_vendor, device_name, (int) freq, (int) shaders);}

	cl_uint compute_units;
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
	
	fConstructorBlockCount = compute_units * GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
	fSelectorBlockCount = compute_units * GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;
	fConstructorThreadCount = GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;

	ocl->context = clCreateContext(NULL, count, ocl->devices, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("Could not create OPENCL Device Context!");
		return(1);
	}

	//Workaround to compile CL kernel during tracker initialization
	/*{
		char* file = "GlobalTracker/opencl/AliGPUReconstructionOCL.cl";
		CAGPUDebug("Reading source file %s\n", file);
		FILE* fp = fopen(file, "rb");
		if (fp == NULL)
		{
			CAGPUDebug("Cannot open %s\n", file);
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

		CAGPUDebug("Creating OpenCL Program Object\n");
		//Create OpenCL program object
		ocl->program = clCreateProgramWithSource(ocl->context, (cl_uint) 1, (const char**) &buffer, NULL, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating program object");

		CAGPUDebug("Compiling OpenCL Program\n");
		//Compile program
		ocl_error = clBuildProgram(ocl->program, count, ocl->devices, "-I. -Iinclude -ISliceTracker -IHLTHeaders -IMerger -IGlobalTracker -I/home/qon/AMD-APP-SDK-v2.8.1.0-RC-lnx64/include -DGPUCA_STANDALONE -DBUILD_GPU -D_64BIT -x clc++", NULL, NULL);
		if (ocl_error != CL_SUCCESS)
		{
			CAGPUDebug("OpenCL Error while building program: %d (Compiler options: %s)\n", ocl_error, "");

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
					CAGPUDebug("Build Log (device %d):\n\n%s\n\n", i, build_log);
					free(build_log);
				}
			}
		}
	}*/

	if (_makefiles_opencl_obtain_program_helper(ocl->context, count, ocl->devices, &ocl->program, _makefile_opencl_program_GlobalTracker_opencl_AliGPUReconstructionOCL_cl))
	{
		clReleaseContext(ocl->context);
		CAGPUError("Could not obtain OpenCL progarm");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("OpenCL program loaded successfully");

	ocl->kernel_row_blocks = clCreateKernel(ocl->program, "PreInitRowBlocks", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 1");return(1);}
	ocl->kernel_neighbours_finder = clCreateKernel(ocl->program, "AliGPUTPCProcess_AliGPUTPCNeighboursFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 1");return(1);}
	ocl->kernel_neighbours_cleaner = clCreateKernel(ocl->program, "AliGPUTPCProcess_AliGPUTPCNeighboursCleaner", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 2");return(1);}
	ocl->kernel_start_hits_finder = clCreateKernel(ocl->program, "AliGPUTPCProcess_AliGPUTPCStartHitsFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 3");return(1);}
	ocl->kernel_start_hits_sorter = clCreateKernel(ocl->program, "AliGPUTPCProcess_AliGPUTPCStartHitsSorter", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 4");return(1);}
	ocl->kernel_tracklet_selector = clCreateKernel(ocl->program, "AliGPUTPCProcessMulti_AliGPUTPCTrackletSelector", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 5");return(1);}
	ocl->kernel_tracklet_constructor = clCreateKernel(ocl->program, "AliGPUTPCTrackletConstructorGPU", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 6");return(1);}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("OpenCL kernels created successfully");

	ocl->mem_gpu = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, fGPUMemSize, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("OPENCL Memory Allocation Error");
		clReleaseContext(ocl->context);
		return(1);
	}

	ocl->mem_constant = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, sizeof(AliGPUCAConstantMem), NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("OPENCL Constant Memory Allocation Error");
		clReleaseMemObject(ocl->mem_gpu);
		clReleaseContext(ocl->context);
		return(1);
	}

	unsigned int nStreams = 3;
	if (nStreams > NSLICES)
	{
		CAGPUError("Uhhh, more than 36 command queues requested, cannot do this. Did the TPC become larger?");
		return(1);
	}
	for (unsigned int i = 0;i < nStreams;i++)
	{
#ifdef CL_VERSION_2_0
		ocl->command_queue[i] = clCreateCommandQueueWithProperties(ocl->context, ocl->device, NULL, &ocl_error);
#else
		ocl->command_queue[i] = clCreateCommandQueue(ocl->context, ocl->device, 0, &ocl_error);
#endif
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
	}
	if (clEnqueueMigrateMemObjects(ocl->command_queue[0], 1, &ocl->mem_gpu, 0, 0, NULL, NULL) != CL_SUCCESS) quit("Error migrating buffer");

	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("GPU Memory used: %lld", fGPUMemSize);

	ocl->mem_host = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, fHostMemSize, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error allocating pinned host memory");

	const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) *host_mem = (size_t) gpu_mem;}";
	cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char**) &krnlGetPtr, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error creating program object");
	ocl_error = clBuildProgram(program, 1, &ocl->device, "", NULL, NULL);
	if (ocl_error != CL_SUCCESS)
	{
		char build_log[16384];
		clGetProgramBuildInfo(program, ocl->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL);
		CAGPUImportant("Build Log:\n\n%s\n\n", build_log);
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

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Mapping hostmemory");
	ocl->mem_host_ptr = clEnqueueMapBuffer(ocl->command_queue[0], ocl->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, fHostMemSize, 0, NULL, NULL, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("Error allocating Page Locked Host Memory");
		return(1);
	}
	fHostLockedMemory = ocl->mem_host_ptr;
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("Host Memory used: %lld", fHostMemSize);

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Obtained Pointer to GPU Memory: %p", *((void**) ocl->mem_host_ptr));
	fGPUMemory = *((void**) ocl->mem_host_ptr);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(ocl->mem_host_ptr, 0, fHostMemSize);
	}

	ocl->selector_events = new cl_event[NSLICES];
	
	mDeviceParam = &mParam;

	CAGPUInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d) Thread %d, %lld bytes used)", bestDevice, device_vendor, device_name, (int) freq, (int) shaders, fThreadId, (long long int) fGPUMemSize);

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
 

bool AliGPUReconstructionOCL::GPUFailedMsgA(int error, const char* file, int line)
{
	//Check for OPENCL Error and in the case of an error display the corresponding error string
	if (error == CL_SUCCESS) return(false);
	CAGPUWarning("OCL Error: %d / %s (%s:%d)", error, opencl_error_string(error), file, line);
	return(true);
}

int AliGPUReconstructionOCL::GPUSync(const char* state, int stream, int slice)
{
	//Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards

	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		if (stream != -1) i = stream;
		clFinish(ocl->command_queue[i]);
		if (stream != -1) break;
	}
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Sync Done");
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

int AliGPUReconstructionOCL::RunTPCTrackingSlices()
{
	//Primary reconstruction function
	if (fGPUStuck)
	{
		CAGPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	if (Reconstruct_Base_Init()) return(1);

	//Copy Tracker Object to GPU Memory
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Tracker objects to GPU");

	cl_event initEvent;
	if (GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[0], ocl->mem_constant, CL_FALSE, 0, sizeof(AliGPUTPCTracker) * NSLICES, fGpuTracker, 0, NULL, &initEvent)))
	{
		CAGPUError("Error filling constant memory");
		ResetHelperThreads(0);
		return 1;
	}
	ocl->cl_queue_event_done[0] = true;
	for (int i = 1;i < 2;i++) //2 queues for first phase
	{
		ocl->cl_queue_event_done[i] = false;
	}

	if (GPUSync("Initialization (1)", 0, 0) RANDOM_ERROR)
	{
		ResetHelperThreads(0);
		return(1);
	}
	
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(iSlice)) return(1);

		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		clSetKernelArgA(ocl->kernel_row_blocks, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_row_blocks, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_row_blocks, 2, iSlice);
		clExecuteKernelA(ocl->command_queue[2], ocl->kernel_row_blocks, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * fConstructorBlockCount, NULL, &initEvent);
		if (GPUSync("Initialization (2)", 2, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}

		//Copy Data to GPU Global Memory
		mTPCSliceTrackersCPU[iSlice].StartTimer(0);
		if (TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResInput(), iSlice & 1, ocl->cl_queue_event_done[iSlice & 1] ? 0 : 1, ocl->cl_queue_event_done[iSlice & 1] ? NULL : &initEvent, NULL) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResRows(), iSlice & 1) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), iSlice & 1))
		{
			CAGPUError("Error copying data to GPU");
			ResetHelperThreads(0);
			return 1;
		}
		ocl->cl_queue_event_done[iSlice & 1] = true;

		if (GPUSync("Initialization (3)", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(0);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, NSLICES);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_neighbours_finder, 2, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(1);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_neighbours_finder, GPUCA_GPU_THREAD_COUNT_FINDER, GPUCA_GPU_THREAD_COUNT_FINDER * GPUCA_ROW_COUNT, NULL);
		if (GPUSync("Neighbours finder", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(1);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 2) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, NSLICES);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_neighbours_cleaner, 2, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(2);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_neighbours_cleaner, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * (GPUCA_ROW_COUNT - 2), NULL);
		if (GPUSync("Neighbours Cleaner", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(2);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, NSLICES);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_start_hits_finder, 2, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(3);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_start_hits_finder, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * (GPUCA_ROW_COUNT - 6), NULL);
		if (GPUSync("Start Hits Finder", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(3);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, NSLICES);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_start_hits_sorter, 2, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(4);
		clExecuteKernelA(ocl->command_queue[iSlice & 1], ocl->kernel_start_hits_sorter, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * fConstructorBlockCount, NULL);
		if (GPUSync("Start Hits Sorter", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(4);

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *mTPCSliceTrackersCPU[iSlice].NTracklets(), iSlice);
			if (*mTPCSliceTrackersCPU[iSlice].NTracklets() > GPUCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				CAGPUError("GPUCA_GPU_MAX_TRACKLETS constant insuffisant");
				ResetHelperThreads(1);
				return(1);
			}
		}

		if (mDeviceProcessingSettings.debugLevel >= 4 && *mTPCSliceTrackersCPU[iSlice].NTracklets())
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 32) mTPCSliceTrackersCPU[iSlice].DumpStartHits(mDebugFile);
		}
	}
	clReleaseEvent(initEvent);

	for (int i = 0;i < mDeviceProcessingSettings.nDeviceHelperThreads;i++)
	{
		pthread_mutex_lock(&((pthread_mutex_t*) fHelperParams[i].fMutex)[1]);
	}

	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Tracklet Constructor");

	cl_event initEvents2[3];
	for (int i = 0;i < 3;i++)
	{
		clEnqueueMarkerWithWaitList(ocl->command_queue[i], 0, NULL, &initEvents2[i]);
		//clFinish(ocl->command_queue[i]);
	}

	cl_event constructorEvent;
	clSetKernelArgA(ocl->kernel_tracklet_constructor, 0, ocl->mem_gpu);
	clSetKernelArgA(ocl->kernel_tracklet_constructor, 1, ocl->mem_constant);
	mTPCSliceTrackersCPU[0].StartTimer(6);
	clExecuteKernelA(ocl->command_queue[0], ocl->kernel_tracklet_constructor, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR * fConstructorBlockCount, &constructorEvent, initEvents2, 3);
	for (int i = 0;i < 3;i++)
	{
		clReleaseEvent(initEvents2[i]);
	}
	if (GPUSync("Tracklet Constructor", 0, 0) RANDOM_ERROR)
	{
		SynchronizeGPU();
		return(1);
	}
	mTPCSliceTrackersCPU[0].StopTimer(6);
	if (mDeviceProcessingSettings.stuckProtection)
	{
		cl_int tmp;
		for (int i = 0;i <= mDeviceProcessingSettings.stuckProtection / 50;i++)
		{
			usleep(50);
			clGetEventInfo(constructorEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(tmp), &tmp, NULL);
			if (tmp == CL_COMPLETE) break;
		}
		if (tmp != CL_COMPLETE)
		{
			CAGPUError("GPU Stuck, future processing in this component is disabled, skipping event (GPU Event State %d)", (int) tmp);
			fGPUStuck = 1;
			return(1);
		}
	}
	else
	{
		clFinish(ocl->command_queue[0]);
	}
	clReleaseEvent(constructorEvent);

	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			CAGPUInfo("Obtained %d tracklets", *mTPCSliceTrackersCPU[iSlice].NTracklets());
			if (mDeviceProcessingSettings.debugMask & 128) mTPCSliceTrackersCPU[iSlice].DumpTrackletHits(mDebugFile);
		}
	}

	int runSlices = 0;
	int useStream = 0;
	int streamMap[NSLICES];
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice += runSlices)
	{
		if (runSlices < GPUCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
		runSlices = CAMath::Min(runSlices, NSLICES - iSlice);
		if (fSelectorBlockCount < runSlices) runSlices = fSelectorBlockCount;
		if (GPUCA_GPU_NUM_STREAMS && useStream + 1 == GPUCA_GPU_NUM_STREAMS) runSlices = NSLICES - iSlice;
		if (fSelectorBlockCount < runSlices)
		{
			CAGPUError("Insufficient number of blocks for tracklet selector");
			SynchronizeGPU();
			return(1);
		}
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 0, ocl->mem_gpu);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 1, ocl->mem_constant);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 2, iSlice);
		clSetKernelArgA(ocl->kernel_tracklet_selector, 3, runSlices);
		mTPCSliceTrackersCPU[iSlice].StartTimer(7);
		clExecuteKernelA(ocl->command_queue[useStream], ocl->kernel_tracklet_selector, GPUCA_GPU_THREAD_COUNT_SELECTOR, GPUCA_GPU_THREAD_COUNT_SELECTOR * fSelectorBlockCount, NULL);
		if (GPUSync("Tracklet Selector", iSlice, iSlice) RANDOM_ERROR)
		{
			SynchronizeGPU();
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(7);
		for (unsigned int k = iSlice;k < iSlice + runSlices;k++)
		{
			if (TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[k].MemoryResCommon(), useStream, 0, NULL, &ocl->selector_events[k]))
			{
				CAGPUImportant("Error transferring number of tracks from GPU to host");
				ResetHelperThreads(1);
				ActivateThreadContext();
				return(1);
			}
			streamMap[k] = useStream;
		}
		useStream++;
		if (useStream >= 3) useStream = 0;
	}

	fSliceOutputReady = 0;
	
	if (Reconstruct_Base_StartGlobal()) return(1);
	
	unsigned int tmpSlice = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Transfering Tracks from GPU to Host");
		cl_int eventdone;

		if (tmpSlice < NSLICES) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		while (tmpSlice < NSLICES && (tmpSlice == iSlice ? (clFinish(ocl->command_queue[streamMap[tmpSlice]]) == CL_SUCCESS) : (eventdone == CL_COMPLETE)))
		{
			if (*mTPCSliceTrackersCPU[tmpSlice].NTracks() > 0)
			{
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTracks(), streamMap[tmpSlice]);
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTrackHits(), streamMap[tmpSlice]);
			}
			tmpSlice++;
			if (tmpSlice < NSLICES) GPUFailedMsg(clGetEventInfo(ocl->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, NULL));
		}

		if (GPUFailedMsg(clFinish(ocl->command_queue[streamMap[tmpSlice]])) RANDOM_ERROR)
		{
			ResetHelperThreads(1);
			ActivateThreadContext();
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(1);
		}

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			SynchronizeGPU();
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) mTPCSliceTrackersCPU[iSlice].DumpHitWeights(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 512) mTPCSliceTrackersCPU[iSlice].DumpTrackHits(mDebugFile);
		}

		if (mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError];
			CAGPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, mTPCSliceTrackersCPU[iSlice].Data().NumberOfHits());
			ResetHelperThreads(1);
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);
			return(1);
		}
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Tracks Transfered: %d / %d", *mTPCSliceTrackersCPU[iSlice].NTracks(), *mTPCSliceTrackersCPU[iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(iSlice)) return(1);
	}
	for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(ocl->selector_events[iSlice2]);

	if (Reconstruct_Base_Finalize()) return(1);

	return(0);
}

int AliGPUReconstructionOCL::ExitDevice_Runtime()
{
	//Uninitialize OPENCL

	const int nStreams = 3;
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

	CAGPUInfo("OPENCL Uninitialized");
	return(0);
}

int AliGPUReconstructionOCL::TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	return GPUFailedMsg(clEnqueueWriteBuffer(ocl->command_queue[stream == -1 ? 0 : stream], ocl->mem_gpu, stream >= 0, (char*) res->PtrDevice() - (char*) fGPUMemory, res->Size(), res->Ptr(), nEvents, (cl_event*) evList, (cl_event*) ev));

}

int AliGPUReconstructionOCL::TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	return GPUFailedMsg(clEnqueueReadBuffer(ocl->command_queue[stream == -1 ? 0 : stream], ocl->mem_gpu, stream >= 0, (char*) res->PtrDevice() - (char*) fGPUMemory, res->Size(), res->Ptr(), nEvents, (cl_event*) evList, (cl_event*) ev));
}

int AliGPUReconstructionOCL::RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers)
{
	CAGPUFatal("Not implemented in OpenCL (Merger)");
	return(1);
}

void AliGPUReconstructionOCL::ActivateThreadContext()
{
}

void AliGPUReconstructionOCL::ReleaseThreadContext()
{
}

void AliGPUReconstructionOCL::SynchronizeGPU()
{
	const int nStreams = 3;
	for (int i = 0;i < nStreams;i++) clFinish(ocl->command_queue[i]);
}
