#define GPUCA_GPUTYPE_RADEON

#include "AliGPUReconstructionOCL.h"

#include "AliGPUCADataTypes.h"
#include "AliCAGPULogging.h"

#include <string.h>
#include "AliGPUReconstructionOCLInternals.h"
#include "AliGPUReconstructionCommon.h"

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

#include "AliGPUCADataTypes.h"

#include <unistd.h>

#include "../makefiles/opencl_obtain_program.h"
extern "C" char _makefile_opencl_program_GlobalTracker_opencl_AliGPUReconstructionOCL_cl[];

#define quit(msg) {CAGPUError(msg);return(1);}

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

AliGPUReconstructionOCLBackend::AliGPUReconstructionOCLBackend(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionDeviceBase(cfg)
{
	mInternals = new AliGPUReconstructionOCLInternals;
	mProcessingSettings.deviceType = OCL;
	mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsCPU);

	mHostMemoryBase = nullptr;
	mInternals->selector_events = nullptr;
	mInternals->devices = nullptr;
}

AliGPUReconstructionOCLBackend::~AliGPUReconstructionOCLBackend()
{
	delete mInternals;
}

AliGPUReconstruction* AliGPUReconstruction_Create_OCL(const AliGPUCASettingsProcessing& cfg)
{
	return new AliGPUReconstructionOCL(cfg);
}

int AliGPUReconstructionOCLBackend::InitDevice_Runtime()
{
	//Find best OPENCL device, initialize and allocate memory

	cl_int ocl_error;
	cl_uint num_platforms;
	if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS) quit("Error getting OpenCL Platform Count");
	if (num_platforms == 0) quit("No OpenCL Platform found");
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("%d OpenCL Platforms found", num_platforms);
	
	//Query platforms
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	if (platforms == nullptr) quit("Memory allocation error");
	if (clGetPlatformIDs(num_platforms, platforms, nullptr) != CL_SUCCESS) quit("Error getting OpenCL Platforms");

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
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_PROFILE, 64, platform_profile, nullptr);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VERSION, 64, platform_version, nullptr);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_NAME, 64, platform_name, nullptr);
			clGetPlatformInfo(platforms[i_platform], CL_PLATFORM_VENDOR, 64, platform_vendor, nullptr);
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
	double bestDeviceSpeed = -1, deviceSpeed;
	if (GPUFailedMsg(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count)))
	{
		CAGPUError("Error getting OPENCL Device Count");
		return(1);
	}

	//Query devices
	mInternals->devices = new cl_device_id[count];
	if (mInternals->devices == nullptr) quit("Memory allocation error");
	if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, mInternals->devices, nullptr) != CL_SUCCESS) quit("Error getting OpenCL devices");

	char device_vendor[64], device_name[64];
	cl_device_type device_type;
	cl_uint freq, shaders;

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Available OPENCL devices:");
	for (unsigned int i = 0;i < count;i++)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3) {CAGPUInfo("Examining device %d\n", i);}
		cl_uint nbits;

		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_NAME, 64, device_name, nullptr);
		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_VENDOR, 64, device_vendor, nullptr);
		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, nullptr);
		clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, nullptr);
		int deviceOK = true;
		const char* deviceFailure = "";
		if (mDeviceProcessingSettings.gpuDeviceOnly && ((device_type & CL_DEVICE_TYPE_CPU) || !(device_type & CL_DEVICE_TYPE_GPU))) {deviceOK = false; deviceFailure = "No GPU device";}
		if (nbits / 8 != sizeof(void*)) {deviceOK = false; deviceFailure = "No 64 bit device";}

		deviceSpeed = (double) freq * (double) shaders;
		if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUImportant("Device %s%2d: %s %s (Frequency %d, Shaders %d, %d bit) (Speed Value: %lld)%s %s\n", deviceOK ? " " : "[", i, device_vendor, device_name, (int) freq, (int) shaders, (int) nbits, (long long int) deviceSpeed, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
		if (!deviceOK) continue;
		if (deviceSpeed > bestDeviceSpeed)
		{
			bestDevice = i;
			bestDeviceSpeed = deviceSpeed;
		}
		else
		{
			if (mDeviceProcessingSettings.debugLevel >= 0) CAGPUInfo("Skipping: Speed %f < %f\n", deviceSpeed, bestDeviceSpeed);
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
	mInternals->device = mInternals->devices[bestDevice];

	clGetDeviceInfo(mInternals->device, CL_DEVICE_NAME, 64, device_name, nullptr);
	clGetDeviceInfo(mInternals->device, CL_DEVICE_VENDOR, 64, device_vendor, nullptr);
	clGetDeviceInfo(mInternals->device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
	clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
	clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, nullptr);
	if (mDeviceProcessingSettings.debugLevel >= 2) {CAGPUDebug("Using OpenCL device %d: %s %s (Frequency %d, Shaders %d)\n", bestDevice, device_vendor, device_name, (int) freq, (int) shaders);}

	cl_uint compute_units;
	clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, nullptr);
	
	fConstructorBlockCount = compute_units * GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
	fSelectorBlockCount = compute_units * GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER;
	fConstructorThreadCount = GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;

	mInternals->context = clCreateContext(nullptr, count, mInternals->devices, nullptr, nullptr, &ocl_error);
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
		if (fp == nullptr)
		{
			CAGPUDebug("Cannot open %s\n", file);
			return(1);
		}
		fseek(fp, 0, SEEK_END);
		size_t file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		char* buffer = (char*) malloc(file_size + 1);
		if (buffer == nullptr)
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
		mInternals->program = clCreateProgramWithSource(mInternals->context, (cl_uint) 1, (const char**) &buffer, nullptr, &ocl_error);
		if (ocl_error != CL_SUCCESS) quit("Error creating program object");

		CAGPUDebug("Compiling OpenCL Program\n");
		//Compile program
		ocl_error = clBuildProgram(mInternals->program, count, mInternals->devices, "-I. -Iinclude -ISliceTracker -IHLTHeaders -IMerger -IGlobalTracker -I/home/qon/AMD-APP-SDK-v2.8.1.0-RC-lnx64/include -DGPUCA_STANDALONE -DBUILD_GPU -D_64BIT -x clc++", nullptr, nullptr);
		if (ocl_error != CL_SUCCESS)
		{
			CAGPUDebug("OpenCL Error while building program: %d (Compiler options: %s)\n", ocl_error, "");

			for (unsigned int i = 0;i < count;i++)
			{
				cl_build_status status;
				clGetProgramBuildInfo(mInternals->program, mInternals->devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr);
				if (status == CL_BUILD_ERROR)
				{
					size_t log_size;
					clGetProgramBuildInfo(mInternals->program, mInternals->devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
					char* build_log = (char*) malloc(log_size + 1);
					if (build_log == nullptr) quit("Memory allocation error");
					clGetProgramBuildInfo(mInternals->program, mInternals->devices[i], CL_PROGRAM_BUILD_LOG, log_size, build_log, nullptr);
					CAGPUDebug("Build Log (device %d):\n\n%s\n\n", i, build_log);
					free(build_log);
				}
			}
		}
	}*/

	if (_makefiles_opencl_obtain_program_helper(mInternals->context, count, mInternals->devices, &mInternals->program, _makefile_opencl_program_GlobalTracker_opencl_AliGPUReconstructionOCL_cl))
	{
		clReleaseContext(mInternals->context);
		CAGPUError("Could not obtain OpenCL progarm");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("OpenCL program loaded successfully");

	mInternals->kernel_row_blocks = clCreateKernel(mInternals->program, "PreInitRowBlocks", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 1");return(1);}
	mInternals->kernel_neighbours_finder = clCreateKernel(mInternals->program, "AliGPUTPCProcess_AliGPUTPCNeighboursFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 1");return(1);}
	mInternals->kernel_neighbours_cleaner = clCreateKernel(mInternals->program, "AliGPUTPCProcess_AliGPUTPCNeighboursCleaner", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 2");return(1);}
	mInternals->kernel_start_hits_finder = clCreateKernel(mInternals->program, "AliGPUTPCProcess_AliGPUTPCStartHitsFinder", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 3");return(1);}
	mInternals->kernel_start_hits_sorter = clCreateKernel(mInternals->program, "AliGPUTPCProcess_AliGPUTPCStartHitsSorter", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 4");return(1);}
	mInternals->kernel_tracklet_selector = clCreateKernel(mInternals->program, "AliGPUTPCProcessMulti_AliGPUTPCTrackletSelector", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 5");return(1);}
	mInternals->kernel_tracklet_constructor = clCreateKernel(mInternals->program, "AliGPUTPCTrackletConstructorGPU", &ocl_error); if (ocl_error != CL_SUCCESS) {CAGPUError("OPENCL Kernel Error 6");return(1);}
	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("OpenCL kernels created successfully");

	mInternals->mem_gpu = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE, mDeviceMemorySize, nullptr, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("OPENCL Memory Allocation Error");
		clReleaseContext(mInternals->context);
		return(1);
	}

	mInternals->mem_constant = clCreateBuffer(mInternals->context, CL_MEM_READ_ONLY, sizeof(AliGPUCAConstantMem), nullptr, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("OPENCL Constant Memory Allocation Error");
		clReleaseMemObject(mInternals->mem_gpu);
		clReleaseContext(mInternals->context);
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
		mInternals->command_queue[i] = clCreateCommandQueueWithProperties(mInternals->context, mInternals->device, nullptr, &ocl_error);
#else
		mInternals->command_queue[i] = clCreateCommandQueue(mInternals->context, mInternals->device, 0, &ocl_error);
#endif
		if (ocl_error != CL_SUCCESS) quit("Error creating OpenCL command queue");
	}
	if (clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_gpu, 0, 0, nullptr, nullptr) != CL_SUCCESS) quit("Error migrating buffer");

	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("GPU Memory used: %lld", (long long int) mDeviceMemorySize);

	mInternals->mem_host = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mHostMemorySize, nullptr, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error allocating pinned host memory");

	const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) *host_mem = (size_t) gpu_mem;}";
	cl_program program = clCreateProgramWithSource(mInternals->context, 1, (const char**) &krnlGetPtr, nullptr, &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error creating program object");
	ocl_error = clBuildProgram(program, 1, &mInternals->device, "", nullptr, nullptr);
	if (ocl_error != CL_SUCCESS)
	{
		char build_log[16384];
		clGetProgramBuildInfo(program, mInternals->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, nullptr);
		CAGPUImportant("Build Log:\n\n%s\n\n", build_log);
		quit("Error compiling program");
	}
	cl_kernel kernel = clCreateKernel(program, "krnlGetPtr", &ocl_error);
	if (ocl_error != CL_SUCCESS) quit("Error creating kernel");
	OCLsetKernelParameters(kernel, mInternals->mem_gpu, mInternals->mem_host);
	clExecuteKernelA(mInternals->command_queue[0], kernel, 16, 16, nullptr);
	clFinish(mInternals->command_queue[0]);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Mapping hostmemory");
	mHostMemoryBase = clEnqueueMapBuffer(mInternals->command_queue[0], mInternals->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, mHostMemorySize, 0, nullptr, nullptr, &ocl_error);
	if (ocl_error != CL_SUCCESS)
	{
		CAGPUError("Error allocating Page Locked Host Memory");
		return(1);
	}
	if (mDeviceProcessingSettings.debugLevel >= 1) CAGPUInfo("Host Memory used: %lld", (long long int) mHostMemorySize);

	if (mDeviceProcessingSettings.debugLevel >= 2) CAGPUInfo("Obtained Pointer to GPU Memory: %p", *((void**) mHostMemoryBase));
	mDeviceMemoryBase = *((void**) mHostMemoryBase);

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		memset(mHostMemoryBase, 0, mHostMemorySize);
	}

	mInternals->selector_events = new cl_event[NSLICES];
	
	mDeviceParam = &mParam;
	printf("WARNING: Device param set to host, needs fixing\n");

	CAGPUInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d) Thread %d, %lld bytes used)", bestDevice, device_vendor, device_name, (int) freq, (int) shaders, fThreadId, (long long int) mDeviceMemorySize);

	return(0);
}
 
int AliGPUReconstructionOCLBackend::GPUSync(const char* state, int stream, int slice)
{
	//Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards

	if (mDeviceProcessingSettings.debugLevel == 0) return(0);
	int retVal = 0;
	if (stream != -1) retVal = GPUFailedMsg(clFinish(mInternals->command_queue[stream]));
	else retVal = SynchronizeGPU();
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("GPU Sync Done");
	return(retVal);
}

int AliGPUReconstructionOCLBackend::RunTPCTrackingSlices()
{
	int retVal = RunTPCTrackingSlices_internal();
	if (retVal) SynchronizeGPU();
	if (retVal >= 2)
	{
		ResetHelperThreads(retVal >= 3);
	}
	ReleaseThreadContext();
	return(retVal != 0);
}

int AliGPUReconstructionOCLBackend::RunTPCTrackingSlices_internal()
{
	//Primary reconstruction function
	if (fGPUStuck)
	{
		CAGPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	if (Reconstruct_Base_Init()) return(1);
	if (PrepareTextures()) return(2);

	//Copy Tracker Object to GPU Memory
	if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Tracker objects to GPU");

	cl_event initEvent;
	if (GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[0], mInternals->mem_constant, CL_FALSE, 0, sizeof(AliGPUTPCTracker) * NSLICES, fGpuTracker, 0, nullptr, &initEvent)))
	{
		CAGPUError("Error writing to constant memory");
		return(2);
	}
	mInternals->cl_queue_event_done[0] = true;
	for (int i = 1;i < 2;i++) //2 queues for first phase
	{
		mInternals->cl_queue_event_done[i] = false;
	}

	if (GPUSync("Initialization (1)", 0, 0) RANDOM_ERROR)
	{
		return(2);
	}
	
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (Reconstruct_Base_SliceInit(iSlice)) return(1);

		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		OCLsetKernelParameters(mInternals->kernel_row_blocks, mInternals->mem_gpu, mInternals->mem_constant, iSlice);
		clExecuteKernelA(mInternals->command_queue[2], mInternals->kernel_row_blocks, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * fConstructorBlockCount, nullptr, &initEvent);
		if (GPUSync("Initialization (2)", 2, iSlice) RANDOM_ERROR)
		{
			return(3);
		}

		//Copy Data to GPU Global Memory
		mTPCSliceTrackersCPU[iSlice].StartTimer(0);
		if (TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResInput(), iSlice & 1, mInternals->cl_queue_event_done[iSlice & 1] ? 0 : 1, mInternals->cl_queue_event_done[iSlice & 1] ? nullptr : &initEvent, nullptr) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].Data().MemoryResRows(), iSlice & 1) ||
			TransferMemoryResourceLinkToGPU(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), iSlice & 1))
		{
			CAGPUError("Error copying data to GPU");
			return(3);
		}
		mInternals->cl_queue_event_done[iSlice & 1] = true;

		if (GPUSync("Initialization (3)", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(0);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Finder (Slice %d/%d)", iSlice, NSLICES);
		OCLsetKernelParameters(mInternals->kernel_neighbours_finder, mInternals->mem_gpu, mInternals->mem_constant, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(1);
		clExecuteKernelA(mInternals->command_queue[iSlice & 1], mInternals->kernel_neighbours_finder, GPUCA_GPU_THREAD_COUNT_FINDER, GPUCA_GPU_THREAD_COUNT_FINDER * GPUCA_ROW_COUNT, nullptr);
		if (GPUSync("Neighbours finder", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(1);

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			memcpy(mTPCSliceTrackersCPU[iSlice].LinkTmpMemory(), Res(mTPCSliceTrackersCPU[iSlice].Data().MemoryResScratch()).Ptr(), Res(mTPCSliceTrackersCPU[iSlice].Data().MemoryResScratch()).Size());
			if (mDeviceProcessingSettings.debugMask & 2) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Neighbours Cleaner (Slice %d/%d)", iSlice, NSLICES);
		OCLsetKernelParameters(mInternals->kernel_neighbours_cleaner, mInternals->mem_gpu, mInternals->mem_constant, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(2);
		clExecuteKernelA(mInternals->command_queue[iSlice & 1], mInternals->kernel_neighbours_cleaner, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * (GPUCA_ROW_COUNT - 2), nullptr);
		if (GPUSync("Neighbours Cleaner", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(2);

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice].Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) mTPCSliceTrackersCPU[iSlice].DumpLinks(mDebugFile);
		}

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Finder (Slice %d/%d)", iSlice, NSLICES);
		OCLsetKernelParameters(mInternals->kernel_start_hits_finder, mInternals->mem_gpu, mInternals->mem_constant, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(3);
		clExecuteKernelA(mInternals->command_queue[iSlice & 1], mInternals->kernel_start_hits_finder, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * (GPUCA_ROW_COUNT - 6), nullptr);
		if (GPUSync("Start Hits Finder", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(3);

		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running GPU Start Hits Sorter (Slice %d/%d)", iSlice, NSLICES);
		OCLsetKernelParameters(mInternals->kernel_start_hits_sorter, mInternals->mem_gpu, mInternals->mem_constant, iSlice);
		mTPCSliceTrackersCPU[iSlice].StartTimer(4);
		clExecuteKernelA(mInternals->command_queue[iSlice & 1], mInternals->kernel_start_hits_sorter, GPUCA_GPU_THREAD_COUNT, GPUCA_GPU_THREAD_COUNT * fConstructorBlockCount, nullptr);
		if (GPUSync("Start Hits Sorter", iSlice & 1, iSlice) RANDOM_ERROR)
		{
			return(3);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(4);

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[iSlice].MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *mTPCSliceTrackersCPU[iSlice].NTracklets(), iSlice);
			if (*mTPCSliceTrackersCPU[iSlice].NTracklets() > GPUCA_GPU_MAX_TRACKLETS RANDOM_ERROR)
			{
				CAGPUError("GPUCA_GPU_MAX_TRACKLETS constant insuffisant");
				return(3);
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
		clEnqueueMarkerWithWaitList(mInternals->command_queue[i], 0, nullptr, &initEvents2[i]);
	}

	cl_event constructorEvent;
	OCLsetKernelParameters(mInternals->kernel_tracklet_constructor, mInternals->mem_gpu, mInternals->mem_constant);
	mTPCSliceTrackersCPU[0].StartTimer(6);
	clExecuteKernelA(mInternals->command_queue[0], mInternals->kernel_tracklet_constructor, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR * fConstructorBlockCount, &constructorEvent, initEvents2, 3);
	for (int i = 0;i < 3;i++)
	{
		clReleaseEvent(initEvents2[i]);
	}
	if (GPUSync("Tracklet Constructor", 0, 0) RANDOM_ERROR)
	{
		return(1);
	}
	mTPCSliceTrackersCPU[0].StopTimer(6);
	if (DoStuckProtection(0, &constructorEvent)) return(1);
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
			return(1);
		}
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
		OCLsetKernelParameters(mInternals->kernel_tracklet_selector, mInternals->mem_gpu, mInternals->mem_constant, iSlice, runSlices);
		mTPCSliceTrackersCPU[iSlice].StartTimer(7);
		clExecuteKernelA(mInternals->command_queue[useStream], mInternals->kernel_tracklet_selector, GPUCA_GPU_THREAD_COUNT_SELECTOR, GPUCA_GPU_THREAD_COUNT_SELECTOR * fSelectorBlockCount, nullptr);
		if (GPUSync("Tracklet Selector", useStream, iSlice) RANDOM_ERROR)
		{
			return(1);
		}
		mTPCSliceTrackersCPU[iSlice].StopTimer(7);
		for (unsigned int k = iSlice;k < iSlice + runSlices;k++)
		{
			if (TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[k].MemoryResCommon(), useStream, 0, nullptr, &mInternals->selector_events[k]))
			{
				CAGPUImportant("Error transferring number of tracks from GPU to host");
				return(3);
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

		if (tmpSlice < NSLICES) GPUFailedMsg(clGetEventInfo(mInternals->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, nullptr));
		while (tmpSlice < NSLICES && (tmpSlice == iSlice ? (clFinish(mInternals->command_queue[streamMap[tmpSlice]]) == CL_SUCCESS) : (eventdone == CL_COMPLETE)))
		{
			if (*mTPCSliceTrackersCPU[tmpSlice].NTracks() > 0)
			{
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTracks(), streamMap[tmpSlice]);
				TransferMemoryResourceLinkToHost(mTPCSliceTrackersCPU[tmpSlice].MemoryResTrackHits(), streamMap[tmpSlice]);
			}
			tmpSlice++;
			if (tmpSlice < NSLICES) GPUFailedMsg(clGetEventInfo(mInternals->selector_events[tmpSlice], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, nullptr));
		}

		if (GPUFailedMsg(clFinish(mInternals->command_queue[streamMap[iSlice]])) RANDOM_ERROR)
		{
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(mInternals->selector_events[iSlice2]);
			return(3);
		}

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&mTPCSliceTrackersCPU[iSlice], -1, true);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) mTPCSliceTrackersCPU[iSlice].DumpHitWeights(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 512) mTPCSliceTrackersCPU[iSlice].DumpTrackHits(mDebugFile);
		}

		if (mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError RANDOM_ERROR)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError];
			CAGPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, mTPCSliceTrackersCPU[iSlice].Data().NumberOfHits());
			for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(mInternals->selector_events[iSlice2]);
			return(3);
		}
		if (mDeviceProcessingSettings.debugLevel >= 3) CAGPUInfo("Tracks Transfered: %d / %d", *mTPCSliceTrackersCPU[iSlice].NTracks(), *mTPCSliceTrackersCPU[iSlice].NTrackHits());

		if (Reconstruct_Base_FinishSlices(iSlice)) return(1);
	}
	for (unsigned int iSlice2 = 0;iSlice2 < NSLICES;iSlice2++) clReleaseEvent(mInternals->selector_events[iSlice2]);

	if (Reconstruct_Base_Finalize()) return(1);

	return(0);
}

int AliGPUReconstructionOCLBackend::ExitDevice_Runtime()
{
	//Uninitialize OPENCL

	const int nStreams = 3;
	for (int i = 0;i < nStreams;i++) clFinish(mInternals->command_queue[i]);

	if (mDeviceMemoryBase)
	{
		clReleaseMemObject(mInternals->mem_gpu);
		clReleaseMemObject(mInternals->mem_constant);
		mDeviceMemoryBase = nullptr;

		clReleaseKernel(mInternals->kernel_neighbours_finder);
		clReleaseKernel(mInternals->kernel_neighbours_cleaner);
		clReleaseKernel(mInternals->kernel_start_hits_finder);
		clReleaseKernel(mInternals->kernel_start_hits_sorter);
		clReleaseKernel(mInternals->kernel_tracklet_constructor);
		clReleaseKernel(mInternals->kernel_tracklet_selector);
		clReleaseKernel(mInternals->kernel_row_blocks);
	}
	if (mHostMemoryBase)
	{
		clEnqueueUnmapMemObject(mInternals->command_queue[0], mInternals->mem_host, mHostMemoryBase, 0, nullptr, nullptr);
		mHostMemoryBase = nullptr;
		for (int i = 0;i < nStreams;i++)
		{
			clReleaseCommandQueue(mInternals->command_queue[i]);
		}
		clReleaseMemObject(mInternals->mem_host);
		fGpuTracker = nullptr;
		mHostMemoryBase = nullptr;
	}

	if (mInternals->selector_events)
	{
		delete[] mInternals->selector_events;
		mInternals->selector_events = nullptr;
	}
	if (mInternals->devices)
	{
		delete[] mInternals->devices;
		mInternals->devices = nullptr;
	}

	clReleaseProgram(mInternals->program);
	clReleaseContext(mInternals->context);

	CAGPUInfo("OPENCL Uninitialized");
	return(0);
}

int AliGPUReconstructionOCLBackend::TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to GPU: %s\n", res->Name());
	if (stream == -1) SynchronizeGPU();
	return GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream >= 0, (char*) res->PtrDevice() - (char*) mDeviceMemoryBase, res->Size(), res->Ptr(), nEvents, (cl_event*) evList, (cl_event*) ev));
}

int AliGPUReconstructionOCLBackend::TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream, int nEvents, deviceEvent* evList, deviceEvent* ev)
{
	if (mDeviceProcessingSettings.debugLevel >= 3) stream = -1;
	if (mDeviceProcessingSettings.debugLevel >= 3) printf("Copying to Host: %s\n", res->Name());
	if (stream == -1) SynchronizeGPU();
	return GPUFailedMsg(clEnqueueReadBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream >= 0, (char*) res->PtrDevice() - (char*) mDeviceMemoryBase, res->Size(), res->Ptr(), nEvents, (cl_event*) evList, (cl_event*) ev));
}

int AliGPUReconstructionOCLBackend::RefitMergedTracks(AliGPUTPCGMMerger* Merger, bool resetTimers)
{
	CAGPUFatal("Not implemented in OpenCL (Merger)");
	return(1);
}

void AliGPUReconstructionOCLBackend::ActivateThreadContext()
{
}

void AliGPUReconstructionOCLBackend::ReleaseThreadContext()
{
}

int AliGPUReconstructionOCLBackend::DoStuckProtection(int stream, void* event)
{
	if (mDeviceProcessingSettings.stuckProtection)
	{
		cl_int tmp = 0;
		for (int i = 0;i <= mDeviceProcessingSettings.stuckProtection / 50;i++)
		{
			usleep(50);
			clGetEventInfo(* (cl_event*) event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(tmp), &tmp, nullptr);
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
		clFinish(mInternals->command_queue[stream]);
	}
	return 0;
}

int AliGPUReconstructionOCLBackend::SynchronizeGPU()
{
	const int nStreams = 3;
	for (int i = 0;i < nStreams;i++) GPUFailedMsg(clFinish(mInternals->command_queue[i]));
	return(0);
}
