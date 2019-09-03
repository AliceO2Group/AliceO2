// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL.cxx
/// \author David Rohr

#define GPUCA_GPUTYPE_RADEON

#include "GPUReconstructionOCL.h"
#include "GPUReconstructionOCLInternals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

constexpr size_t gGPUConstantMemBufferSize = (sizeof(GPUConstantMem) + sizeof(uint4) - 1);

#define quit(msg)  \
  {                \
    GPUError(msg); \
    return (1);    \
  }

GPUReconstructionOCL::GPUReconstructionOCL(const GPUSettingsProcessing& cfg) : GPUReconstructionDeviceBase(cfg)
{
  mInternals = new GPUReconstructionOCLInternals;
  mProcessingSettings.deviceType = DeviceType::OCL;
  mHostMemoryBase = nullptr;
}

GPUReconstructionOCL::~GPUReconstructionOCL()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit CUDA
  delete mInternals;
}

int GPUReconstructionOCL::InitDevice_Runtime()
{
  // Find best OPENCL device, initialize and allocate memory

  cl_int ocl_error;
  cl_uint num_platforms;
  if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS) {
    quit("Error getting OpenCL Platform Count");
  }
  if (num_platforms == 0) {
    quit("No OpenCL Platform found");
  }
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("%d OpenCL Platforms found", num_platforms);
  }

  // Query platforms
  mInternals->platforms.reset(new cl_platform_id[num_platforms]);
  if (clGetPlatformIDs(num_platforms, mInternals->platforms.get(), nullptr) != CL_SUCCESS) {
    quit("Error getting OpenCL Platforms");
  }

  bool found = false;
  if (mDeviceProcessingSettings.platformNum >= 0) {
    if (mDeviceProcessingSettings.platformNum >= (int)num_platforms) {
      GPUError("Invalid platform specified");
      return (1);
    }
    mInternals->platform = mInternals->platforms[mDeviceProcessingSettings.platformNum];
    found = true;
  } else {
    for (unsigned int i_platform = 0; i_platform < num_platforms; i_platform++) {
      char platform_profile[64], platform_version[64], platform_name[64], platform_vendor[64];
      clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_PROFILE, sizeof(platform_profile), platform_profile, nullptr);
      clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
      clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
      clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
      if (mDeviceProcessingSettings.debugLevel >= 2) {
        GPUInfo("Available Platform %d: (%s %s) %s %s", i_platform, platform_profile, platform_version, platform_vendor, platform_name);
      }
      if (CheckPlatform(i_platform)) {
        found = true;
        mInternals->platform = mInternals->platforms[i_platform];
      }
    }
  }

  if (found == false) {
    GPUError("Did not find compatible OpenCL Platform");
    return (1);
  }

  cl_uint count, bestDevice = (cl_uint)-1;
  if (GPUFailedMsgI(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count))) {
    GPUError("Error getting OPENCL Device Count");
    return (1);
  }

  // Query devices
  mInternals->devices.reset(new cl_device_id[count]);
  if (clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, count, mInternals->devices.get(), nullptr) != CL_SUCCESS) {
    quit("Error getting OpenCL devices");
  }

  char device_vendor[64], device_name[64];
  cl_device_type device_type;
  cl_uint freq, shaders;

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Available OPENCL devices:");
  }
  std::vector<bool> devicesOK(count, false);
  for (unsigned int i = 0; i < count; i++) {
    if (mDeviceProcessingSettings.debugLevel >= 3) {
      GPUInfo("Examining device %d", i);
    }
    cl_uint nbits;
    cl_bool endian;

    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_NAME, 64, device_name, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_VENDOR, 64, device_vendor, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(nbits), &nbits, nullptr);
    clGetDeviceInfo(mInternals->devices[i], CL_DEVICE_ENDIAN_LITTLE, sizeof(endian), &endian, nullptr);
    int deviceOK = true;
    const char* deviceFailure = "";
    if (mDeviceProcessingSettings.gpuDeviceOnly && ((device_type & CL_DEVICE_TYPE_CPU) || !(device_type & CL_DEVICE_TYPE_GPU))) {
      deviceOK = false;
      deviceFailure = "No GPU device";
    }
    if (nbits / 8 != sizeof(void*)) {
      deviceOK = false;
      deviceFailure = "No 64 bit device";
    }
    if (!endian) {
      deviceOK = false;
      deviceFailure = "No Little Endian Mode";
    }

    double bestDeviceSpeed = -1, deviceSpeed = (double)freq * (double)shaders;
    if (mDeviceProcessingSettings.debugLevel >= 2) {
      GPUImportant("Device %s%2d: %s %s (Frequency %d, Shaders %d, %d bit) (Speed Value: %lld)%s %s", deviceOK ? " " : "[", i, device_vendor, device_name, (int)freq, (int)shaders, (int)nbits, (long long int)deviceSpeed, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure);
    }
    if (!deviceOK) {
      continue;
    }
    devicesOK[i] = true;
    if (deviceSpeed > bestDeviceSpeed) {
      bestDevice = i;
      bestDeviceSpeed = deviceSpeed;
    } else {
      if (mDeviceProcessingSettings.debugLevel >= 0) {
        GPUInfo("Skipping: Speed %f < %f", deviceSpeed, bestDeviceSpeed);
      }
    }
  }
  if (bestDevice == (cl_uint)-1) {
    GPUWarning("No %sOPENCL Device available, aborting OPENCL Initialisation", count ? "appropriate " : "");
    return (1);
  }

  if (mDeviceProcessingSettings.deviceNum > -1) {
    if (mDeviceProcessingSettings.deviceNum >= (signed)count) {
      GPUWarning("Requested device ID %d does not exist", mDeviceProcessingSettings.deviceNum);
      return (1);
    } else if (!devicesOK[mDeviceProcessingSettings.deviceNum]) {
      GPUWarning("Unsupported device requested (%d)", mDeviceProcessingSettings.deviceNum);
      return (1);
    } else {
      bestDevice = mDeviceProcessingSettings.deviceNum;
    }
  }
  mInternals->device = mInternals->devices[bestDevice];

  cl_ulong constantBuffer, globalMem, localMem;
  char deviceVersion[64];
  size_t maxWorkGroup, maxWorkItems[3];
  clGetDeviceInfo(mInternals->device, CL_DEVICE_NAME, 64, device_name, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_VENDOR, 64, device_vendor, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(shaders), &shaders, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constantBuffer), &constantBuffer, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_VERSION, sizeof(deviceVersion) - 1, deviceVersion, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroup), &maxWorkGroup, nullptr);
  clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItems), maxWorkItems, nullptr);
  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Using OpenCL device %d: %s %s with properties:", bestDevice, device_vendor, device_name);
    GPUInfo("\tVersion = %s", deviceVersion);
    GPUInfo("\tFrequency = %d", (int)freq);
    GPUInfo("\tShaders = %d", (int)shaders);
    GPUInfo("\tGLobalMemory = %lld", (long long int)globalMem);
    GPUInfo("\tContantMemoryBuffer = %lld", (long long int)constantBuffer);
    GPUInfo("\tLocalMemory = %lld", (long long int)localMem);
    GPUInfo("\tmaxThreadsPerBlock = %lld", (long long int)maxWorkGroup);
    GPUInfo("\tmaxThreadsDim = %lld %lld %lld", (long long int)maxWorkItems[0], (long long int)maxWorkItems[1], (long long int)maxWorkItems[2]);
    GPUInfo(" ");
  }
#ifndef GPUCA_OPENCL_NO_CONSTANT_MEMORY
  if (gGPUConstantMemBufferSize > constantBuffer) {
    GPUError("Insufficient constant memory available on GPU %d < %d!", (int)constantBuffer, (int)gGPUConstantMemBufferSize);
    return (1);
  }
#endif

  mDeviceName = device_name;
  mDeviceName += " (OpenCL)";
  mCoreCount = shaders;

  mInternals->context = clCreateContext(nullptr, ContextForAllPlatforms() ? count : 1, ContextForAllPlatforms() ? mInternals->devices.get() : &mInternals->device, nullptr, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    GPUError("Could not create OPENCL Device Context!");
    return (1);
  }

  if (GetOCLPrograms()) {
    return 1;
  }

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("OpenCL program loaded successfully");
  }
  if (AddKernel<GPUMemClean16>()) {
    return 1;
  }
  if (AddKernel<GPUTPCNeighboursFinder>()) {
    return 1;
  }
  if (AddKernel<GPUTPCNeighboursCleaner>()) {
    return 1;
  }
  if (AddKernel<GPUTPCStartHitsFinder>()) {
    return 1;
  }
  if (AddKernel<GPUTPCStartHitsSorter>()) {
    return 1;
  }
  if (AddKernel<GPUTPCTrackletConstructor>()) {
    return 1;
  }
  if (AddKernel<GPUTPCTrackletConstructor, 1>()) {
    return 1;
  }
  if (AddKernel<GPUTPCTrackletSelector>(false)) {
    return 1;
  }
  if (AddKernel<GPUTPCTrackletSelector>(true)) {
    return 1;
  }

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("OpenCL kernels created successfully");
  }

  mInternals->mem_gpu = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE, mDeviceMemorySize, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    GPUError("OPENCL Memory Allocation Error");
    clReleaseContext(mInternals->context);
    return (1);
  }

  mInternals->mem_constant = clCreateBuffer(mInternals->context, CL_MEM_READ_ONLY, gGPUConstantMemBufferSize, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    GPUError("OPENCL Constant Memory Allocation Error");
    clReleaseMemObject(mInternals->mem_gpu);
    clReleaseContext(mInternals->context);
    return (1);
  }

  mNStreams = std::max(mDeviceProcessingSettings.nStreams, 3);

  for (int i = 0; i < mNStreams; i++) {
#ifdef CL_VERSION_2_0
    mInternals->command_queue[i] = clCreateCommandQueueWithProperties(mInternals->context, mInternals->device, nullptr, &ocl_error);
#else
    mInternals->command_queue[i] = clCreateCommandQueue(mInternals->context, mInternals->device, 0, &ocl_error);
#endif
    if (ocl_error != CL_SUCCESS) {
      quit("Error creating OpenCL command queue");
    }
  }
  if (clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_gpu, 0, 0, nullptr, nullptr) != CL_SUCCESS) {
    quit("Error migrating buffer");
  }
  if (clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_constant, 0, 0, nullptr, nullptr) != CL_SUCCESS) {
    quit("Error migrating buffer");
  }

  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("GPU Memory used: %lld (Ptr 0x%p)", (long long int)mDeviceMemorySize, mDeviceMemoryBase);
  }

  mInternals->mem_host = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mHostMemorySize, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    quit("Error allocating pinned host memory");
  }

  const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global char* constant_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) {host_mem[0] = (size_t) gpu_mem; host_mem[1] = (size_t) constant_mem;}}";
  cl_program program = clCreateProgramWithSource(mInternals->context, 1, (const char**)&krnlGetPtr, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    quit("Error creating program object");
  }
  ocl_error = clBuildProgram(program, 1, &mInternals->device, "", nullptr, nullptr);
  if (ocl_error != CL_SUCCESS) {
    char build_log[16384];
    clGetProgramBuildInfo(program, mInternals->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, nullptr);
    GPUImportant("Build Log:\n\n%s\n\n", build_log);
    quit("Error compiling program");
  }
  cl_kernel kernel = clCreateKernel(program, "krnlGetPtr", &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    quit("Error creating kernel");
  }
  OCLsetKernelParameters(kernel, mInternals->mem_gpu, mInternals->mem_constant, mInternals->mem_host);
  clExecuteKernelA(mInternals->command_queue[0], kernel, 16, 16, nullptr);
  clFinish(mInternals->command_queue[0]);
  clReleaseKernel(kernel);
  clReleaseProgram(program);

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Mapping hostmemory");
  }
  mHostMemoryBase = clEnqueueMapBuffer(mInternals->command_queue[0], mInternals->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, mHostMemorySize, 0, nullptr, nullptr, &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    GPUError("Error allocating Page Locked Host Memory");
    return (1);
  }
  if (mDeviceProcessingSettings.debugLevel >= 1) {
    GPUInfo("Host Memory used: %lld (Ptr 0x%p)", (long long int)mHostMemorySize, mHostMemoryBase);
  }

  if (mDeviceProcessingSettings.debugLevel >= 2) {
    GPUInfo("Obtained Pointer to GPU Memory: %p", *((void**)mHostMemoryBase));
  }
  mDeviceMemoryBase = ((void**)mHostMemoryBase)[0];
  mDeviceConstantMem = (GPUConstantMem*)((void**)mHostMemoryBase)[1];

  if (mDeviceProcessingSettings.debugLevel >= 1) {
    memset(mHostMemoryBase, 0xDD, mHostMemorySize);
  }

  GPUInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", bestDevice, device_vendor, device_name, (int)freq, (int)shaders, (long long int)mDeviceMemorySize,
          (long long int)mHostMemorySize, -1, (long long int)gGPUConstantMemBufferSize);

  return (0);
}

int GPUReconstructionOCL::ExitDevice_Runtime()
{
  // Uninitialize OPENCL
  SynchronizeGPU();

  if (mDeviceMemoryBase) {
    clReleaseMemObject(mInternals->mem_gpu);
    clReleaseMemObject(mInternals->mem_constant);
    mDeviceMemoryBase = nullptr;
    for (unsigned int i = 0; i < mInternals->kernels.size(); i++) {
      clReleaseKernel(mInternals->kernels[i].first);
    }
    mInternals->kernels.clear();
  }
  if (mHostMemoryBase) {
    clEnqueueUnmapMemObject(mInternals->command_queue[0], mInternals->mem_host, mHostMemoryBase, 0, nullptr, nullptr);
    mHostMemoryBase = nullptr;
    for (int i = 0; i < mNStreams; i++) {
      clReleaseCommandQueue(mInternals->command_queue[i]);
    }
    clReleaseMemObject(mInternals->mem_host);
    mHostMemoryBase = nullptr;
  }

  clReleaseProgram(mInternals->program);
  clReleaseContext(mInternals->context);

  GPUInfo("OPENCL Uninitialized");
  return (0);
}

void GPUReconstructionOCL::GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  if (evList == nullptr) {
    nEvents = 0;
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
  }
  if (toGPU) {
    GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)dst - (char*)mDeviceMemoryBase, size, src, nEvents, (cl_event*)evList, (cl_event*)ev));
  } else {
    GPUFailedMsg(clEnqueueReadBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)src - (char*)mDeviceMemoryBase, size, dst, nEvents, (cl_event*)evList, (cl_event*)ev));
  }
}

void GPUReconstructionOCL::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (mDeviceProcessingSettings.debugLevel >= 4) {
      GPUInfo("Skipped transfer of non-GPU memory resource: %s", res->Name());
    }
    return;
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo(toGPU ? "Copying to GPU: %s\n" : "Copying to Host: %s", res->Name());
  }
  GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

void GPUReconstructionOCL::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
  if (stream == -1) {
    SynchronizeGPU();
  }
  GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_constant, stream == -1, offset, size, src, 0, nullptr, (cl_event*)ev));
}

void GPUReconstructionOCL::ReleaseEvent(deviceEvent* ev) { GPUFailedMsg(clReleaseEvent(*(cl_event*)ev)); }

void GPUReconstructionOCL::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream], 0, nullptr, (cl_event*)ev)); }

int GPUReconstructionOCL::DoStuckProtection(int stream, void* event)
{
  if (mDeviceProcessingSettings.stuckProtection) {
    cl_int tmp = 0;
    for (int i = 0; i <= mDeviceProcessingSettings.stuckProtection / 50; i++) {
      usleep(50);
      clGetEventInfo(*(cl_event*)event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(tmp), &tmp, nullptr);
      if (tmp == CL_COMPLETE) {
        break;
      }
    }
    if (tmp != CL_COMPLETE) {
      GPUError("GPU Stuck, future processing in this component is disabled, skipping event (GPU Event State %d)", (int)tmp);
      mGPUStuck = 1;
      return (1);
    }
  } else {
    clFinish(mInternals->command_queue[stream]);
  }
  return 0;
}

void GPUReconstructionOCL::SynchronizeGPU()
{
  for (int i = 0; i < mNStreams; i++) {
    GPUFailedMsg(clFinish(mInternals->command_queue[i]));
  }
}

void GPUReconstructionOCL::SynchronizeStream(int stream) { GPUFailedMsg(clFinish(mInternals->command_queue[stream])); }

void GPUReconstructionOCL::SynchronizeEvents(deviceEvent* evList, int nEvents) { GPUFailedMsg(clWaitForEvents(nEvents, (cl_event*)evList)); }

bool GPUReconstructionOCL::IsEventDone(deviceEvent* evList, int nEvents)
{
  cl_int eventdone;
  for (int i = 0; i < nEvents; i++) {
    GPUFailedMsg(clGetEventInfo(((cl_event*)evList)[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, nullptr));
    if (eventdone != CL_COMPLETE) {
      return false;
    }
  }
  return true;
}

int GPUReconstructionOCL::GPUDebug(const char* state, int stream)
{
  // Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards, in case of debugmode
  if (mDeviceProcessingSettings.debugLevel == 0) {
    return (0);
  }
  for (int i = 0; i < mNStreams; i++) {
    if (GPUFailedMsgI(clFinish(mInternals->command_queue[i]))) {
      GPUError("OpenCL Error while synchronizing (%s) (Stream %d/%d)", state, stream, i);
    }
  }
  if (mDeviceProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

template <class T, int I>
int GPUReconstructionOCL::AddKernel(bool multi)
{
  std::string name("GPUTPCProcess_");
  if (multi) {
    name += "Multi_";
  }
  name += typeid(T).name();
  name += std::to_string(I);

  cl_int ocl_error;
  cl_kernel krnl = clCreateKernel(mInternals->program, name.c_str(), &ocl_error);
  if (ocl_error != CL_SUCCESS) {
    GPUError("OPENCL Kernel Error %s", name.c_str());
    return (1);
  }
  mInternals->kernels.emplace_back(krnl, name);
  return 0;
}

void GPUReconstructionOCL::SetThreadCounts()
{
  mThreadCount = GPUCA_THREAD_COUNT;
  mBlockCount = mCoreCount;
  mConstructorBlockCount = mBlockCount * GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER;
  mSelectorBlockCount = mBlockCount * GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER;
  mConstructorThreadCount = GPUCA_THREAD_COUNT_CONSTRUCTOR;
  mSelectorThreadCount = GPUCA_THREAD_COUNT_SELECTOR;
  mFinderThreadCount = GPUCA_THREAD_COUNT_FINDER;
  mTRDThreadCount = GPUCA_THREAD_COUNT_TRD;
}
