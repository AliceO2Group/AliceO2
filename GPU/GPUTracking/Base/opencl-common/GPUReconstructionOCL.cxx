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

#define GPUCA_GPUTYPE_OPENCL
#define __OPENCL_HOST__

#include "GPUReconstructionOCL.h"
#include "GPUReconstructionOCLInternals.h"
#include "GPUReconstructionIncludes.h"

using namespace GPUCA_NAMESPACE::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

#define quit(...)          \
  {                        \
    GPUError(__VA_ARGS__); \
    return (1);            \
  }

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_PROP(x_class, x_attributes)
#define GPUCA_KRNL_BACKEND_CLASS GPUReconstructionOCL
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL

GPUReconstructionOCL::GPUReconstructionOCL(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionDeviceBase(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionOCLInternals;
  }
  mDeviceBackendSettings.deviceType = DeviceType::OCL;
}

GPUReconstructionOCL::~GPUReconstructionOCL()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

void GPUReconstructionOCL::UpdateSettings()
{
  GPUCA_GPUReconstructionUpdateDefailts();
}

int GPUReconstructionOCL::InitDevice_Runtime()
{
  if (mMaster == nullptr) {
    cl_int ocl_error;
    cl_uint num_platforms;
    if (GPUFailedMsgI(clGetPlatformIDs(0, nullptr, &num_platforms))) {
      quit("Error getting OpenCL Platform Count");
    }
    if (num_platforms == 0) {
      quit("No OpenCL Platform found");
    }
    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("%d OpenCL Platforms found", num_platforms);
    }

    // Query platforms
    mInternals->platforms.reset(new cl_platform_id[num_platforms]);
    if (GPUFailedMsgI(clGetPlatformIDs(num_platforms, mInternals->platforms.get(), nullptr))) {
      quit("Error getting OpenCL Platforms");
    }

    bool found = false;
    if (mProcessingSettings.platformNum >= 0) {
      if (mProcessingSettings.platformNum >= (int)num_platforms) {
        quit("Invalid platform specified");
      }
      mInternals->platform = mInternals->platforms[mProcessingSettings.platformNum];
      found = true;
    } else {
      for (unsigned int i_platform = 0; i_platform < num_platforms; i_platform++) {
        char platform_profile[64] = {}, platform_version[64] = {}, platform_name[64] = {}, platform_vendor[64] = {};
        clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_PROFILE, sizeof(platform_profile), platform_profile, nullptr);
        clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
        clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        clGetPlatformInfo(mInternals->platforms[i_platform], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, nullptr);
        if (mProcessingSettings.debugLevel >= 2) {
          GPUInfo("Available Platform %d: (%s %s) %s %s", i_platform, platform_profile, platform_version, platform_vendor, platform_name);
        }
        if (CheckPlatform(i_platform)) {
          found = true;
          mInternals->platform = mInternals->platforms[i_platform];
        }
      }
    }

    if (found == false) {
      quit("Did not find compatible OpenCL Platform");
    }

    cl_uint count, bestDevice = (cl_uint)-1;
    if (GPUFailedMsgI(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count))) {
      quit("Error getting OPENCL Device Count");
    }

    // Query devices
    mInternals->devices.reset(new cl_device_id[count]);
    if (GPUFailedMsgI(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, count, mInternals->devices.get(), nullptr))) {
      quit("Error getting OpenCL devices");
    }

    char device_vendor[64], device_name[64];
    cl_device_type device_type;
    cl_uint freq, shaders;

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Available OPENCL devices:");
    }
    std::vector<bool> devicesOK(count, false);
    for (unsigned int i = 0; i < count; i++) {
      if (mProcessingSettings.debugLevel >= 3) {
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
      if (mProcessingSettings.gpuDeviceOnly && ((device_type & CL_DEVICE_TYPE_CPU) || !(device_type & CL_DEVICE_TYPE_GPU))) {
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
      if (mProcessingSettings.debugLevel >= 2) {
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
        if (mProcessingSettings.debugLevel >= 2) {
          GPUInfo("Skipping: Speed %f < %f", deviceSpeed, bestDeviceSpeed);
        }
      }
    }
    if (bestDevice == (cl_uint)-1) {
      quit("No %sOPENCL Device available, aborting OPENCL Initialisation", count ? "appropriate " : "");
    }

    if (mProcessingSettings.deviceNum > -1) {
      if (mProcessingSettings.deviceNum >= (signed)count) {
        quit("Requested device ID %d does not exist", mProcessingSettings.deviceNum);
      } else if (!devicesOK[mProcessingSettings.deviceNum]) {
        quit("Unsupported device requested (%d)", mProcessingSettings.deviceNum);
      } else {
        bestDevice = mProcessingSettings.deviceNum;
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
    if (mProcessingSettings.debugLevel >= 2) {
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
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > constantBuffer) {
      quit("Insufficient constant memory available on GPU %d < %d!", (int)constantBuffer, (int)gGPUConstantMemBufferSize);
    }
#endif

    mDeviceName = device_name;
    mDeviceName += " (OpenCL)";
    mBlockCount = shaders;
    mWarpSize = 32;
    mMaxThreads = std::max<int>(mMaxThreads, maxWorkGroup * mBlockCount);

    mInternals->context = clCreateContext(nullptr, ContextForAllPlatforms() ? count : 1, ContextForAllPlatforms() ? mInternals->devices.get() : &mInternals->device, nullptr, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      quit("Could not create OPENCL Device Context!");
    }

    if (GetOCLPrograms()) {
      return 1;
    }

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("OpenCL program and kernels loaded successfully");
    }

    mInternals->mem_gpu = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE, mDeviceMemorySize, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      clReleaseContext(mInternals->context);
      quit("OPENCL Memory Allocation Error");
    }

    mInternals->mem_constant = clCreateBuffer(mInternals->context, CL_MEM_READ_ONLY, gGPUConstantMemBufferSize, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      clReleaseMemObject(mInternals->mem_gpu);
      clReleaseContext(mInternals->context);
      quit("OPENCL Constant Memory Allocation Error");
    }

    for (int i = 0; i < mNStreams; i++) {
#ifdef CL_VERSION_2_0
      cl_queue_properties prop = 0;
      if (mProcessingSettings.deviceTimers) {
        prop |= CL_QUEUE_PROFILING_ENABLE;
      }
      mInternals->command_queue[i] = clCreateCommandQueueWithProperties(mInternals->context, mInternals->device, &prop, &ocl_error);
#else
      mInternals->command_queue[i] = clCreateCommandQueue(mInternals->context, mInternals->device, 0, &ocl_error);
#endif
      if (GPUFailedMsgI(ocl_error)) {
        quit("Error creating OpenCL command queue");
      }
    }
    if (GPUFailedMsgI(clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_gpu, 0, 0, nullptr, nullptr))) {
      quit("Error migrating buffer");
    }
    if (GPUFailedMsgI(clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_constant, 0, 0, nullptr, nullptr))) {
      quit("Error migrating buffer");
    }

    mInternals->mem_host = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mHostMemorySize, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      quit("Error allocating pinned host memory");
    }

    const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global char* constant_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) {host_mem[0] = (size_t) gpu_mem; host_mem[1] = (size_t) constant_mem;}}";
    cl_program program = clCreateProgramWithSource(mInternals->context, 1, (const char**)&krnlGetPtr, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      quit("Error creating program object");
    }
    ocl_error = clBuildProgram(program, 1, &mInternals->device, "", nullptr, nullptr);
    if (GPUFailedMsgI(ocl_error)) {
      char build_log[16384];
      clGetProgramBuildInfo(program, mInternals->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, nullptr);
      GPUImportant("Build Log:\n\n%s\n\n", build_log);
      quit("Error compiling program");
    }
    cl_kernel kernel = clCreateKernel(program, "krnlGetPtr", &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      quit("Error creating kernel");
    }
    OCLsetKernelParameters(kernel, mInternals->mem_gpu, mInternals->mem_constant, mInternals->mem_host);
    if (GPUFailedMsgI(clExecuteKernelA(mInternals->command_queue[0], kernel, 16, 16, nullptr)) ||
        GPUFailedMsgI(clFinish(mInternals->command_queue[0])) ||
        GPUFailedMsgI(clReleaseKernel(kernel)) ||
        GPUFailedMsgI(clReleaseProgram(program))) {
      quit("Error obtaining device memory ptr");
    }

    if (mProcessingSettings.debugLevel >= 2) {
      GPUInfo("Mapping hostmemory");
    }
    mHostMemoryBase = clEnqueueMapBuffer(mInternals->command_queue[0], mInternals->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, mHostMemorySize, 0, nullptr, nullptr, &ocl_error);
    if (GPUFailedMsgI(ocl_error)) {
      quit("Error allocating Page Locked Host Memory");
    }

    mDeviceMemoryBase = ((void**)mHostMemoryBase)[0];
    mDeviceConstantMem = (GPUConstantMem*)((void**)mHostMemoryBase)[1];

    if (mProcessingSettings.debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%lld bytes): %p - Host (%lld bytes): %p", (long long int)mDeviceMemorySize, mDeviceMemoryBase, (long long int)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
    }

    GPUInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d), %lld / %lld bytes host / global memory, Stack frame %d, Constant memory %lld)", bestDevice, device_vendor, device_name, (int)freq, (int)shaders, (long long int)mDeviceMemorySize,
            (long long int)mHostMemorySize, -1, (long long int)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionOCL* master = dynamic_cast<GPUReconstructionOCL*>(mMaster);
    mBlockCount = master->mBlockCount;
    mWarpSize = master->mWarpSize;
    mMaxThreads = master->mMaxThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mInternals = master->mInternals;
  }

  return (0);
}

int GPUReconstructionOCL::ExitDevice_Runtime()
{
  // Uninitialize OPENCL
  SynchronizeGPU();

  if (mMaster == nullptr) {
    if (mDeviceMemoryBase) {
      clReleaseMemObject(mInternals->mem_gpu);
      clReleaseMemObject(mInternals->mem_constant);
      for (unsigned int i = 0; i < mInternals->kernels.size(); i++) {
        clReleaseKernel(mInternals->kernels[i].first);
      }
      mInternals->kernels.clear();
    }
    if (mHostMemoryBase) {
      clEnqueueUnmapMemObject(mInternals->command_queue[0], mInternals->mem_host, mHostMemoryBase, 0, nullptr, nullptr);
      for (int i = 0; i < mNStreams; i++) {
        clReleaseCommandQueue(mInternals->command_queue[i]);
      }
      clReleaseMemObject(mInternals->mem_host);
    }

    clReleaseProgram(mInternals->program);
    clReleaseContext(mInternals->context);
    GPUInfo("OPENCL Uninitialized");
  }
  mDeviceMemoryBase = nullptr;
  mHostMemoryBase = nullptr;

  return (0);
}

size_t GPUReconstructionOCL::GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev, deviceEvent* evList, int nEvents)
{
  if (evList == nullptr) {
    nEvents = 0;
  }
  if (mProcessingSettings.debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
  }
  if (toGPU == -2) {
    GPUFailedMsg(clEnqueueCopyBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, mInternals->mem_gpu, (char*)src - (char*)mDeviceMemoryBase, (char*)dst - (char*)mDeviceMemoryBase, size, nEvents, (cl_event*)evList, (cl_event*)ev));
  } else if (toGPU) {
    GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)dst - (char*)mDeviceMemoryBase, size, src, nEvents, (cl_event*)evList, (cl_event*)ev));
  } else {
    GPUFailedMsg(clEnqueueReadBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)src - (char*)mDeviceMemoryBase, size, dst, nEvents, (cl_event*)evList, (cl_event*)ev));
  }
  return size;
}

size_t GPUReconstructionOCL::TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst)
{
  if (!(res->Type() & GPUMemoryResource::MEMORY_GPU)) {
    if (mProcessingSettings.debugLevel >= 4) {
      GPUInfo("Skipped transfer of non-GPU memory resource: %s", res->Name());
    }
    return 0;
  }
  if (mProcessingSettings.debugLevel >= 3 && (strcmp(res->Name(), "ErrorCodes") || mProcessingSettings.debugLevel >= 4)) {
    GPUInfo("Copying to %s: %s - %lld bytes", toGPU ? "GPU" : "Host", res->Name(), (long long int)res->Size());
  }
  return GPUMemCpy(dst, src, res->Size(), stream, toGPU, ev, evList, nEvents);
}

size_t GPUReconstructionOCL::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev)
{
  if (stream == -1) {
    SynchronizeGPU();
  }
  GPUFailedMsg(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_constant, stream == -1, offset, size, src, 0, nullptr, (cl_event*)ev));
  return size;
}

void GPUReconstructionOCL::ReleaseEvent(deviceEvent* ev) { GPUFailedMsg(clReleaseEvent(*(cl_event*)ev)); }

void GPUReconstructionOCL::RecordMarker(deviceEvent* ev, int stream) { GPUFailedMsg(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream], 0, nullptr, (cl_event*)ev)); }

int GPUReconstructionOCL::DoStuckProtection(int stream, void* event)
{
  if (mProcessingSettings.stuckProtection) {
    cl_int tmp = 0;
    for (int i = 0; i <= mProcessingSettings.stuckProtection / 50; i++) {
      usleep(50);
      clGetEventInfo(*(cl_event*)event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(tmp), &tmp, nullptr);
      if (tmp == CL_COMPLETE) {
        break;
      }
    }
    if (tmp != CL_COMPLETE) {
      mGPUStuck = 1;
      quit("GPU Stuck, future processing in this component is disabled, skipping event (GPU Event State %d)", (int)tmp);
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

void GPUReconstructionOCL::StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents)
{
  if (nEvents) {
    GPUFailedMsg(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream], nEvents, (cl_event*)evList, nullptr));
  }
}

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
  if (mProcessingSettings.debugLevel <= 0) {
    return (0);
  }
  for (int i = 0; i < mNStreams; i++) {
    if (GPUFailedMsgI(clFinish(mInternals->command_queue[i]))) {
      GPUError("OpenCL Error while synchronizing (%s) (Stream %d/%d)", state, stream, i);
    }
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}
