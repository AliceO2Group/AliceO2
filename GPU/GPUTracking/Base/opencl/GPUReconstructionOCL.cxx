// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL.cxx
/// \author David Rohr

#include "GPUReconstructionOCLIncludesHost.h"
#include "GPUDefParametersLoad.inc"
#include "GPUConstantMem.h"

#include <map>

static_assert(std::is_convertible_v<cl_event, void*>, "OpenCL event type incompatible to deviceEvent");

#define GPUErrorReturn(...) \
  {                         \
    GPUError(__VA_ARGS__);  \
    return (1);             \
  }

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(GPUReconstructionOCLCode_src);
#ifdef OPENCL_ENABLED_SPIRV
QGET_LD_BINARY_SYMBOLS(GPUReconstructionOCLCode_spirv);
#endif

GPUReconstruction* GPUReconstruction_Create_OCL(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionOCL(cfg); }

GPUReconstructionOCL::GPUReconstructionOCL(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing::KernelInterface<GPUReconstructionOCL, GPUReconstructionDeviceBase>(cfg, sizeof(GPUReconstructionDeviceBase))
{
  if (mMaster == nullptr) {
    mInternals = new GPUReconstructionOCLInternals;
    *mParDevice = o2::gpu::internal::GPUDefParametersLoad();
  }
  mDeviceBackendSettings->deviceType = DeviceType::OCL;
}

GPUReconstructionOCL::~GPUReconstructionOCL()
{
  Exit(); // Make sure we destroy everything (in particular the ITS tracker) before we exit
  if (mMaster == nullptr) {
    delete mInternals;
  }
}

static_assert(sizeof(cl_int) <= sizeof(int64_t) && CL_SUCCESS == 0);
int32_t GPUReconstructionOCL::GPUChkErrInternal(const int64_t error, const char* file, int32_t line) const
{
  // Check for OPENCL Error and in the case of an error display the corresponding error string
  if (error != CL_SUCCESS) {
    GPUError("OpenCL Error: %ld / %s (%s:%d)", error, convertErrorToString(error), file, line);
  }
  return error != CL_SUCCESS;
}

int32_t GPUReconstructionOCL::InitDevice_Runtime()
{
  // Propagate processing settings to PoCL runtime.
  // Won't affect other OpenCL runtimes.
  if (int nThreads = mProcessingSettings->nHostThreads; nThreads > 0) {
    auto nThreadsStr = std::to_string(nThreads);
    setenv("POCL_CPU_MAX_CU_COUNT", nThreadsStr.c_str(), 1);
  }

  if (mMaster == nullptr) {
    cl_int ocl_error;
    cl_uint num_platforms;
    if (GPUChkErrI(clGetPlatformIDs(0, nullptr, &num_platforms))) {
      GPUErrorReturn("Error getting OpenCL Platform Count");
    }
    if (num_platforms == 0) {
      GPUErrorReturn("No OpenCL Platform found");
    }
    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("%d OpenCL Platforms found", num_platforms);
    }

    // Query platforms and devices
    std::unique_ptr<cl_platform_id[]> platforms;
    platforms.reset(new cl_platform_id[num_platforms]);
    if (GPUChkErrI(clGetPlatformIDs(num_platforms, platforms.get(), nullptr))) {
      GPUErrorReturn("Error getting OpenCL Platforms");
    }

    auto query = [&](auto func, auto obj, auto var) {
      size_t size;
      func(obj, var, 0, nullptr, &size);
      std::string retVal(size - 1, ' ');
      func(obj, var, size, retVal.data(), nullptr);
      return retVal;
    };

    std::string platform_profile, platform_version, platform_name, platform_vendor;
    float platform_version_f;
    auto queryPlatform = [&](auto platform) {
      platform_profile = query(clGetPlatformInfo, platform, CL_PLATFORM_PROFILE);
      platform_version = query(clGetPlatformInfo, platform, CL_PLATFORM_VERSION);
      platform_name = query(clGetPlatformInfo, platform, CL_PLATFORM_NAME);
      platform_vendor = query(clGetPlatformInfo, platform, CL_PLATFORM_VENDOR);
      sscanf(platform_version.c_str(), "OpenCL %f", &platform_version_f);
    };

    std::vector<cl_device_id> devices;
    std::string device_vendor, device_name, device_il_version;
    cl_device_type device_type;
    cl_uint device_freq, device_shaders, device_nbits;
    cl_bool device_endian;
    auto queryDevice = [&](auto device) {
      platform_name = query(clGetDeviceInfo, device, CL_DEVICE_NAME);
      device_vendor = query(clGetDeviceInfo, device, CL_DEVICE_VENDOR);
      device_il_version = query(clGetDeviceInfo, device, CL_DEVICE_IL_VERSION);
      clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
      clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_freq), &device_freq, nullptr);
      clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_shaders), &device_shaders, nullptr);
      clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(device_nbits), &device_nbits, nullptr);
      clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(device_endian), &device_endian, nullptr);
    };

    cl_uint deviceCount, bestDevice = (cl_uint)-1, bestPlatform = (cl_uint)-1;
    for (uint32_t iPlatform = 0; iPlatform < num_platforms; iPlatform++) {
      if (GetProcessingSettings().oclPlatformNum >= 0) {
        if (GetProcessingSettings().oclPlatformNum >= (int32_t)num_platforms) {
          GPUErrorReturn("Invalid platform specified");
        }
        iPlatform = GetProcessingSettings().oclPlatformNum;
      }
      std::string platformUsageInfo;
      bool platformCompatible = false;
      queryPlatform(platforms[iPlatform]);
      if (clGetDeviceIDs(platforms[iPlatform], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount) != CL_SUCCESS) {
        if (GetProcessingSettings().oclPlatformNum >= 0) {
          GPUErrorReturn("No device in requested platform or error obtaining device count");
        }
        platformUsageInfo += " - no devices";
      } else {
        if (platform_version_f >= 2.1f) {
          platformUsageInfo += " - OpenCL 2.2 capable";
          platformCompatible = true;
        }
      }

      if (GetProcessingSettings().oclPlatformNum >= 0 || GetProcessingSettings().debugLevel >= 2) {
        GPUInfo("%s Platform %d: (%s %s) %s %s (Compatible: %s)%s", GetProcessingSettings().oclPlatformNum >= 0 ? "Enforced" : "Available", iPlatform, platform_profile.c_str(), platform_version.c_str(), platform_vendor.c_str(), platform_name.c_str(), platformCompatible ? "yes" : "no", GetProcessingSettings().debugLevel >= 2 ? platformUsageInfo.c_str() : "");
      }

      if (platformCompatible || GetProcessingSettings().oclPlatformNum >= 0 || (GetProcessingSettings().oclPlatformNum == -2 && deviceCount)) {
        if (deviceCount > devices.size()) {
          devices.resize(deviceCount);
        }
        if (clGetDeviceIDs(platforms[iPlatform], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr) != CL_SUCCESS) {
          if (GetProcessingSettings().oclPlatformNum >= 0) {
            GPUErrorReturn("Error getting OpenCL devices");
          }
          continue;
        }

        for (uint32_t i = 0; i < deviceCount; i++) {
          if (GetProcessingSettings().deviceNum >= 0) {
            if (GetProcessingSettings().deviceNum >= (signed)deviceCount) {
              GPUErrorReturn("Requested device ID %d does not exist", GetProcessingSettings().deviceNum);
            }
            i = GetProcessingSettings().deviceNum;
          }
          bool deviceOK = true;
          queryDevice(devices[i]);
          std::string deviceFailure;
          if (GetProcessingSettings().gpuDeviceOnly && ((device_type & CL_DEVICE_TYPE_CPU) || !(device_type & CL_DEVICE_TYPE_GPU))) {
            deviceOK = false;
            deviceFailure += " - No GPU device";
          }
          if (device_nbits / 8 != sizeof(void*)) {
            deviceOK = false;
            deviceFailure += " - No 64 bit device";
          }
          if (!device_endian) {
            deviceOK = false;
            deviceFailure += " - No Little Endian Mode";
          }
          if (!GetProcessingSettings().oclCompileFromSources) {
            size_t pos = 0;
            while ((pos = device_il_version.find("SPIR-V", pos)) != std::string::npos) {
              float spirvVersion;
              sscanf(device_il_version.c_str() + pos, "SPIR-V_%f", &spirvVersion);
              if (spirvVersion >= GPUCA_OCL_SPIRV_VERSION) {
                break;
              }
              pos += strlen("SPIR-V_0.0");
            }
            if (pos == std::string::npos) {
              deviceOK = false;
              deviceFailure += " - No SPIR-V " + std::to_string(GPUCA_OCL_SPIRV_VERSION) + " (" + device_il_version + ")";
            }
          }

          double bestDeviceSpeed = -1, deviceSpeed = (double)device_freq * (double)device_shaders;
          if (GetProcessingSettings().debugLevel >= 2) {
            GPUInfo("  Device %s%2d: %s %s (Frequency %d, Shaders %d, %d bit) (Speed Value: %ld)%s %s", deviceOK ? " " : "[", i, device_vendor.c_str(), device_name.c_str(), (int32_t)device_freq, (int32_t)device_shaders, (int32_t)device_nbits, (int64_t)deviceSpeed, deviceOK ? " " : " ]", deviceOK ? "" : deviceFailure.c_str());
          }
          if (!deviceOK) {
            if (GetProcessingSettings().deviceNum >= 0) {
              GPUInfo("Unsupported device requested on platform %d: (%d)", iPlatform, GetProcessingSettings().deviceNum);
              break;
            }
            continue;
          }
          if (deviceSpeed > bestDeviceSpeed) {
            bestDevice = i;
            bestPlatform = iPlatform;
            bestDeviceSpeed = deviceSpeed;
            mOclVersion = platform_version_f;
          }
          if (GetProcessingSettings().deviceNum >= 0) {
            break;
          }
        }
      }
      if (GetProcessingSettings().oclPlatformNum >= 0) {
        break;
      }
    }

    if (bestDevice == (cl_uint)-1) {
      GPUErrorReturn("Did not find compatible OpenCL Platform / Device, aborting OPENCL Initialisation");
    }
    mInternals->platform = platforms[bestPlatform];
    GPUChkErr(clGetDeviceIDs(mInternals->platform, CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), nullptr));
    mInternals->device = devices[bestDevice];
    queryDevice(mInternals->device);

    cl_ulong deviceConstantBuffer, deviceGlobalMem, deviceLocalMem;
    std::string deviceVersion;
    size_t deviceMaxWorkGroup, deviceMaxWorkItems[3];
    clGetDeviceInfo(mInternals->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceGlobalMem), &deviceGlobalMem, nullptr);
    clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(deviceConstantBuffer), &deviceConstantBuffer, nullptr);
    clGetDeviceInfo(mInternals->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceLocalMem), &deviceLocalMem, nullptr);
    clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(deviceMaxWorkGroup), &deviceMaxWorkGroup, nullptr);
    clGetDeviceInfo(mInternals->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(deviceMaxWorkItems), deviceMaxWorkItems, nullptr);
    deviceVersion = query(clGetDeviceInfo, mInternals->device, CL_DEVICE_VERSION);
    int versionMajor, versionMinor;
    sscanf(deviceVersion.c_str(), "OpenCL %d.%d", &versionMajor, &versionMinor);
    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Using OpenCL platform %d / device %d: %s %s with properties:", bestPlatform, bestDevice, device_vendor.c_str(), device_name.c_str());
      GPUInfo("\tVersion = %s", deviceVersion);
      GPUInfo("\tFrequency = %d", (int32_t)device_freq);
      GPUInfo("\tShaders = %d", (int32_t)device_shaders);
      GPUInfo("\tGLobalMemory = %ld", (int64_t)deviceGlobalMem);
      GPUInfo("\tContantMemoryBuffer = %ld", (int64_t)deviceConstantBuffer);
      GPUInfo("\tLocalMemory = %ld", (int64_t)deviceLocalMem);
      GPUInfo("\tmaxThreadsPerBlock = %ld", (int64_t)deviceMaxWorkGroup);
      GPUInfo("\tmaxThreadsDim = %ld %ld %ld", (int64_t)deviceMaxWorkItems[0], (int64_t)deviceMaxWorkItems[1], (int64_t)deviceMaxWorkItems[2]);
      GPUInfo(" ");
    }
#ifndef GPUCA_NO_CONSTANT_MEMORY
    if (gGPUConstantMemBufferSize > deviceConstantBuffer) {
      GPUErrorReturn("Insufficient constant memory available on GPU %d < %d!", (int32_t)deviceConstantBuffer, (int32_t)gGPUConstantMemBufferSize);
    }
#endif

    mDeviceName = device_name.c_str();
    mDeviceName += " (OpenCL)";
    mMultiprocessorCount = device_shaders;
    mWarpSize = 32;
    mMaxBackendThreads = std::max<int32_t>(mMaxBackendThreads, deviceMaxWorkGroup * mMultiprocessorCount);

    mInternals->context = clCreateContext(nullptr, 1, &mInternals->device, nullptr, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      GPUErrorReturn("Could not create OPENCL Device Context!");
    }

    if (GetOCLPrograms()) {
      return 1;
    }

    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("OpenCL program and kernels loaded successfully");
    }

    mInternals->mem_gpu = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE, mDeviceMemorySize, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      clReleaseContext(mInternals->context);
      GPUErrorReturn("OPENCL Memory Allocation Error");
    }

    mInternals->mem_constant = clCreateBuffer(mInternals->context, CL_MEM_READ_ONLY, gGPUConstantMemBufferSize, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      clReleaseMemObject(mInternals->mem_gpu);
      clReleaseContext(mInternals->context);
      GPUErrorReturn("OPENCL Constant Memory Allocation Error");
    }

    if (device_type & CL_DEVICE_TYPE_CPU) {
      if (GetProcessingSettings().deviceTimers && GetProcessingSettings().debugLevel >= 2) {
        GPUInfo("Disabling device timers for CPU device");
      }
      mProcessingSettings->deviceTimers = 0;
    }
    for (int32_t i = 0; i < mNStreams; i++) {
#ifdef CL_VERSION_2_0
      cl_queue_properties prop = 0;
      if (versionMajor >= 2 && IsGPU() && GetProcessingSettings().deviceTimers) {
        prop |= CL_QUEUE_PROFILING_ENABLE;
      }
      mInternals->command_queue[i] = clCreateCommandQueueWithProperties(mInternals->context, mInternals->device, &prop, &ocl_error);
      if (GetProcessingSettings().deviceTimers && ocl_error == CL_INVALID_QUEUE_PROPERTIES) {
        GPUError("GPU device timers not supported by OpenCL platform, disabling");
        mProcessingSettings->deviceTimers = 0;
        prop = 0;
        mInternals->command_queue[i] = clCreateCommandQueueWithProperties(mInternals->context, mInternals->device, &prop, &ocl_error);
      }
#else
      mInternals->command_queue[i] = clCreateCommandQueue(mInternals->context, mInternals->device, 0, &ocl_error);
#endif
      if (GPUChkErrI(ocl_error)) {
        GPUErrorReturn("Error creating OpenCL command queue");
      }
    }
    if (GPUChkErrI(clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_gpu, 0, 0, nullptr, nullptr))) {
      GPUErrorReturn("Error migrating buffer");
    }
    if (GPUChkErrI(clEnqueueMigrateMemObjects(mInternals->command_queue[0], 1, &mInternals->mem_constant, 0, 0, nullptr, nullptr))) {
      GPUErrorReturn("Error migrating buffer");
    }

    mInternals->mem_host = clCreateBuffer(mInternals->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mHostMemorySize, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      GPUErrorReturn("Error allocating pinned host memory");
    }

    const char* krnlGetPtr = "__kernel void krnlGetPtr(__global char* gpu_mem, __global char* constant_mem, __global size_t* host_mem) {if (get_global_id(0) == 0) {host_mem[0] = (size_t) gpu_mem; host_mem[1] = (size_t) constant_mem;}}";
    cl_program program = clCreateProgramWithSource(mInternals->context, 1, (const char**)&krnlGetPtr, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      GPUErrorReturn("Error creating program object");
    }
    ocl_error = clBuildProgram(program, 1, &mInternals->device, "", nullptr, nullptr);
    if (GPUChkErrI(ocl_error)) {
      char build_log[16384];
      clGetProgramBuildInfo(program, mInternals->device, CL_PROGRAM_BUILD_LOG, 16384, build_log, nullptr);
      GPUImportant("Build Log:\n\n%s\n\n", build_log);
      GPUErrorReturn("Error compiling program");
    }
    cl_kernel kernel = clCreateKernel(program, "krnlGetPtr", &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      GPUErrorReturn("Error creating kernel");
    }

    if (GPUChkErrI(OCLsetKernelParameters(kernel, mInternals->mem_gpu, mInternals->mem_constant, mInternals->mem_host)) ||
        GPUChkErrI(clExecuteKernelA(mInternals->command_queue[0], kernel, 16, 16, nullptr)) ||
        GPUChkErrI(clFinish(mInternals->command_queue[0])) ||
        GPUChkErrI(clReleaseKernel(kernel)) ||
        GPUChkErrI(clReleaseProgram(program))) {
      GPUErrorReturn("Error obtaining device memory ptr");
    }

    if (GetProcessingSettings().debugLevel >= 2) {
      GPUInfo("Mapping hostmemory");
    }
    mHostMemoryBase = clEnqueueMapBuffer(mInternals->command_queue[0], mInternals->mem_host, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, mHostMemorySize, 0, nullptr, nullptr, &ocl_error);
    if (GPUChkErrI(ocl_error)) {
      GPUErrorReturn("Error allocating Page Locked Host Memory");
    }

    mDeviceMemoryBase = ((void**)mHostMemoryBase)[0];
    mDeviceConstantMem = (GPUConstantMem*)((void**)mHostMemoryBase)[1];

    if (GetProcessingSettings().debugLevel >= 1) {
      GPUInfo("Memory ptrs: GPU (%ld bytes): %p - Host (%ld bytes): %p", (int64_t)mDeviceMemorySize, mDeviceMemoryBase, (int64_t)mHostMemorySize, mHostMemoryBase);
      memset(mHostMemoryBase, 0xDD, mHostMemorySize);
    }

    GPUInfo("OPENCL Initialisation successfull (%d: %s %s (Frequency %d, Shaders %d), %ld / %ld bytes host / global memory, Stack frame %d, Constant memory %ld)", bestDevice, device_vendor, device_name, (int32_t)device_freq, (int32_t)device_shaders, (int64_t)mDeviceMemorySize, (int64_t)mHostMemorySize, -1, (int64_t)gGPUConstantMemBufferSize);
  } else {
    GPUReconstructionOCL* master = dynamic_cast<GPUReconstructionOCL*>(mMaster);
    mMultiprocessorCount = master->mMultiprocessorCount;
    mWarpSize = master->mWarpSize;
    mMaxBackendThreads = master->mMaxBackendThreads;
    mDeviceName = master->mDeviceName;
    mDeviceConstantMem = master->mDeviceConstantMem;
    mInternals = master->mInternals;
  }

  for (uint32_t i = 0; i < mEvents.size(); i++) {
    cl_event* events = (cl_event*)mEvents[i].data();
    new (events) cl_event[mEvents[i].size()];
  }

  return (0);
}

int32_t GPUReconstructionOCL::ExitDevice_Runtime()
{
  // Uninitialize OPENCL
  SynchronizeGPU();

  if (mMaster == nullptr) {
    if (mDeviceMemoryBase) {
      clReleaseMemObject(mInternals->mem_gpu);
      clReleaseMemObject(mInternals->mem_constant);
      for (uint32_t i = 0; i < mInternals->kernels.size(); i++) {
        clReleaseKernel(mInternals->kernels[i]);
      }
      mInternals->kernels.clear();
    }
    if (mHostMemoryBase) {
      clEnqueueUnmapMemObject(mInternals->command_queue[0], mInternals->mem_host, mHostMemoryBase, 0, nullptr, nullptr);
      for (int32_t i = 0; i < mNStreams; i++) {
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

size_t GPUReconstructionOCL::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  if (evList == nullptr) {
    nEvents = 0;
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    stream = -1;
  }
  if (stream == -1) {
    SynchronizeGPU();
  }
  if (size == 0) {
    if (ev || nEvents) { // Workaround for OCL runtimes, which can throw an error in case size = 0
      GPUChkErr(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream == -1 ? 0 : stream], nEvents, evList->getEventList<cl_event>(), ev->getEventList<cl_event>()));
    }
  } else if (toGPU == -2) {
    GPUChkErr(clEnqueueCopyBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, mInternals->mem_gpu, (char*)src - (char*)mDeviceMemoryBase, (char*)dst - (char*)mDeviceMemoryBase, size, nEvents, evList->getEventList<cl_event>(), ev->getEventList<cl_event>()));
  } else if (toGPU) {
    GPUChkErr(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)dst - (char*)mDeviceMemoryBase, size, src, nEvents, evList->getEventList<cl_event>(), ev->getEventList<cl_event>()));
  } else {
    GPUChkErr(clEnqueueReadBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_gpu, stream == -1, (char*)src - (char*)mDeviceMemoryBase, size, dst, nEvents, evList->getEventList<cl_event>(), ev->getEventList<cl_event>()));
  }
  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug(("GPUMemCpy " + std::to_string(toGPU)).c_str(), stream, true);
  }
  return size;
}

size_t GPUReconstructionOCL::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev)
{
  if (stream == -1) {
    SynchronizeGPU();
  }
  GPUChkErr(clEnqueueWriteBuffer(mInternals->command_queue[stream == -1 ? 0 : stream], mInternals->mem_constant, stream == -1, offset, size, src, 0, nullptr, ev->getEventList<cl_event>()));
  if (GetProcessingSettings().serializeGPU & 2) {
    GPUDebug("WriteToConstantMemory", stream, true);
  }
  return size;
}

void GPUReconstructionOCL::ReleaseEvent(deviceEvent ev) { GPUChkErr(clReleaseEvent(ev.get<cl_event>())); }

void GPUReconstructionOCL::RecordMarker(deviceEvent* ev, int32_t stream) { GPUChkErr(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream], 0, nullptr, ev->getEventList<cl_event>())); }

void GPUReconstructionOCL::SynchronizeGPU()
{
  for (int32_t i = 0; i < mNStreams; i++) {
    GPUChkErr(clFinish(mInternals->command_queue[i]));
  }
}

void GPUReconstructionOCL::SynchronizeStream(int32_t stream) { GPUChkErr(clFinish(mInternals->command_queue[stream])); }

void GPUReconstructionOCL::SynchronizeEvents(deviceEvent* evList, int32_t nEvents) { GPUChkErr(clWaitForEvents(nEvents, evList->getEventList<cl_event>())); }

void GPUReconstructionOCL::StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents)
{
  if (nEvents) {
    GPUChkErr(clEnqueueMarkerWithWaitList(mInternals->command_queue[stream], nEvents, evList->getEventList<cl_event>(), nullptr));
  }
}

bool GPUReconstructionOCL::IsEventDone(deviceEvent* evList, int32_t nEvents)
{
  cl_int eventdone;
  for (int32_t i = 0; i < nEvents; i++) {
    GPUChkErr(clGetEventInfo(evList[i].get<cl_event>(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventdone), &eventdone, nullptr));
    if (eventdone != CL_COMPLETE) {
      return false;
    }
  }
  return true;
}

int32_t GPUReconstructionOCL::GPUDebug(const char* state, int32_t stream, bool force)
{
  // Wait for OPENCL-Kernel to finish and check for OPENCL errors afterwards, in case of debugmode
  if (!force && GetProcessingSettings().debugLevel <= 0) {
    return (0);
  }
  for (int32_t i = 0; i < mNStreams; i++) {
    if (GPUChkErrI(clFinish(mInternals->command_queue[i]))) {
      GPUError("OpenCL Error while synchronizing (%s) (Stream %d/%d)", state, stream, i);
    }
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("GPU Sync Done");
  }
  return (0);
}

int32_t GPUReconstructionOCL::GetOCLPrograms()
{
  cl_int ocl_error;

  const char* oclBuildFlags = GetProcessingSettings().oclOverrideSourceBuildFlags != "" ? GetProcessingSettings().oclOverrideSourceBuildFlags.c_str() : GPUCA_M_STR(GPUCA_OCL_BUILD_FLAGS);

#ifdef OPENCL_ENABLED_SPIRV // clang-format off
  if (mOclVersion >= 2.1f && !GetProcessingSettings().oclCompileFromSources) {
    GPUInfo("Reading OpenCL program from SPIR-V IL (Platform version %4.2f)", mOclVersion);
    mInternals->program = clCreateProgramWithIL(mInternals->context, _binary_GPUReconstructionOCLCode_spirv_start, _binary_GPUReconstructionOCLCode_spirv_len, &ocl_error);
    oclBuildFlags = "";
  } else
#endif // clang-format on
  {
    GPUInfo("Compiling OpenCL program from sources (Platform version %4.2f)", mOclVersion);
    size_t program_sizes[1] = {_binary_GPUReconstructionOCLCode_src_len};
    char* programs_sources[1] = {_binary_GPUReconstructionOCLCode_src_start};
    mInternals->program = clCreateProgramWithSource(mInternals->context, (cl_uint)1, (const char**)&programs_sources, program_sizes, &ocl_error);
  }

  if (GPUChkErrI(ocl_error)) {
    GPUError("Error creating OpenCL program from binary");
    return 1;
  }

  if (GPUChkErrI(clBuildProgram(mInternals->program, 1, &mInternals->device, oclBuildFlags, nullptr, nullptr))) {
    cl_build_status status;
    if (GPUChkErrI(clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr)) == 0 && status == CL_BUILD_ERROR) {
      size_t log_size;
      clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::unique_ptr<char[]> build_log(new char[log_size + 1]);
      clGetProgramBuildInfo(mInternals->program, mInternals->device, CL_PROGRAM_BUILD_LOG, log_size, build_log.get(), nullptr);
      build_log[log_size] = 0;
      GPUError("Build Log:\n\n%s\n", build_log.get());
    }
    return 1;
  }

  return AddKernels();
}

const char* GPUReconstructionOCL::convertErrorToString(int32_t errorcode)
{
  static const std::map<cl_int, const char*> error_map = {
    {CL_SUCCESS, "CL_SUCCESS"},
    {CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND"},
    {CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE"},
    {CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES"},
    {CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY"},
    {CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP"},
    {CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE"},
    {CL_MAP_FAILURE, "CL_MAP_FAILURE"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {CL_COMPILE_PROGRAM_FAILURE, "CL_COMPILE_PROGRAM_FAILURE"},
    {CL_LINKER_NOT_AVAILABLE, "CL_LINKER_NOT_AVAILABLE"},
    {CL_LINK_PROGRAM_FAILURE, "CL_LINK_PROGRAM_FAILURE"},
    {CL_DEVICE_PARTITION_FAILED, "CL_DEVICE_PARTITION_FAILED"},
    {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {CL_INVALID_VALUE, "CL_INVALID_VALUE"},
    {CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE"},
    {CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM"},
    {CL_INVALID_DEVICE, "CL_INVALID_DEVICE"},
    {CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT"},
    {CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES"},
    {CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE"},
    {CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR"},
    {CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE"},
    {CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER"},
    {CL_INVALID_BINARY, "CL_INVALID_BINARY"},
    {CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS"},
    {CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM"},
    {CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME"},
    {CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION"},
    {CL_INVALID_KERNEL, "CL_INVALID_KERNEL"},
    {CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX"},
    {CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE"},
    {CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE"},
    {CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS"},
    {CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION"},
    {CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE"},
    {CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE"},
    {CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET"},
    {CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST"},
    {CL_INVALID_EVENT, "CL_INVALID_EVENT"},
    {CL_INVALID_OPERATION, "CL_INVALID_OPERATION"},
    {CL_INVALID_GL_OBJECT, "CL_INVALID_GL_OBJECT"},
    {CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE"},
    {CL_INVALID_MIP_LEVEL, "CL_INVALID_MIP_LEVEL"},
    {CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {CL_INVALID_PROPERTY, "CL_INVALID_PROPERTY"},
    {CL_INVALID_IMAGE_DESCRIPTOR, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {CL_INVALID_COMPILER_OPTIONS, "CL_INVALID_COMPILER_OPTIONS"},
    {CL_INVALID_LINKER_OPTIONS, "CL_INVALID_LINKER_OPTIONS"},
    {CL_INVALID_DEVICE_PARTITION_COUNT, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {CL_INVALID_PIPE_SIZE, "CL_INVALID_PIPE_SIZE"},
    {CL_INVALID_DEVICE_QUEUE, "CL_INVALID_DEVICE_QUEUE"},
    {CL_INVALID_SPEC_ID, "CL_INVALID_SPEC_ID"},
    {CL_MAX_SIZE_RESTRICTION_EXCEEDED, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"}};
  auto entry = error_map.find(errorcode);
  return (entry != error_map.end()) ? entry->second : "Unknown Errorcode";
}
