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

/// \file GPUReconstruction.cxx
/// \author David Rohr

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#include <conio.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#endif

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "GPUReconstruction.h"

#include "utils/qlibload.h"

#include "GPULogging.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUReconstruction* GPUReconstruction::CreateInstance(DeviceType type, bool forceType, GPUReconstruction* master)
{
  GPUSettingsDeviceBackend cfg;
  new (&cfg) GPUSettingsDeviceBackend;
  cfg.deviceType = type;
  cfg.forceDeviceType = forceType;
  cfg.master = master;
  return CreateInstance(cfg);
}

GPUReconstruction* GPUReconstruction::CreateInstance(const GPUSettingsDeviceBackend& cfg)
{
  GPUReconstruction* retVal = nullptr;
  unsigned int type = cfg.deviceType;
#ifdef DEBUG_STREAMER
  if (type != DeviceType::CPU) {
    GPUError("Cannot create GPUReconstruction for a non-CPU device if DEBUG_STREAMER are enabled");
    return nullptr;
  }
#endif
  if (type == DeviceType::CPU) {
    retVal = GPUReconstruction_Create_CPU(cfg);
  } else if (type == DeviceType::CUDA) {
    if ((retVal = sLibCUDA->GetPtr(cfg))) {
      retVal->mMyLib = sLibCUDA;
    }
  } else if (type == DeviceType::HIP) {
    if ((retVal = sLibHIP->GetPtr(cfg))) {
      retVal->mMyLib = sLibHIP;
    }
  } else if (type == DeviceType::OCL) {
    if ((retVal = sLibOCL->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL;
    }
  } else if (type == DeviceType::OCL2) {
    if ((retVal = sLibOCL2->GetPtr(cfg))) {
      retVal->mMyLib = sLibOCL2;
    }
  } else {
    GPUError("Error: Invalid device type %u", type);
    return nullptr;
  }

  if (retVal == nullptr) {
    if (cfg.forceDeviceType) {
      GPUError("Error: Could not load GPUReconstruction for specified device: %s (%u)", GPUDataTypes::DEVICE_TYPE_NAMES[type], type);
    } else if (type != DeviceType::CPU) {
      GPUError("Could not load GPUReconstruction for device type %s (%u), falling back to CPU version", GPUDataTypes::DEVICE_TYPE_NAMES[type], type);
      GPUSettingsDeviceBackend cfg2 = cfg;
      cfg2.deviceType = DeviceType::CPU;
      retVal = CreateInstance(cfg2);
    }
  } else {
    GPUInfo("Created GPUReconstruction instance for device type %s (%u)%s", GPUDataTypes::DEVICE_TYPE_NAMES[type], type, cfg.master ? " (slave)" : "");
  }

  return retVal;
}

bool GPUReconstruction::CheckInstanceAvailable(DeviceType type)
{
  if (type == DeviceType::CPU) {
    return true;
  } else if (type == DeviceType::CUDA) {
    return sLibCUDA->LoadLibrary() == 0;
  } else if (type == DeviceType::HIP) {
    return sLibHIP->LoadLibrary() == 0;
  } else if (type == DeviceType::OCL) {
    return sLibOCL->LoadLibrary() == 0;
  } else if (type == DeviceType::OCL2) {
    return sLibOCL2->LoadLibrary() == 0;
  } else {
    GPUError("Error: Invalid device type %u", (unsigned)type);
    return false;
  }
}

GPUReconstruction* GPUReconstruction::CreateInstance(const char* type, bool forceType, GPUReconstruction* master)
{
  DeviceType t = GPUDataTypes::GetDeviceType(type);
  if (t == DeviceType::INVALID_DEVICE) {
    GPUError("Invalid device type: %s", type);
    return nullptr;
  }
  return CreateInstance(t, forceType, master);
}

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibCUDA(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "CUDA" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "CUDA"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibHIP(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "HIP" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "HIP"));
std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                  "OCL" LIBRARY_EXTENSION,
                                                                                                                  "GPUReconstruction_Create_"
                                                                                                                  "OCL"));

std::shared_ptr<GPUReconstruction::LibraryLoader> GPUReconstruction::sLibOCL2(new GPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "GPUTracking"
                                                                                                                   "OCL2" LIBRARY_EXTENSION,
                                                                                                                   "GPUReconstruction_Create_"
                                                                                                                   "OCL2"));

GPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr) {}

GPUReconstruction::LibraryLoader::~LibraryLoader() { CloseLibrary(); }

int GPUReconstruction::LibraryLoader::LoadLibrary()
{
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);

  if (mGPUEntry) {
    return 0;
  }

  LIBRARY_TYPE hGPULib;
  hGPULib = LIBRARY_LOAD(mLibName);
  if (hGPULib == nullptr) {
#ifndef _WIN32
    GPUImportant("The following error occured during dlopen: %s", dlerror());
#endif
    GPUError("Error Opening cagpu library for GPU Tracker (%s)", mLibName);
    return 1;
  } else {
    void* createFunc = LIBRARY_FUNCTION(hGPULib, mFuncName);
    if (createFunc == nullptr) {
      GPUError("Error fetching entry function in GPU library\n");
      LIBRARY_CLOSE(hGPULib);
      return 1;
    } else {
      mGPULib = (void*)(size_t)hGPULib;
      mGPUEntry = createFunc;
      GPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
    }
  }
  return 0;
}

GPUReconstruction* GPUReconstruction::LibraryLoader::GetPtr(const GPUSettingsDeviceBackend& cfg)
{
  if (LoadLibrary()) {
    return nullptr;
  }
  if (mGPUEntry == nullptr) {
    return nullptr;
  }
  GPUReconstruction* (*tmp)(const GPUSettingsDeviceBackend& cfg) = (GPUReconstruction * (*)(const GPUSettingsDeviceBackend& cfg)) mGPUEntry;
  return tmp(cfg);
}

int GPUReconstruction::LibraryLoader::CloseLibrary()
{
  if (mGPUEntry == nullptr) {
    return 1;
  }
  LIBRARY_CLOSE((LIBRARY_TYPE)(size_t)mGPULib);
  mGPULib = nullptr;
  mGPUEntry = nullptr;
  return 0;
}
