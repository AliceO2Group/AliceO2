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

/// \file GPUReconstructionHIP.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONHIP_H
#define GPURECONSTRUCTIONHIP_H

#include "GPUReconstructionDeviceBase.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_HIP(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_HIP(const GPUCA_NAMESPACE::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUReconstructionHIPInternals;

class GPUReconstructionHIPBackend : public GPUReconstructionDeviceBase
{
 public:
  ~GPUReconstructionHIPBackend() override;
  int GPUFailedMsgAI(const long long int error, const char* file, int line);
  void GPUFailedMsgA(const long long int error, const char* file, int line);

 protected:
  GPUReconstructionHIPBackend(const GPUSettingsDeviceBackend& cfg);

  int InitDevice_Runtime() override;
  int ExitDevice_Runtime() override;
  void UpdateSettings() override;

  void SynchronizeGPU() override;
  int GPUDebug(const char* state = "UNKNOWN", int stream = -1, bool force = false) override;
  void SynchronizeStream(int stream) override;
  void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
  void StreamWaitForEvents(int stream, deviceEvent* evList, int nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;
  int registerMemoryForGPU(const void* ptr, size_t size) override;
  int unregisterMemoryForGPU(const void* ptr) override;
  void* getGPUPointer(void* ptr) override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
  size_t TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int stream, int toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
  void ReleaseEvent(deviceEvent* ev) override;
  void RecordMarker(deviceEvent* ev, int stream) override;

  void GetITSTraits(std::unique_ptr<o2::its::TrackerTraits>* trackerTraits, std::unique_ptr<o2::its::VertexerTraits>* vertexerTraits, std::unique_ptr<o2::its::TimeFrame>* timeFrame) override;

  void PrintKernelOccupancies() override;

  template <class T, int I = 0, typename... Args>
  int runKernelBackend(krnlSetup& _xyz, const Args&... args);
  template <class T, int I = 0, typename... Args>
  void runKernelBackendInternal(krnlSetup& _xyz, const Args&... args);
  template <class T, int I = 0>
  const krnlProperties getKernelPropertiesBackend();
  template <class T, int I>
  class backendInternal;

 private:
  GPUReconstructionHIPInternals* mInternals;
};

using GPUReconstructionHIP = GPUReconstructionKernels<GPUReconstructionHIPBackend>;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
