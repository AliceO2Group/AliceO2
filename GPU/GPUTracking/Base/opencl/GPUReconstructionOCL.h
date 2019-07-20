// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionOCL.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL_H
#define GPURECONSTRUCTIONOCL_H

#include "GPUReconstructionDeviceBase.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCLconst GPUCA_NAMESPACE::gpu::GPUSettingsProcessing& cfg);
#else
extern "C" GPUCA_NAMESPACE::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const GPUCA_NAMESPACE::gpu::GPUSettingsProcessing& cfg);
#endif

namespace GPUCA_NAMESPACE::gpu
{
struct GPUReconstructionOCLInternals;

class GPUReconstructionOCLBackend : public GPUReconstructionDeviceBase
{
 public:
  ~GPUReconstructionOCLBackend() override;

 protected:
  GPUReconstructionOCLBackend(const GPUSettingsProcessing& cfg);

  int InitDevice_Runtime() override;
  int ExitDevice_Runtime() override;
  void SetThreadCounts() override;

  void SynchronizeGPU() override;
  int DoStuckProtection(int stream, void* event) override;
  int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override;
  void SynchronizeStream(int stream) override;
  void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;

  void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
  void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, const void* src, void* dst) override;
  void GPUMemCpy(void* dst, const void* src, size_t size, int stream, bool toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) override;
  void ReleaseEvent(deviceEvent* ev) override;
  void RecordMarker(deviceEvent* ev, int stream) override;

  template <class T, int I = 0, typename... Args>
  int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);

  RecoStepField AvailableRecoSteps() override { return (RecoStep::TPCSliceTracking); }

 private:
  template <class S, class T, int I = 0>
  S& getKernelObject(int num);
  template <class T, int I = 0>
  int AddKernel(bool multi = false);
  template <class T, int I = 0>
  int FindKernel(int num);

  GPUReconstructionOCLInternals* mInternals;
  int mCoreCount = 0;
};

using GPUReconstructionOCL = GPUReconstructionKernels<GPUReconstructionOCLBackend>;
} // namespace GPUCA_NAMESPACE::gpu

#endif
