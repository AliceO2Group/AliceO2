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
struct GPUReconstructionOCLInternals;

#ifdef _WIN32
extern "C" __declspec(dllexport) GPUReconstruction* GPUReconstruction_Create_OCLconst GPUSettingsProcessing& cfg);
#else
extern "C" GPUReconstruction* GPUReconstruction_Create_OCL(const GPUSettingsProcessing& cfg);
#endif

class GPUReconstructionOCLBackend : public GPUReconstructionDeviceBase
{
public:
	virtual ~GPUReconstructionOCLBackend();

protected:
	GPUReconstructionOCLBackend(const GPUSettingsProcessing& cfg);
    
	virtual int InitDevice_Runtime() override;
	virtual int ExitDevice_Runtime() override;
	virtual void SetThreadCounts() override;

	virtual void SynchronizeGPU() override;
	virtual int DoStuckProtection(int stream, void* event) override;
	virtual int GPUDebug(const char* state = "UNKNOWN", int stream = -1) override;
	virtual void SynchronizeStream(int stream) override;
	virtual void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) override;
	virtual bool IsEventDone(deviceEvent* evList, int nEvents = 1) override;

	virtual void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) override;
	virtual void TransferMemoryInternal(GPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) override;
	virtual void ReleaseEvent(deviceEvent* ev) override;
	virtual void RecordMarker(deviceEvent* ev, int stream) override;
	
	template <class T, int I = 0, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args);
	
	virtual RecoStepField AvailableRecoSteps() override {return (RecoStep::TPCSliceTracking);}

private:
	template <class S, class T, int I = 0> S& getKernelObject(int num);
	template <class T, int I = 0> int AddKernel(bool multi = false);
	template <class T, int I = 0> int FindKernel(int num);
	
	GPUReconstructionOCLInternals* mInternals;
	int mCoreCount = 0;
};

using GPUReconstructionOCL = GPUReconstructionKernels<GPUReconstructionOCLBackend>;

#endif
