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

/// \file GPUChainITS.cxx
/// \author David Rohr

#include "GPUChainITS.h"
#include "GPUReconstructionIncludesITS.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/ExternalAllocator.h"
#include <algorithm>

using namespace GPUCA_NAMESPACE::gpu;

namespace o2::its
{
class GPUFrameworkExternalAllocator : public o2::its::ExternalAllocator
{
 public:
  void* allocate(size_t size) override
  {
    return mFWReco->AllocateUnmanagedMemory(size, GPUMemoryResource::MEMORY_GPU);
  }

  void setReconstructionFramework(o2::gpu::GPUReconstruction* fwr) { mFWReco = fwr; }

 private:
  o2::gpu::GPUReconstruction* mFWReco;
};
} // namespace o2::its

GPUChainITS::~GPUChainITS()
{
  mITSTrackerTraits.reset();
  mITSVertexerTraits.reset();
}

GPUChainITS::GPUChainITS(GPUReconstruction* rec, unsigned int maxTracks) : GPUChain(rec), mMaxTracks(maxTracks) {}

void GPUChainITS::RegisterPermanentMemoryAndProcessors() { mRec->RegisterGPUProcessor(&processors()->itsFitter, GetRecoStepsGPU() & RecoStep::ITSTracking); }

void GPUChainITS::RegisterGPUProcessors()
{
  if (GetRecoStepsGPU() & RecoStep::ITSTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->itsFitter, &processors()->itsFitter);
  }
}

void GPUChainITS::MemorySize(size_t& gpuMem, size_t& pageLockedHostMem)
{
  gpuMem = mMaxTracks * sizeof(GPUITSTrack) + GPUCA_MEMALIGN;
  pageLockedHostMem = gpuMem;
}

int GPUChainITS::Init() { return 0; }

o2::its::TrackerTraits* GPUChainITS::GetITSTrackerTraits()
{
  if (mITSTrackerTraits == nullptr) {
    mRec->GetITSTraits(&mITSTrackerTraits, nullptr, nullptr);
  }
  return mITSTrackerTraits.get();
}

o2::its::VertexerTraits* GPUChainITS::GetITSVertexerTraits()
{
  if (mITSVertexerTraits == nullptr) {
    mRec->GetITSTraits(nullptr, &mITSVertexerTraits, nullptr);
  }
  return mITSVertexerTraits.get();
}

o2::its::TimeFrame* GPUChainITS::GetITSTimeframe()
{
  if (mITSTimeFrame == nullptr) {
    mRec->GetITSTraits(nullptr, nullptr, &mITSTimeFrame);
  }
#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_NO_ITS_TRAITS) // Do not access ITS traits related classes if not compiled in standalone version
  if (mITSTimeFrame->mIsGPU) {
    auto doFWExtAlloc = [this](size_t size) -> void* { return rec()->AllocateUnmanagedMemory(size, GPUMemoryResource::MEMORY_GPU); };

    mFrameworkAllocator.reset(new o2::its::GPUFrameworkExternalAllocator);
    mFrameworkAllocator->setReconstructionFramework(rec());
    mITSTimeFrame->setExternalAllocator(mFrameworkAllocator.get());
    LOGP(debug, "GPUChainITS is giving me ps: {} prop: {}", (void*)processorsShadow(), (void*)processorsShadow()->calibObjects.o2Propagator);
    mITSTimeFrame->setDevicePropagator(processorsShadow()->calibObjects.o2Propagator);
  }
#endif
  return mITSTimeFrame.get();
}

int GPUChainITS::PrepareEvent() { return 0; }

int GPUChainITS::Finalize() { return 0; }

int GPUChainITS::RunChain() { return 0; }
