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
#include "GPUConstantMem.h"
#include "GPUDefParametersConstants.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITSMFTTracking/ExternalAllocator.h"
#include "GPUReconstructionIncludesITS.h"

using namespace o2::gpu;

namespace
{
class GPUFrameworkExternalAllocator final : public o2::itsmft::tracking::ExternalAllocator
{
 public:
  explicit GPUFrameworkExternalAllocator(GPUReconstruction* fwr) : mFWReco(fwr) {}
  void* allocate(size_t size, Type type) final
  {
    return mFWReco->AllocateDirectMemory(size, type);
  }
  void deallocate(char*, size_t) final {}
  void pushTagOnStack(uint64_t tag) final { mFWReco->PushNonPersistentMemory(tag); }
  void popTagOffStack(uint64_t tag) final { mFWReco->PopNonPersistentMemory(gpudatatypes::RecoStep::ITSTracking, tag); }

 private:
  GPUReconstruction* mFWReco;
};
} // namespace

GPUChainITS::~GPUChainITS() = default;

GPUChainITS::GPUChainITS(GPUReconstruction* rec) : GPUChain(rec) {}

int32_t GPUChainITS::Init() { return 0; }

void GPUChainITS::MemorySize(size_t& gpuMem, size_t& pageLockedHostMem)
{
  gpuMem = constants::GPU_DEFAULT_MEMORY_SIZE;
  pageLockedHostMem = constants::GPU_DEFAULT_HOST_MEMORY_SIZE;
}

o2::its::TrackerTraits<7>* GPUChainITS::GetITSTrackerTraits()
{
  if (mITSTrackerTraits == nullptr) {
    mRec->GetITSTraits(&mITSTrackerTraits, nullptr, nullptr);
  }
  return mITSTrackerTraits.get();
}

o2::its::VertexerTraits<7>* GPUChainITS::GetITSVertexerTraits()
{
  if (mITSVertexerTraits == nullptr) {
    mRec->GetITSTraits(nullptr, &mITSVertexerTraits, nullptr);
  }
  return mITSVertexerTraits.get();
}

o2::its::TimeFrame<7>* GPUChainITS::GetITSTimeframe()
{
  if (mITSTimeFrame == nullptr) {
    mRec->GetITSTraits(nullptr, nullptr, &mITSTimeFrame);
  }
#if !defined(GPUCA_STANDALONE)
  if (mITSTimeFrame->isGPU()) {
    mITSTimeFrame->setFrameworkAllocator(GetITSMFTFrameworkAllocator());
  }
#endif
  return mITSTimeFrame.get();
}

o2::itsmft::tracking::ExternalAllocator* GPUChainITS::GetITSMFTFrameworkAllocator()
{
  if (mFrameworkAllocator == nullptr && mRec->IsGPU()) {
    mFrameworkAllocator = std::make_unique<GPUFrameworkExternalAllocator>(rec());
  }
  return mFrameworkAllocator.get();
}

int32_t GPUChainITS::PrepareEvent() { return 0; }

int32_t GPUChainITS::Finalize() { return 0; }

int32_t GPUChainITS::RunChain() { return 0; }
