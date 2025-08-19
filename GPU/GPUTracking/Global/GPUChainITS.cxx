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
#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/ExternalAllocator.h"
#include "GPUReconstructionIncludesITS.h"
#include <algorithm>

using namespace o2::gpu;

namespace o2::its
{
class GPUFrameworkExternalAllocator final : public o2::its::ExternalAllocator
{
 public:
  void* allocate(size_t size) override
  {
    return mFWReco->AllocateDirectMemory(size, GPUMemoryResource::MEMORY_GPU);
  }
  void deallocate(char* ptr, size_t) override {}
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

GPUChainITS::GPUChainITS(GPUReconstruction* rec) : GPUChain(rec) {}

int32_t GPUChainITS::Init() { return 0; }

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
  if (mITSTimeFrame->mIsGPU) {
    auto doFWExtAlloc = [this](size_t size) -> void* { return rec()->AllocateDirectMemory(size, GPUMemoryResource::MEMORY_GPU); };

    mFrameworkAllocator.reset(new o2::its::GPUFrameworkExternalAllocator);
    mFrameworkAllocator->setReconstructionFramework(rec());
    mITSTimeFrame->setExternalAllocator(mFrameworkAllocator.get());
  }
#endif
  return mITSTimeFrame.get();
}

int32_t GPUChainITS::PrepareEvent() { return 0; }

int32_t GPUChainITS::Finalize() { return 0; }

int32_t GPUChainITS::RunChain() { return 0; }
