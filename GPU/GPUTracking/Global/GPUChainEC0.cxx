// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainEC0.cxx
/// \author David Rohr

#include "GPUChainEC0.h"
//#include "GPUReconstructionIncludesITS.h"
#include "GPUReconstructionIncludesEC0.h"
#include "DataFormatsITS/TrackITS.h"
#include <algorithm>

using namespace GPUCA_NAMESPACE::gpu;
//using namespace o2::its;
using namespace o2::ecl;

GPUChainEC0::~GPUChainEC0()
{
  mEC0TrackerTraits.reset();
  mEC0VertexerTraits.reset();
}

GPUChainEC0::GPUChainEC0(GPUReconstruction* rec, unsigned int maxTracks) : GPUChain(rec), mMaxTracks(maxTracks) {}

void GPUChainEC0::RegisterPermanentMemoryAndProcessors() { mRec->RegisterGPUProcessor(&processors()->ec0Fitter, GetRecoStepsGPU() & RecoStep::ITSTracking); }

void GPUChainEC0::RegisterGPUProcessors()
{
  if (GetRecoStepsGPU() & RecoStep::ITSTracking) {
    mRec->RegisterGPUDeviceProcessor(&processorsShadow()->ec0Fitter, &processors()->ec0Fitter);
  }
}

void GPUChainEC0::MemorySize(size_t& gpuMem, size_t& pageLockedHostMem)
{
  gpuMem = mMaxTracks * sizeof(GPUEC0Track) + GPUCA_MEMALIGN;
  pageLockedHostMem = gpuMem;
}

int GPUChainEC0::Init() { return 0; }

TrackerTraitsEC0* GPUChainEC0::GetEC0TrackerTraits()
{
#ifndef GPUCA_NO_ITS_TRAITS
  if (mEC0TrackerTraits == nullptr) {
    mRec->GetEC0Traits(&mEC0TrackerTraits, nullptr);
    mEC0TrackerTraits->SetRecoChain(this, &GPUChainEC0::PrepareAndRunEC0TrackFit);
  }
#endif
  return mEC0TrackerTraits.get();
}
VertexerTraitsEC0* GPUChainEC0::GetEC0VertexerTraits()
{
#ifndef GPUCA_NO_ITS_TRAITS
  if (mEC0VertexerTraits == nullptr) {
    mRec->GetEC0Traits(nullptr, &mEC0VertexerTraits);
  }
#endif
  return mEC0VertexerTraits.get();
}

int GPUChainEC0::PrepareEvent() { return 0; }

int GPUChainEC0::Finalize() { return 0; }

int GPUChainEC0::RunChain() { return 0; }

int GPUChainEC0::PrepareAndRunEC0TrackFit(std::vector<o2::ecl::Road>& roads, std::array<const o2::ecl::Cluster*, 7> clusters, std::array<const o2::ecl::Cell*, 5> cells, const std::array<std::vector<o2::ecl::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks)
{
  mRec->PrepareEvent();
  return RunEC0TrackFit(roads, clusters, cells, tf, tracks);
}

int GPUChainEC0::RunEC0TrackFit(std::vector<o2::ecl::Road>& roads, std::array<const o2::ecl::Cluster*, 7> clusters, std::array<const o2::ecl::Cell*, 5> cells, const std::array<std::vector<o2::ecl::TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks)
{
  auto threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::ITSTracking;
  GPUEC0Fitter& Fitter = processors()->ec0Fitter;
  GPUEC0Fitter& FitterShadow = doGPU ? processorsShadow()->ec0Fitter : Fitter;

  Fitter.clearMemory();
  Fitter.SetNumberOfRoads(roads.size());
  for (int i = 0; i < 7; i++) {
    Fitter.SetNumberTF(i, tf[i].size());
  }
  Fitter.SetMaxData(processors()->ioPtrs);
  std::copy(clusters.begin(), clusters.end(), Fitter.clusters());
  std::copy(cells.begin(), cells.end(), Fitter.cells());
  SetupGPUProcessor(&Fitter, true);
  std::copy(roads.begin(), roads.end(), Fitter.roads());
  for (int i = 0; i < 7; i++) {
    std::copy(tf[i].begin(), tf[i].end(), Fitter.trackingFrame()[i]);
  }

  WriteToConstantMemory(RecoStep::ITSTracking, (char*)&processors()->ec0Fitter - (char*)processors(), &FitterShadow, sizeof(FitterShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::ITSTracking, &Fitter, 0);
  runKernel<GPUEC0FitterKernel>(GetGridBlk(BlockCount(), 0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(RecoStep::ITSTracking, &Fitter, 0);

  SynchronizeGPU();

  for (unsigned int i = 0; i < Fitter.NumberOfTracks(); i++) {
    auto& trkin = Fitter.tracks()[i];

    tracks.emplace_back(o2::its::TrackITSExt{{trkin.X(),
                                              trkin.mAlpha,
                                              {trkin.Par()[0], trkin.Par()[1], trkin.Par()[2], trkin.Par()[3], trkin.Par()[4]},
                                              {trkin.Cov()[0], trkin.Cov()[1], trkin.Cov()[2], trkin.Cov()[3], trkin.Cov()[4], trkin.Cov()[5], trkin.Cov()[6], trkin.Cov()[7], trkin.Cov()[8], trkin.Cov()[9], trkin.Cov()[10], trkin.Cov()[11], trkin.Cov()[12], trkin.Cov()[13], trkin.Cov()[14]}},
                                             (short int)((trkin.NDF() + 5) / 2),
                                             trkin.Chi2(),
                                             0,
                                             {trkin.mOuterParam.X,
                                              trkin.mOuterParam.alpha,
                                              {trkin.mOuterParam.P[0], trkin.mOuterParam.P[1], trkin.mOuterParam.P[2], trkin.mOuterParam.P[3], trkin.mOuterParam.P[4]},
                                              {trkin.mOuterParam.C[0], trkin.mOuterParam.C[1], trkin.mOuterParam.C[2], trkin.mOuterParam.C[3], trkin.mOuterParam.C[4], trkin.mOuterParam.C[5], trkin.mOuterParam.C[6], trkin.mOuterParam.C[7], trkin.mOuterParam.C[8], trkin.mOuterParam.C[9],
                                               trkin.mOuterParam.C[10], trkin.mOuterParam.C[11], trkin.mOuterParam.C[12], trkin.mOuterParam.C[13], trkin.mOuterParam.C[14]}},
                                             {{trkin.mClusters[0], trkin.mClusters[1], trkin.mClusters[2], trkin.mClusters[3], trkin.mClusters[4], trkin.mClusters[5], trkin.mClusters[6]}}});
  }
  return 0;
}
