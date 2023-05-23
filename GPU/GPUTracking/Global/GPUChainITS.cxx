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
#include <algorithm>

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::its;

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

TrackerTraits* GPUChainITS::GetITSTrackerTraits()
{
#ifndef GPUCA_NO_ITS_TRAITS
  if (mITSTrackerTraits == nullptr) {
    mRec->GetITSTraits(&mITSTrackerTraits, nullptr);
    // mITSTrackerTraits->SetRecoChain(this, &GPUChainITS::PrepareAndRunITSTrackFit);
  }
#endif
  return mITSTrackerTraits.get();
}
VertexerTraits* GPUChainITS::GetITSVertexerTraits()
{
#ifndef GPUCA_NO_ITS_TRAITS
  if (mITSVertexerTraits == nullptr) {
    mRec->GetITSTraits(nullptr, &mITSVertexerTraits);
  }
#endif
  return mITSVertexerTraits.get();
}
TimeFrame* GPUChainITS::GetITSTimeframe()
{
  mRec->GetITSTimeframe(&mITSTimeFrame);
  return mITSTimeFrame.get();
}

int GPUChainITS::PrepareEvent() { return 0; }

int GPUChainITS::Finalize() { return 0; }

int GPUChainITS::RunChain() { return 0; }

int GPUChainITS::PrepareAndRunITSTrackFit(std::vector<o2::its::Road<5>>& roads, std::vector<const o2::its::Cluster*>& clusters, std::vector<const o2::its::Cell*>& cells, const std::vector<std::vector<o2::its::TrackingFrameInfo>>& tf, std::vector<o2::its::TrackITSExt>& tracks)
{
  mRec->PrepareEvent();
  return RunITSTrackFit(roads, clusters, cells, tf, tracks);
}

int GPUChainITS::RunITSTrackFit(std::vector<o2::its::Road<5>>& roads, std::vector<const o2::its::Cluster*>& clusters, std::vector<const o2::its::Cell*>& cells, const std::vector<std::vector<o2::its::TrackingFrameInfo>>& tf, std::vector<o2::its::TrackITSExt>& tracks)
{
  auto threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::ITSTracking;
  GPUITSFitter& Fitter = processors()->itsFitter;
  GPUITSFitter& FitterShadow = doGPU ? processorsShadow()->itsFitter : Fitter;

  Fitter.clearMemory();
  Fitter.SetNumberOfRoads(roads.size());
  for (unsigned int i = 0; i < clusters.size(); i++) {
    Fitter.SetNumberTF(i, tf[i].size());
  }
  Fitter.SetMaxData(processors()->ioPtrs);
  std::copy(clusters.begin(), clusters.end(), Fitter.clusters());
  std::copy(cells.begin(), cells.end(), Fitter.cells());
  SetupGPUProcessor(&Fitter, true);
  std::copy(roads.begin(), roads.end(), Fitter.roads());
  for (unsigned int i = 0; i < clusters.size(); i++) {
    std::copy(tf[i].begin(), tf[i].end(), Fitter.trackingFrame()[i]);
  }

  WriteToConstantMemory(RecoStep::ITSTracking, (char*)&processors()->itsFitter - (char*)processors(), &FitterShadow, sizeof(FitterShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::ITSTracking, &Fitter, 0);
  runKernel<GPUITSFitterKernel>(GetGridBlk(BlockCount(), 0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(RecoStep::ITSTracking, &Fitter, 0);

  SynchronizeGPU();

  for (unsigned int i = 0; i < Fitter.NumberOfTracks(); i++) {
    auto& trkin = Fitter.tracks()[i];

    tracks.emplace_back(TrackITSExt{{trkin.X(),
                                     trkin.mAlpha,
                                     {trkin.Par()[0], trkin.Par()[1], trkin.Par()[2], trkin.Par()[3], trkin.Par()[4]},
                                     {trkin.Cov()[0], trkin.Cov()[1], trkin.Cov()[2], trkin.Cov()[3], trkin.Cov()[4], trkin.Cov()[5], trkin.Cov()[6], trkin.Cov()[7], trkin.Cov()[8], trkin.Cov()[9], trkin.Cov()[10], trkin.Cov()[11], trkin.Cov()[12], trkin.Cov()[13], trkin.Cov()[14]}},
                                    (short int)((trkin.NDF() + 5) / 2),
                                    trkin.Chi2(),
                                    {trkin.mOuterParam.X,
                                     trkin.mOuterParam.alpha,
                                     {trkin.mOuterParam.P[0], trkin.mOuterParam.P[1], trkin.mOuterParam.P[2], trkin.mOuterParam.P[3], trkin.mOuterParam.P[4]},
                                     {trkin.mOuterParam.C[0], trkin.mOuterParam.C[1], trkin.mOuterParam.C[2], trkin.mOuterParam.C[3], trkin.mOuterParam.C[4], trkin.mOuterParam.C[5], trkin.mOuterParam.C[6], trkin.mOuterParam.C[7], trkin.mOuterParam.C[8], trkin.mOuterParam.C[9],
                                      trkin.mOuterParam.C[10], trkin.mOuterParam.C[11], trkin.mOuterParam.C[12], trkin.mOuterParam.C[13], trkin.mOuterParam.C[14]}},
                                    {{trkin.mClusters[0], trkin.mClusters[1], trkin.mClusters[2], trkin.mClusters[3], trkin.mClusters[4], trkin.mClusters[5], trkin.mClusters[6]}}});
  }
  return 0;
}
