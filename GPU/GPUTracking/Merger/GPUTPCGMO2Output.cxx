// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMO2Output.cxx
/// \author David Rohr

#include "GPUTPCDef.h"
#include "GPUTPCGMO2Output.h"
#include "GPUCommonAlgorithm.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCFastTransform.h"

#ifndef GPUCA_GPUCODE
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "GPUQAHelper.h"
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::tpc::constants;

GPUdi() static constexpr unsigned char getFlagsReject() { return GPUTPCGMMergedTrackHit::flagReject | GPUTPCGMMergedTrackHit::flagNotFit; }
GPUdi() static unsigned int getFlagsRequired(const GPUSettingsRec& rec) { return rec.dropSecondaryLegsInOutput ? gputpcgmmergertypes::attachGoodLeg : gputpcgmmergertypes::attachZero; }

struct tmpSort {
  unsigned int x;
  float y;
};

template <>
GPUdii() void GPUTPCGMO2Output::Thread<GPUTPCGMO2Output::prepare>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  const GPUTPCGMMergedTrack* tracks = merger.OutputTracks();
  const unsigned int nTracks = merger.NOutputTracks();
  const GPUTPCGMMergedTrackHit* trackClusters = merger.Clusters();
  constexpr unsigned char flagsReject = getFlagsReject();
  const unsigned int flagsRequired = getFlagsRequired(merger.Param().rec);

  tmpSort* trackSort = (tmpSort*)merger.TmpMem();
  uint2* tmpData = (uint2*)(trackSort + merger.NOutputTracks());
  for (unsigned int i = get_global_id(0); i < nTracks; i += get_global_size(0)) {
    unsigned int nCl = 0;
    for (unsigned int j = 0; j < tracks[i].NClusters(); j++) {
      if (!((trackClusters[tracks[i].FirstClusterRef() + j].state & flagsReject) || (merger.ClusterAttachment()[trackClusters[tracks[i].FirstClusterRef() + j].num] & flagsRequired) != flagsRequired)) {
        nCl++;
      }
    }
    if (nCl == 0) {
      continue;
    }
    unsigned int myId = CAMath::AtomicAdd(&merger.Memory()->nO2Tracks, 1u);
    tmpData[i] = {nCl, CAMath::AtomicAdd(&merger.Memory()->nO2ClusRefs, nCl + (nCl + 1) / 2)};
    trackSort[myId] = {i, (merger.Param().par.earlyTpcTransform || tracks[i].CSide()) ? tracks[i].GetParam().GetTZOffset() : -tracks[i].GetParam().GetTZOffset()};
  }
}

template <>
GPUdii() void GPUTPCGMO2Output::Thread<GPUTPCGMO2Output::sort>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  if (iThread || iBlock) {
    return;
  }
  tmpSort* trackSort = (tmpSort*)merger.TmpMem();
  auto comp = [](const auto& a, const auto& b) { return (a.y > b.y); };
  GPUCommonAlgorithm::sortDeviceDynamic(trackSort, trackSort + merger.Memory()->nO2Tracks, comp);
}

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GPUCA_GPUCODE_GENRTC) // Specialize GPUTPCGMMergerSortTracks and GPUTPCGMMergerSortTracksQPt
struct GPUTPCGMO2OutputSort_comp {
  GPUd() bool operator()(const tmpSort& a, const tmpSort& b)
  {
    return (a.y > b.y);
  }
};

template <>
void GPUCA_KRNL_BACKEND_CLASS::runKernelBackendInternal<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(krnlSetup& _xyz)
{
  GPUDebugTiming timer(mProcessingSettings.debugLevel, nullptr, mInternals->Streams, _xyz, this);
  thrust::device_ptr<tmpSort> trackSort((tmpSort*)mProcessorsShadow->tpcMerger.TmpMem());
  ThrustVolatileAsyncAllocator alloc(this);
  thrust::sort(GPUCA_THRUST_NAMESPACE::par(alloc).on(mInternals->Streams[_xyz.x.stream]), trackSort, trackSort + processors()->tpcMerger.NOutputTracksTPCO2(), GPUTPCGMO2OutputSort_comp());
}
#endif // __CUDACC__ || __HIPCC__ - Specialize GPUTPCGMMergerSortTracks and GPUTPCGMMergerSortTracksQPt

template <>
GPUdii() void GPUTPCGMO2Output::Thread<GPUTPCGMO2Output::output>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
  constexpr float MinDelta = 0.1;
  const GPUTPCGMMergedTrack* tracks = merger.OutputTracks();
  const int nTracks = merger.NOutputTracksTPCO2();
  const GPUTPCGMMergedTrackHit* trackClusters = merger.Clusters();
  constexpr unsigned char flagsReject = getFlagsReject();
  const unsigned int flagsRequired = getFlagsRequired(merger.Param().rec);
  TrackTPC* outputTracks = merger.OutputTracksTPCO2();
  unsigned int* clusRefs = merger.OutputClusRefsTPCO2();

  tmpSort* trackSort = (tmpSort*)merger.TmpMem();
  uint2* tmpData = (uint2*)(trackSort + merger.NOutputTracks());

  for (int iTmp = get_global_id(0); iTmp < nTracks; iTmp += get_global_size(0)) {
    auto& oTrack = outputTracks[iTmp];
    const int i = trackSort[iTmp].x;

    oTrack.set(tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
               {tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(), tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt()},
               {tracks[i].GetParam().GetCov(0),
                tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
                tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
                tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8), tracks[i].GetParam().GetCov(9),
                tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11), tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14)});

    oTrack.setChi2(tracks[i].GetParam().GetChi2());
    auto& outerPar = tracks[i].OuterParam();
    oTrack.setdEdx(tracks[i].dEdxInfo());
    oTrack.setOuterParam(o2::track::TrackParCov(
      outerPar.X, outerPar.alpha,
      {outerPar.P[0], outerPar.P[1], outerPar.P[2], outerPar.P[3], outerPar.P[4]},
      {outerPar.C[0], outerPar.C[1], outerPar.C[2], outerPar.C[3], outerPar.C[4], outerPar.C[5],
       outerPar.C[6], outerPar.C[7], outerPar.C[8], outerPar.C[9], outerPar.C[10], outerPar.C[11],
       outerPar.C[12], outerPar.C[13], outerPar.C[14]}));
    unsigned int nOutCl = tmpData[i].x;
    unsigned int clBuff = tmpData[i].y;
    oTrack.setClusterRef(clBuff, nOutCl);
    unsigned int* clIndArr = &clusRefs[clBuff];
    unsigned char* sectorIndexArr = reinterpret_cast<unsigned char*>(clIndArr + nOutCl);
    unsigned char* rowIndexArr = sectorIndexArr + nOutCl;

    unsigned int nOutCl2 = 0;
    float t1 = 0, t2 = 0;
    int sector1 = 0, sector2 = 0;
    const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = merger.GetConstantMem()->ioPtrs.clustersNative;
    for (unsigned int j = 0; j < tracks[i].NClusters(); j++) {
      if ((trackClusters[tracks[i].FirstClusterRef() + j].state & flagsReject) || (merger.ClusterAttachment()[trackClusters[tracks[i].FirstClusterRef() + j].num] & flagsRequired) != flagsRequired) {
        continue;
      }
      int clusterIdGlobal = trackClusters[tracks[i].FirstClusterRef() + j].num;
      int sector = trackClusters[tracks[i].FirstClusterRef() + j].slice;
      int globalRow = trackClusters[tracks[i].FirstClusterRef() + j].row;
      int clusterIdInRow = clusterIdGlobal - clusters->clusterOffset[sector][globalRow];
      clIndArr[nOutCl2] = clusterIdInRow;
      sectorIndexArr[nOutCl2] = sector;
      rowIndexArr[nOutCl2] = globalRow;
      if (nOutCl2 == 0) {
        t1 = clusters->clustersLinear[clusterIdGlobal].getTime();
        sector1 = sector;
      }
      nOutCl2++;
      if (nOutCl2 == nOutCl) {
        t2 = clusters->clustersLinear[clusterIdGlobal].getTime();
        sector2 = sector;
      }
    }

    bool cce = tracks[i].CCE() && ((sector1 < MAXSECTOR / 2) ^ (sector2 < MAXSECTOR / 2));
    float time0 = 0.f, tFwd = 0.f, tBwd = 0.f;
    if (merger.Param().par.ContinuousTracking) {
      time0 = tracks[i].GetParam().GetTZOffset();
      if (cce) {
        bool lastSide = trackClusters[tracks[i].FirstClusterRef()].slice < MAXSECTOR / 2;
        float delta = 0.f;
        for (unsigned int iCl = 1; iCl < tracks[i].NClusters(); iCl++) {
          if (lastSide ^ (trackClusters[tracks[i].FirstClusterRef() + iCl].slice < MAXSECTOR / 2)) {
            auto& cacl1 = trackClusters[tracks[i].FirstClusterRef() + iCl];
            auto& cacl2 = trackClusters[tracks[i].FirstClusterRef() + iCl - 1];
            auto& cl1 = clusters->clustersLinear[cacl1.num];
            auto& cl2 = clusters->clustersLinear[cacl2.num];
            delta = fabs(cl1.getTime() - cl2.getTime()) * 0.5f;
            if (delta < MinDelta) {
              delta = MinDelta;
            }
            break;
          }
        }
        tFwd = tBwd = delta;
      } else {
        // estimate max/min time increments which still keep track in the physical limits of the TPC
        float tmin = CAMath::Min(t1, t2);
        float tmax = CAMath::Max(t1, t2);
        tFwd = tmin - time0;
        tBwd = time0 - tmax + merger.GetConstantMem()->calibObjects.fastTransform->getMaxDriftTime(t1 > t2 ? sector1 : sector2);
      }
    }
    oTrack.setTime0(time0);
    oTrack.setDeltaTBwd(tBwd);
    oTrack.setDeltaTFwd(tFwd);
    if (cce) {
      oTrack.setHasCSideClusters();
      oTrack.setHasASideClusters();
    } else if (tracks[i].CSide()) {
      oTrack.setHasCSideClusters();
    } else {
      oTrack.setHasASideClusters();
    }
  }
}

template <>
GPUdii() void GPUTPCGMO2Output::Thread<GPUTPCGMO2Output::mc>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() merger)
{
#ifndef GPUCA_GPUCODE
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = merger.GetConstantMem()->ioPtrs.clustersNative;
  if (clusters == nullptr || clusters->clustersMCTruth == nullptr) {
    return;
  }
  if (merger.OutputTracksTPCO2MC() == nullptr) {
    return;
  }

  auto labelAssigner = GPUTPCTrkLbl(clusters->clustersMCTruth, 0.1f);
  unsigned int* clusRefs = merger.OutputClusRefsTPCO2();
  for (unsigned int i = get_global_id(0); i < merger.NOutputTracksTPCO2(); i += get_global_size(0)) {
    labelAssigner.reset();
    const auto& trk = merger.OutputTracksTPCO2()[i];
    for (int j = 0; j < trk.getNClusters(); j++) {
      uint8_t sectorIndex, rowIndex;
      uint32_t clusterIndex;
      trk.getClusterReference(clusRefs, j, sectorIndex, rowIndex, clusterIndex);
      unsigned int clusterIdGlobal = clusters->clusterOffset[sectorIndex][rowIndex] + clusterIndex;
      labelAssigner.addLabel(clusterIdGlobal);
    }
    merger.OutputTracksTPCO2MC()[i] = labelAssigner.computeLabel();
  }
#endif
}
