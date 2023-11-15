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

/// \file GPUTPCGMMergerDump.cxx
/// \author David Rohr

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUO2DataTypes.h"
#include "GPUCommonMath.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUReconstruction.h"
#include "GPUDebugStreamer.h"
#include "GPUTrackingRefit.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace gputpcgmmergertypes;

static std::vector<int> trackOrder, trackOrderReverse;
static int getTrackOrderReverse(int i) { return i == -1 ? -1 : trackOrderReverse[i]; }

void GPUTPCGMMerger::DumpSliceTracks(std::ostream& out)
{
  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Slice Tracks\n";
  trackOrder.resize(mSliceTrackInfoIndex[2 * NSLICES]);
  trackOrderReverse.resize(mSliceTrackInfoIndex[2 * NSLICES]);
  std::iota(trackOrder.begin(), trackOrder.end(), 0);
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    out << "Slice Track Info Index " << (mSliceTrackInfoIndex[iSlice + 1] - mSliceTrackInfoIndex[iSlice]) << " / " << (mSliceTrackInfoIndex[NSLICES + iSlice + 1] - mSliceTrackInfoIndex[NSLICES + iSlice]) << "\n";
    for (int iGlobal = 0; iGlobal < 2; iGlobal++) {
      std::sort(&trackOrder[mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal]], &trackOrder[mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal + 1]], [this](const int& aa, const int& bb) {
        const GPUTPCGMSliceTrack& a = mSliceTrackInfos[aa];
        const GPUTPCGMSliceTrack& b = mSliceTrackInfos[bb];
        return (a.X() != b.X()) ? (a.X() < b.X()) : (a.Y() != b.Y()) ? (a.Y() < b.Y()) : (a.Z() < b.Z());
      });
      out << "  Track type " << iGlobal << "\n";
      for (int j = mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal]; j < mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal + 1]; j++) {
        trackOrderReverse[trackOrder[j]] = j;
        const auto& trk = mSliceTrackInfos[trackOrder[j]];
        out << "    Track " << j << ": X " << trk.X() << " A " << trk.Alpha() << " Y " << trk.Y() << " Z " << trk.Z() << " Phi " << trk.SinPhi() << " Tgl " << trk.DzDs() << " QPt " << trk.QPt() << "\n";
      }
    }
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergedWithinSlices(std::ostream& out)
{
  out << "\nTPC Merger Merge Within Slices\n";
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    for (int j = mSliceTrackInfoIndex[iSlice]; j < mSliceTrackInfoIndex[iSlice + 1]; j++) {
      const auto& trk = mSliceTrackInfos[trackOrder[j]];
      if (trk.NextSegmentNeighbour()) {
        out << "  Track " << j << ": Neighbour " << getTrackOrderReverse(trk.PrevSegmentNeighbour()) << " / " << getTrackOrderReverse(trk.NextSegmentNeighbour()) << "\n";
      }
    }
  }
}

void GPUTPCGMMerger::DumpMergedBetweenSlices(std::ostream& out)
{
  out << "\nTPC Merger Merge Within Slices\n";
  for (int iSlice = 0; iSlice < NSLICES; iSlice++) {
    for (int j = mSliceTrackInfoIndex[iSlice]; j < mSliceTrackInfoIndex[iSlice + 1]; j++) {
      const auto& trk = mSliceTrackInfos[trackOrder[j]];
      if (trk.NextNeighbour() || trk.PrevNeighbour()) {
        out << "  Track " << j << ": Neighbour " << getTrackOrderReverse(trk.PrevNeighbour()) << " / " << getTrackOrderReverse(trk.NextNeighbour()) << "\n";
      }
      if (trk.PrevNeighbour() == -1 && trk.PrevSegmentNeighbour() == -1) {
        PrintMergeGraph(&trk, out);
      }
    }
  }
}

void GPUTPCGMMerger::DumpCollected(std::ostream& out)
{
  trackOrder.resize(mMemory->nOutputTracks);
  trackOrderReverse.resize(mMemory->nOutputTracks);
  std::iota(trackOrder.begin(), trackOrder.end(), 0);
  std::sort(trackOrder.begin(), trackOrder.end(), [this](const int& aa, const int& bb) {
    const GPUTPCGMMergedTrack& a = mOutputTracks[aa];
    const GPUTPCGMMergedTrack& b = mOutputTracks[bb];
    return (a.GetAlpha() != b.GetAlpha()) ? (a.GetAlpha() < b.GetAlpha()) : (a.GetParam().GetX() != b.GetParam().GetX()) ? (a.GetParam().GetX() < b.GetParam().GetX()) : (a.GetParam().GetY() != b.GetParam().GetY()) ? (a.GetParam().GetY() < b.GetParam().GetY()) : (a.GetParam().GetZ() < b.GetParam().GetZ());
  });

  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Collected Tracks\n";
  for (unsigned int i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[trackOrder[i]];
    const auto& p = trk.GetParam();
    out << "  Track " << i << ": Loop " << trk.Looper() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergeCE(std::ostream& out)
{
  out << "\nTPC Merger Merge CE\n";
  for (unsigned int i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[trackOrder[i]];
    if (trk.CCE()) {
      out << "  Track " << i << ": CCE\n";
    }
  }
}

void GPUTPCGMMerger::DumpFitPrepare(std::ostream& out)
{
  out << "\nTPC Merger Refit Prepare\n";
  out << "  Sort\n";
  for (unsigned int i = 0; i < mMemory->nOutputTracks; i++) {
    out << "    " << i << ": " << mTrackOrderAttach[trackOrder[i]] << "\n";
  }
  out << "  Clusters\n";
  for (unsigned int j = 0; j < mMemory->nOutputTracks; j++) {
    const auto& trk = mOutputTracks[trackOrder[j]];
    for (unsigned int i = trk.FirstClusterRef(); i < trk.FirstClusterRef() + trk.NClusters(); i++) {
      out << "    Cluster state " << j << "/" << (i - trk.FirstClusterRef()) << ": " << (int)mClusters[i].state << "\n";
    }
  }
  unsigned int maxId = mRec->GetParam().rec.nonConsecutiveIDs ? mMemory->nOutputTrackClusters : mNMaxClusters;
  for (unsigned int i = 0; i < maxId; i++) {
    out << "    Cluster attachment " << i << ": " << getTrackOrderReverse(mClusterAttachment[i] & attachTrackMask) << " / " << (mClusterAttachment[i] & attachFlagMask) << "\n";
  }
}

void GPUTPCGMMerger::DumpRefit(std::ostream& out)
{
  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Refit\n";
  for (unsigned int i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[trackOrder[i]];
    const auto& p = trk.GetParam();
    const auto& po = trk.OuterParam();
    out << "  Track " << i << ": OK " << trk.OK() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << " / " << trk.NClustersFitted() << " Cov " << p.GetErr2Y() << "/" << p.GetErr2Z()
#ifdef GPUCA_HAVE_O2HEADERS
        << " dEdx " << mOutputTracksdEdx[trackOrder[i]].dEdxTotTPC << "/" << mOutputTracksdEdx[trackOrder[i]].dEdxMaxTPC
#endif
        << " Outer " << po.P[0] << "/" << po.P[1] << "/" << po.P[2] << "/" << po.P[3] << "/" << po.P[4] << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpFinal(std::ostream& out)
{
  out << "\nTPC Merger Finalized\n";
  for (unsigned int j = 0; j < mMemory->nOutputTracks; j++) {
    const auto& trk = mOutputTracks[trackOrder[j]];
    for (unsigned int i = trk.FirstClusterRef(); i < trk.FirstClusterRef() + trk.NClusters(); i++) {
      out << "    Cluster state " << j << "/" << (i - trk.FirstClusterRef()) << ": " << (int)mClusters[i].state << "\n";
    }
  }
  unsigned int maxId = mRec->GetParam().rec.nonConsecutiveIDs ? mMemory->nOutputTrackClusters : mNMaxClusters;
  for (unsigned int i = 0; i < maxId; i++) {
    out << "    Cluster attachment " << i << ": " << getTrackOrderReverse(mClusterAttachment[i] & attachTrackMask) << " / " << (mClusterAttachment[i] & attachFlagMask) << "\n";
  }
}

template <int mergeType>
inline void GPUTPCGMMerger::MergedTrackStreamerInternal(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int slice1, int slice2, int mergeMode, float weight, float frac)
{
#ifdef DEBUG_STREAMER
  std::vector<int> hits1(152), hits2(152);
  for (int i = 0; i < 152; i++) {
    hits1[i] = hits2[i] = -1;
  }
  const GPUTPCTracker& tracker1 = GetConstantMem()->tpcTrackers[slice1];
  const GPUTPCGMSliceTrack& sliceTrack1 = mSliceTrackInfos[b1.TrackID()];
  const GPUTPCTrack& inTrack1 = *sliceTrack1.OrigTrack();
  for (int i = 0; i < inTrack1.NHits(); i++) {
    const GPUTPCHitId& ic1 = tracker1.TrackHits()[inTrack1.FirstHitID() + i];
    int clusterIndex = tracker1.Data().ClusterDataIndex(tracker1.Data().Row(ic1.RowIndex()), ic1.HitIndex());
    hits1[ic1.RowIndex()] = clusterIndex;
  }
  const GPUTPCTracker& tracker2 = GetConstantMem()->tpcTrackers[slice2];
  const GPUTPCGMSliceTrack& sliceTrack2 = mSliceTrackInfos[b2.TrackID()];
  const GPUTPCTrack& inTrack2 = *sliceTrack2.OrigTrack();
  for (int i = 0; i < inTrack2.NHits(); i++) {
    const GPUTPCHitId& ic2 = tracker2.TrackHits()[inTrack2.FirstHitID() + i];
    int clusterIndex = tracker2.Data().ClusterDataIndex(tracker2.Data().Row(ic2.RowIndex()), ic2.HitIndex());
    hits2[ic2.RowIndex()] = clusterIndex;
  }

  std::string debugname = std::string("debug_") + name;
  std::string treename = std::string("tree_") + name;
  o2::utils::DebugStreamer::instance()->getStreamer(debugname.c_str(), "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName(treename.c_str()).data() << "slice1=" << slice1 << "slice2=" << slice2 << "b1=" << b1 << "b2=" << b2 << "clusters1=" << hits1 << "clusters2=" << hits2 << "sliceTrack1=" << sliceTrack1 << "sliceTrack2=" << sliceTrack2 << "mergeMode=" << mergeMode << "weight=" << weight << "fraction=" << frac << "\n";
#endif
}

void GPUTPCGMMerger::MergedTrackStreamer(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int slice1, int slice2, int mergeMode, float weight, float frac)
{
#ifdef DEBUG_STREAMER
  if (mergeMode == 0) {
    MergedTrackStreamerInternal<0>(b1, b2, name, slice1, slice2, mergeMode, weight, frac);
  } else if (mergeMode >= 1 && mergeMode <= 0) {
    // MergedTrackStreamerInternal<1>(b1, b2, name, slice1, slice2, mergeMode, weight, frac); Not yet working
  }
#endif
}

const GPUTPCGMBorderTrack& GPUTPCGMMerger::MergedTrackStreamerFindBorderTrack(const GPUTPCGMBorderTrack* tracks, int N, int trackId)
{
  for (int i = 0; i < N; i++) {
    if (tracks[i].TrackID() == trackId) {
      return tracks[i];
    }
  }
  throw std::runtime_error("didn't find border track");
}

void GPUTPCGMMerger::DebugRefitMergedTrack(const GPUTPCGMMergedTrack& track)
{
  GPUTPCGMMergedTrack trk = track;
  GPUTrackingRefit refit;
  ((GPUConstantMem*)GetConstantMem())->ioPtrs.mergedTrackHitStates = ClusterStateExt();
  ((GPUConstantMem*)GetConstantMem())->ioPtrs.mergedTrackHits = Clusters();
  refit.SetPtrsFromGPUConstantMem(GetConstantMem());
  int retval = refit.RefitTrackAsGPU(trk, false, true);
  if (retval > 0) {
    GPUTPCGMPropagator prop;
    prop.SetMaterialTPC();
    prop.SetPolynomialField(&Param().polynomialField);
    prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
    prop.SetPropagateBzOnly(false);
    prop.SetMatLUT(Param().rec.useMatLUT ? GetConstantMem()->calibObjects.matLUT : nullptr);
    prop.SetTrack(&trk.Param(), trk.GetAlpha());
    int err = prop.PropagateToXAlpha(track.GetParam().GetX(), track.GetAlpha(), false);
    if (err == 0) {
      printf("REFIT RESULT %d, SnpDiff %f\n", retval, trk.GetParam().GetSinPhi() - track.GetParam().GetSinPhi());
      if (retval > 20 && fabsf(trk.GetParam().GetSinPhi() - track.GetParam().GetSinPhi()) > 0.01) {
        printf("LARGE DIFF\n");
      }
    } else {
      printf("PROPAGATE ERROR\n");
    }
  } else {
    printf("REFIT ERROR\n");
  }
}
