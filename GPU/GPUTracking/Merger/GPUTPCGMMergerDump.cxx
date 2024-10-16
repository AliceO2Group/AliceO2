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
#include "GPUParam.inc"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUReconstruction.h"
#include "GPUDebugStreamer.h"
#include "GPUTPCClusterOccupancyMap.h"
#ifdef GPUCA_HAVE_O2HEADERS
#include "GPUTrackingRefit.h"
#include "CorrectionMapsHelper.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace gputpcgmmergertypes;

void GPUTPCGMMerger::DumpSliceTracks(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Slice Tracks\n";
  for (int32_t iSlice = 0; iSlice < NSLICES; iSlice++) {
    out << "Slice Track Info Index " << (mSliceTrackInfoIndex[iSlice + 1] - mSliceTrackInfoIndex[iSlice]) << " / " << (mSliceTrackInfoIndex[NSLICES + iSlice + 1] - mSliceTrackInfoIndex[NSLICES + iSlice]) << "\n";
    for (int32_t iGlobal = 0; iGlobal < 2; iGlobal++) {
      out << "  Track type " << iGlobal << "\n";
      for (int32_t j = mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal]; j < mSliceTrackInfoIndex[iSlice + NSLICES * iGlobal + 1]; j++) {
        const auto& trk = mSliceTrackInfos[j];
        out << "    Track " << j << ": LocalId " << (iGlobal ? (trk.LocalTrackId() >> 24) : -1) << "/" << (iGlobal ? (trk.LocalTrackId() & 0xFFFFFF) : -1) << " X " << trk.X() << " offsetz " << trk.TZOffset() << " A " << trk.Alpha() << " Y " << trk.Y() << " Z " << trk.Z() << " SinPhi " << trk.SinPhi() << " CosPhi " << trk.CosPhi() << " SecPhi " << trk.SecPhi() << " Tgl " << trk.DzDs() << " QPt " << trk.QPt() << "\n";
      }
    }
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergeRanges(std::ostream& out, int32_t withinSlice, int32_t mergeMode) const
{
  int32_t n = withinSlice == -1 ? NSLICES / 2 : NSLICES;
  for (int32_t i = 0; i < n; i++) {
    int32_t n1, n2;
    GPUTPCGMBorderTrack *b1, *b2;
    int32_t jSlice;
    MergeBorderTracksSetup(n1, n2, b1, b2, jSlice, i, withinSlice, mergeMode);
    const int32_t nTrk = Param().rec.tpc.mergerReadFromTrackerDirectly ? *mRec->GetConstantMem().tpcTrackers[jSlice].NTracks() : mkSlices[jSlice]->NTracks();
    const gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = BorderRange(i);
    const gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = BorderRange(jSlice) + nTrk;
    out << "\nBorder Tracks : i " << i << " withinSlice " << withinSlice << " mergeMode " << mergeMode << "\n";
    for (int32_t k = 0; k < n1; k++) {
      out << "  " << k << ": t " << b1[k].TrackID() << " ncl " << b1[k].NClusters() << " row " << (mergeMode > 0 ? b1[k].Row() : -1) << " par " << b1[k].Par()[0] << " " << b1[k].Par()[1] << " " << b1[k].Par()[2] << " " << b1[k].Par()[3] << " " << b1[k].Par()[4]
          << " offset " << b1[k].ZOffsetLinear() << " cov " << b1[k].Cov()[0] << " " << b1[k].Cov()[1] << " " << b1[k].Cov()[2] << " " << b1[k].Cov()[3] << " " << b1[k].Cov()[4] << " covd " << b1[k].CovD()[0] << " " << b1[k].CovD()[1] << "\n";
    }
    if (i != jSlice) {
      for (int32_t k = 0; k < n2; k++) {
        out << "  " << k << ": t " << b2[k].TrackID() << " ncl " << b2[k].NClusters() << " row " << (mergeMode > 0 ? b2[k].Row() : -1) << " par " << b2[k].Par()[0] << " " << b2[k].Par()[1] << " " << b2[k].Par()[2] << " " << b2[k].Par()[3] << " " << b2[k].Par()[4]
            << " offset " << b2[k].ZOffsetLinear() << " cov " << b2[k].Cov()[0] << " " << b2[k].Cov()[1] << " " << b2[k].Cov()[2] << " " << b2[k].Cov()[3] << " " << b2[k].Cov()[4] << " covd " << b2[k].CovD()[0] << " " << b2[k].CovD()[1] << "\n";
      }
    }
    out << "\nBorder Range : i " << i << " withinSlice " << withinSlice << " mergeMode " << mergeMode << "\n";
    for (int32_t k = 0; k < n1; k++) {
      out << "  " << k << ": " << range1[k].fId << " " << range1[k].fMin << " " << range1[k].fMax << "\n";
    }
    for (int32_t k = 0; k < n2; k++) {
      out << "  " << k << ": " << range2[k].fId << " " << range2[k].fMin << " " << range2[k].fMax << "\n";
    }
  }
}

void GPUTPCGMMerger::DumpTrackLinks(std::ostream& out, bool output, const char* type) const
{
  out << "\nTPC Merger Links " << type << "\n";
  const int32_t n = output ? mMemory->nOutputTracks : SliceTrackInfoLocalTotal();
  for (int32_t i = 0; i < n; i++) {
    if (mTrackLinks[i] != -1) {
      out << "  " << i << ": " << mTrackLinks[i] << "\n";
    }
  }
}

void GPUTPCGMMerger::DumpMergedWithinSlices(std::ostream& out) const
{
  DumpTrackLinks(out, false, "within Slices");
  out << "\nTPC Merger Merge Within Slices\n";
  for (int32_t iSlice = 0; iSlice < NSLICES; iSlice++) {
    for (int32_t j = mSliceTrackInfoIndex[iSlice]; j < mSliceTrackInfoIndex[iSlice + 1]; j++) {
      const auto& trk = mSliceTrackInfos[j];
      if (trk.NextSegmentNeighbour() >= 0 || trk.PrevSegmentNeighbour() >= 0) {
        out << "  Track " << j << ": Neighbour " << trk.PrevSegmentNeighbour() << " / " << trk.NextSegmentNeighbour() << "\n";
      }
    }
  }
}

void GPUTPCGMMerger::DumpMergedBetweenSlices(std::ostream& out) const
{
  DumpTrackLinks(out, false, "between Slices");
  out << "\nTPC Merger Merge Between Slices\n";
  for (int32_t iSlice = 0; iSlice < NSLICES; iSlice++) {
    for (int32_t j = mSliceTrackInfoIndex[iSlice]; j < mSliceTrackInfoIndex[iSlice + 1]; j++) {
      const auto& trk = mSliceTrackInfos[j];
      if (trk.NextNeighbour() >= 0 || trk.PrevNeighbour() >= 0) {
        out << "  Track " << j << ": Neighbour " << trk.PrevNeighbour() << " / " << trk.NextNeighbour() << "\n";
      }
      if (trk.PrevNeighbour() == -1 && trk.PrevSegmentNeighbour() == -1) {
        PrintMergeGraph(&trk, out);
      }
    }
  }
}

void GPUTPCGMMerger::DumpCollected(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Collected Tracks\n";
  for (uint32_t i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[i];
    const auto& p = trk.GetParam();
    out << "  Track " << i << ": Loop " << trk.Looper() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " offset " << p.GetTZOffset() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergeCE(std::ostream& out) const
{
  DumpTrackLinks(out, true, " for CE merging");
  out << "\nTPC Merger Merge CE\n";
  for (uint32_t i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[i];
    if (trk.CCE()) {
      out << "  Track " << i << ": CCE\n";
    }
  }
}

void GPUTPCGMMerger::DumpFitPrepare(std::ostream& out) const
{
  out << "\nTPC Merger Refit Prepare\n";
  out << "  Sort\n";
  for (uint32_t i = 0; i < mMemory->nOutputTracks; i++) {
    out << "    " << i << ": " << mTrackOrderAttach[i] << "\n";
  }
  out << "  Clusters\n";
  for (uint32_t j = 0; j < mMemory->nOutputTracks; j++) {
    const auto& trk = mOutputTracks[j];
    out << "  Track " << j << ": ";
    for (uint32_t i = trk.FirstClusterRef(); i < trk.FirstClusterRef() + trk.NClusters(); i++) {
      out << j << "/" << (i - trk.FirstClusterRef()) << ": " << mClusters[i].num << "/" << (int32_t)mClusters[i].state << ", ";
    }
    out << "\n";
  }
  uint32_t maxId = Param().rec.nonConsecutiveIDs ? mMemory->nOutputTrackClusters : mNMaxClusters;
  uint32_t j = 0;
  for (uint32_t i = 0; i < maxId; i++) {
    if ((mClusterAttachment[i] & attachFlagMask) != 0) {
      if (++j % 10 == 0) {
        out << "    Cluster attachment ";
      }
      out << i << ": " << (mClusterAttachment[i] & attachTrackMask) << " / " << (mClusterAttachment[i] & attachFlagMask) << " - ";
      if (j % 10 == 0) {
        out << "\n";
      }
    }
  }
  out << "\n";
}

void GPUTPCGMMerger::DumpRefit(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(2);
  out << "\nTPC Merger Refit\n";
  for (uint32_t i = 0; i < mMemory->nOutputTracks; i++) {
    const auto& trk = mOutputTracks[i];
    if (trk.NClusters() == 0) {
      continue;
    }
    const auto& p = trk.GetParam();
    const auto& po = trk.OuterParam();
    out << "  Track " << i << ": OK " << trk.OK() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " offset " << p.GetTZOffset() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << " / " << trk.NClustersFitted() << " Cov " << p.GetErr2Y() << "/" << p.GetErr2Z()
#ifdef GPUCA_HAVE_O2HEADERS
        << " dEdx " << (trk.OK() ? mOutputTracksdEdx[i].dEdxTotTPC : -1.f) << "/" << (trk.OK() ? mOutputTracksdEdx[i].dEdxMaxTPC : -1.f)
#endif
        << " Outer " << po.P[0] << "/" << po.P[1] << "/" << po.P[2] << "/" << po.P[3] << "/" << po.P[4] << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpFinal(std::ostream& out) const
{
  out << "\nTPC Merger Finalized\n";
  for (uint32_t j = 0; j < mMemory->nOutputTracks; j++) {
    const auto& trk = mOutputTracks[j];
    if (trk.NClusters() == 0) {
      continue;
    }
    out << "  Track " << j << ": ";
    for (uint32_t i = trk.FirstClusterRef(); i < trk.FirstClusterRef() + trk.NClusters(); i++) {
      if (mClusters[i].state != 0) {
        out << j << "/" << (i - trk.FirstClusterRef()) << ": " << mClusters[i].num << "/" << (int32_t)mClusters[i].state << ", ";
      }
    }
    out << "\n";
  }
  uint32_t maxId = Param().rec.nonConsecutiveIDs ? mMemory->nOutputTrackClusters : mNMaxClusters;
  uint32_t j = 0;
  for (uint32_t i = 0; i < maxId; i++) {
    if ((mClusterAttachment[i] & attachFlagMask) != 0) {
      if (++j % 10 == 0) {
        out << "    Cluster attachment ";
      }
      out << i << ": " << (mClusterAttachment[i] & attachTrackMask) << " / " << (mClusterAttachment[i] & attachFlagMask) << " - ";
      if (j % 10 == 0) {
        out << "\n";
      }
    }
  }
  out << "\n";
}

template <int32_t mergeType>
inline void GPUTPCGMMerger::MergedTrackStreamerInternal(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t slice1, int32_t slice2, int32_t mergeMode, float weight, float frac) const
{
#ifdef DEBUG_STREAMER
  std::vector<int32_t> hits1(152), hits2(152);
  for (int32_t i = 0; i < 152; i++) {
    hits1[i] = hits2[i] = -1;
  }
  const GPUTPCTracker& tracker1 = GetConstantMem()->tpcTrackers[slice1];
  const GPUTPCGMSliceTrack& sliceTrack1 = mSliceTrackInfos[b1.TrackID()];
  const GPUTPCTrack& inTrack1 = *sliceTrack1.OrigTrack();
  for (int32_t i = 0; i < inTrack1.NHits(); i++) {
    const GPUTPCHitId& ic1 = tracker1.TrackHits()[inTrack1.FirstHitID() + i];
    int32_t clusterIndex = tracker1.Data().ClusterDataIndex(tracker1.Data().Row(ic1.RowIndex()), ic1.HitIndex());
    hits1[ic1.RowIndex()] = clusterIndex;
  }
  const GPUTPCTracker& tracker2 = GetConstantMem()->tpcTrackers[slice2];
  const GPUTPCGMSliceTrack& sliceTrack2 = mSliceTrackInfos[b2.TrackID()];
  const GPUTPCTrack& inTrack2 = *sliceTrack2.OrigTrack();
  for (int32_t i = 0; i < inTrack2.NHits(); i++) {
    const GPUTPCHitId& ic2 = tracker2.TrackHits()[inTrack2.FirstHitID() + i];
    int32_t clusterIndex = tracker2.Data().ClusterDataIndex(tracker2.Data().Row(ic2.RowIndex()), ic2.HitIndex());
    hits2[ic2.RowIndex()] = clusterIndex;
  }

  std::string debugname = std::string("debug_") + name;
  std::string treename = std::string("tree_") + name;
  o2::utils::DebugStreamer::instance()->getStreamer(debugname.c_str(), "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName(treename.c_str()).data() << "slice1=" << slice1 << "slice2=" << slice2 << "b1=" << b1 << "b2=" << b2 << "clusters1=" << hits1 << "clusters2=" << hits2 << "sliceTrack1=" << sliceTrack1 << "sliceTrack2=" << sliceTrack2 << "mergeMode=" << mergeMode << "weight=" << weight << "fraction=" << frac << "\n";
#endif
}

void GPUTPCGMMerger::MergedTrackStreamer(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t slice1, int32_t slice2, int32_t mergeMode, float weight, float frac) const
{
#ifdef DEBUG_STREAMER
  if (mergeMode == 0) {
    MergedTrackStreamerInternal<0>(b1, b2, name, slice1, slice2, mergeMode, weight, frac);
  } else if (mergeMode >= 1 && mergeMode <= 0) {
    // MergedTrackStreamerInternal<1>(b1, b2, name, slice1, slice2, mergeMode, weight, frac); Not yet working
  }
#endif
}

const GPUTPCGMBorderTrack& GPUTPCGMMerger::MergedTrackStreamerFindBorderTrack(const GPUTPCGMBorderTrack* tracks, int32_t N, int32_t trackId) const
{
  for (int32_t i = 0; i < N; i++) {
    if (tracks[i].TrackID() == trackId) {
      return tracks[i];
    }
  }
  throw std::runtime_error("didn't find border track");
}

void GPUTPCGMMerger::DebugRefitMergedTrack(const GPUTPCGMMergedTrack& track) const
{
#ifdef GPUCA_HAVE_O2HEADERS
  GPUTPCGMMergedTrack trk = track;
  GPUTrackingRefit refit;
  ((GPUConstantMem*)GetConstantMem())->ioPtrs.mergedTrackHitStates = ClusterStateExt();
  ((GPUConstantMem*)GetConstantMem())->ioPtrs.mergedTrackHits = Clusters();
  refit.SetPtrsFromGPUConstantMem(GetConstantMem());
  int32_t retval = refit.RefitTrackAsGPU(trk, false, true);
  if (retval > 0) {
    GPUTPCGMPropagator prop;
    prop.SetMaterialTPC();
    prop.SetPolynomialField(&Param().polynomialField);
    prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
    prop.SetPropagateBzOnly(false);
    prop.SetMatLUT(Param().rec.useMatLUT ? GetConstantMem()->calibObjects.matLUT : nullptr);
    prop.SetTrack(&trk.Param(), trk.GetAlpha());
    int32_t err = prop.PropagateToXAlpha(track.GetParam().GetX(), track.GetAlpha(), false);
    if (err == 0) {
      printf("REFIT RESULT %d, SnpDiff %f\n", retval, trk.GetParam().GetSinPhi() - track.GetParam().GetSinPhi());
      if (retval > 20 && fabsf(trk.GetParam().GetSinPhi() - track.GetParam().GetSinPhi()) > 0.01f) {
        printf("LARGE DIFF\n");
      }
    } else {
      printf("PROPAGATE ERROR\n");
    }
  } else {
    printf("REFIT ERROR\n");
  }
#endif
}

std::vector<uint32_t> GPUTPCGMMerger::StreamerOccupancyBin(int32_t iSlice, int32_t iRow, float time) const
{
  static int32_t size = getenv("O2_DEBUG_STREAMER_OCCUPANCY_NBINS") ? atoi(getenv("O2_DEBUG_STREAMER_OCCUPANCY_NBINS")) : Param().rec.tpc.occupancyMapTimeBinsAverage;
  std::vector<uint32_t> retVal(1 + 2 * size);
#ifdef DEBUG_STREAMER
  const int32_t bin = CAMath::Max(0.f, time / Param().rec.tpc.occupancyMapTimeBins);
  for (int32_t i = 0; i < 1 + 2 * size; i++) {
    const int32_t mybin = bin + i - size;
    retVal[i] = (mybin >= 0 && mybin < (int32_t)GPUTPCClusterOccupancyMapBin::getNBins(Param())) ? Param().occupancyMap[mybin] : 0;
  }
#endif
  return retVal;
}

std::vector<float> GPUTPCGMMerger::StreamerUncorrectedZY(int32_t iSlice, int32_t iRow, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop) const
{
  std::vector<float> retVal(2);
#ifdef DEBUG_STREAMER
  GetConstantMem()->calibObjects.fastTransformHelper->InverseTransformYZtoNominalYZ(iSlice, iRow, track.GetY(), track.GetZ(), retVal[0], retVal[1]);
#endif
  return retVal;
}

void GPUTPCGMMerger::DebugStreamerUpdate(int32_t iTrk, int32_t ihit, float xx, float yy, float zz, const GPUTPCGMMergedTrackHit& cluster, const o2::tpc::ClusterNative& clusterNative, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop, const gputpcgmmergertypes::InterpolationErrorHit& interpolation, int8_t rejectChi2, bool refit, int32_t retVal, float avgInvCharge, float posY, float posZ, int16_t clusterState, int32_t retValReject, float err2Y, float err2Z) const
{
#ifdef DEBUG_STREAMER
  float time = clusterNative.getTime();
  auto occupancyBins = StreamerOccupancyBin(cluster.slice, cluster.row, time);
  auto uncorrectedYZ = StreamerUncorrectedZY(cluster.slice, cluster.row, track, prop);
  float invCharge = 1.f / clusterNative.qMax;
  int32_t iRow = cluster.row;
  float unscaledMult = (time >= 0.f ? Param().GetUnscaledMult(time) / Param().tpcGeometry.Row2X(iRow) : 0.f);
  const float clAlpha = Param().Alpha(cluster.slice);
  uint32_t occupancyTotal = Param().occupancyTotal;
  o2::utils::DebugStreamer::instance()->getStreamer("debug_update_track", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_update_track").data()
                                                                                    << "iTrk=" << iTrk
                                                                                    << "ihit=" << ihit
                                                                                    << "xx=" << xx
                                                                                    << "yy=" << yy
                                                                                    << "zz=" << zz
                                                                                    << "cluster=" << cluster
                                                                                    << "clusterNative=" << clusterNative
                                                                                    << "track=" << track
                                                                                    << "rejectChi2=" << rejectChi2
                                                                                    << "interpolationhit=" << interpolation
                                                                                    << "refit=" << refit
                                                                                    << "retVal=" << retVal
                                                                                    << "occupancyBins=" << occupancyBins
                                                                                    << "occupancyTotal=" << occupancyTotal
                                                                                    << "trackUncorrectedYZ=" << uncorrectedYZ
                                                                                    << "avgInvCharge=" << avgInvCharge
                                                                                    << "invCharge=" << invCharge
                                                                                    << "unscaledMultiplicity=" << unscaledMult
                                                                                    << "alpha=" << clAlpha
                                                                                    << "iRow=" << iRow
                                                                                    << "posY=" << posY
                                                                                    << "posZ=" << posZ
                                                                                    << "clusterState=" << clusterState
                                                                                    << "retValReject=" << retValReject
                                                                                    << "err2Y=" << err2Y
                                                                                    << "err2Z=" << err2Z
                                                                                    << "\n";
#endif
}
