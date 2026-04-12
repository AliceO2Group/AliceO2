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
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMerger.h"
#include "GPUO2DataTypes.h"
#include "GPUCommonMath.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUParam.inc"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSectorTrack.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUReconstruction.h"
#include "GPUDebugStreamer.h"
#include "GPUTPCClusterOccupancyMap.h"
#include "GPUTrackingRefit.h"
#include "GPUConstantMem.h"
#include "TPCFastTransformPOD.h"

using namespace o2::gpu;
using namespace gputpcgmmergertypes;

void GPUTPCGMMerger::DumpSectorTracks(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(6);
  out << "\nTPC Merger Sector Tracks\n";
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    out << "Sector Track Info Sector " << iSector << " Index " << (mSectorTrackInfoIndex[iSector + 1] - mSectorTrackInfoIndex[iSector]) << " / " << (mSectorTrackInfoIndex[NSECTORS + iSector + 1] - mSectorTrackInfoIndex[NSECTORS + iSector]) << "\n";
    for (int32_t iGlobal = 0; iGlobal < 2; iGlobal++) {
      out << "  Track type " << iGlobal << "\n";
      for (int32_t j = mSectorTrackInfoIndex[iSector + NSECTORS * iGlobal]; j < mSectorTrackInfoIndex[iSector + NSECTORS * iGlobal + 1]; j++) {
        const auto& trk = mSectorTrackInfos[j];
        out << "    Track " << j << ": LocalId " << (iGlobal ? (trk.LocalTrackId() >> 24) : -1) << "/" << (iGlobal ? (trk.LocalTrackId() & 0xFFFFFF) : -1) << " NCl " << trk.NClusters() << " X " << trk.X() << " offsetz " << trk.TOffset() << " A " << trk.Alpha() << " Y " << trk.Y() << " Z " << trk.Z() << " SinPhi " << trk.SinPhi() << " CosPhi " << trk.CosPhi() << " SecPhi " << trk.SecPhi() << " Tgl " << trk.DzDs() << " QPt " << trk.QPt() << "\n";
      }
    }
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergeRanges(std::ostream& out, uint8_t mergeMode) const
{
  int32_t n = (mergeMode & mergeModes::mergeAcrossCE) ? NSECTORS / 2 : NSECTORS;
  for (int32_t i = 0; i < n; i++) {
    int32_t n1, n2;
    GPUTPCGMBorderTrack *b1, *b2;
    int32_t jSector;
    MergeBorderTracksSetup(n1, n2, b1, b2, jSector, i, mergeMode);
    const int32_t nTrk = *mRec->GetConstantMem().tpcTrackers[jSector].NTracks();
    const gputpcgmmergertypes::GPUTPCGMBorderRange* range1 = BorderRange(i);
    const gputpcgmmergertypes::GPUTPCGMBorderRange* range2 = BorderRange(jSector) + nTrk;
    out << "\nBorder Tracks : i " << i << " mergeMode " << (uint32_t)mergeMode << "\n";
    for (int32_t k = 0; k < n1; k++) {
      out << "  " << k << ": t " << b1[k].TrackID() << " ncl " << b1[k].NClusters() << " row " << ((mergeMode & mergeModes::mergeAcrossCE) ? b1[k].Row() : -1) << " par " << b1[k].Par()[0] << " " << b1[k].Par()[1] << " " << b1[k].Par()[2] << " " << b1[k].Par()[3] << " " << b1[k].Par()[4]
          << " offset " << b1[k].ZOffsetLinear() << " cov " << b1[k].Cov()[0] << " " << b1[k].Cov()[1] << " " << b1[k].Cov()[2] << " " << b1[k].Cov()[3] << " " << b1[k].Cov()[4] << " covd " << b1[k].CovD()[0] << " " << b1[k].CovD()[1] << "\n";
    }
    if (i != jSector) {
      for (int32_t k = 0; k < n2; k++) {
        out << "  " << k << ": t " << b2[k].TrackID() << " ncl " << b2[k].NClusters() << " row " << ((mergeMode & mergeModes::mergeAcrossCE) ? b2[k].Row() : -1) << " par " << b2[k].Par()[0] << " " << b2[k].Par()[1] << " " << b2[k].Par()[2] << " " << b2[k].Par()[3] << " " << b2[k].Par()[4]
            << " offset " << b2[k].ZOffsetLinear() << " cov " << b2[k].Cov()[0] << " " << b2[k].Cov()[1] << " " << b2[k].Cov()[2] << " " << b2[k].Cov()[3] << " " << b2[k].Cov()[4] << " covd " << b2[k].CovD()[0] << " " << b2[k].CovD()[1] << "\n";
      }
    }
    out << "\nBorder Range : i " << i << " mergeMode " << (uint32_t)mergeMode << "\n";
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
  const int32_t n = output ? mMemory->nMergedTracks : SectorTrackInfoLocalTotal();
  for (int32_t i = 0; i < n; i++) {
    if (mTrackLinks[i] != -1) {
      out << "  " << i << ": " << mTrackLinks[i] << "\n";
    }
  }
}

void GPUTPCGMMerger::DumpMergedWithinSectors(std::ostream& out) const
{
  DumpTrackLinks(out, false, "within Sectors");
  out << "\nTPC Merger Merge Within Sectors\n";
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    for (int32_t j = mSectorTrackInfoIndex[iSector]; j < mSectorTrackInfoIndex[iSector + 1]; j++) {
      const auto& trk = mSectorTrackInfos[j];
      if (trk.NextSegmentNeighbour() >= 0 || trk.PrevSegmentNeighbour() >= 0) {
        out << "  Track " << j << ": Neighbour " << trk.PrevSegmentNeighbour() << " / " << trk.NextSegmentNeighbour() << "\n";
      }
    }
  }
}

void GPUTPCGMMerger::DumpMergedBetweenSectors(std::ostream& out) const
{
  DumpTrackLinks(out, false, "between Sectors");
  out << "\nTPC Merger Merge Between Sectors\n";
  for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
    for (int32_t j = mSectorTrackInfoIndex[iSector]; j < mSectorTrackInfoIndex[iSector + 1]; j++) {
      const auto& trk = mSectorTrackInfos[j];
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
  out << "\nTPC Merger Collected Tracks\n";
  DumpTrackParam(out);
}

void GPUTPCGMMerger::DumpTrackParam(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(6);
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    const auto& trk = mMergedTracks[i];
    const auto& p = trk.GetParam();
    out << "  Track " << i << ": Loop " << trk.Looper() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " offset " << p.GetTOffset() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpMergeCE(std::ostream& out) const
{
  DumpTrackLinks(out, true, " for CE merging");
  out << "\nTPC Merger Merge CE\n";
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    const auto& trk = mMergedTracks[i];
    if (trk.CCE()) {
      out << "  Track " << i << ": CCE\n";
    }
  }
}

void GPUTPCGMMerger::DumpTrackClusters(std::ostream& out, bool non0StateOnly, bool noNDF0) const
{
  for (uint32_t j = 0; j < mMemory->nMergedTracks; j++) {
    const auto& trk = mMergedTracks[j];
    if (trk.NClusters() == 0) {
      continue;
    }
    if (noNDF0 && (!trk.OK() || trk.GetParam().GetNDF() < 0)) {
      continue;
    }
    out << "  Track " << j << ": (" << trk.NClusters() << "): ";
    for (uint32_t i = trk.FirstClusterRef(); i < trk.FirstClusterRef() + trk.NClusters(); i++) {
      if (!non0StateOnly || mClusters[i].state != 0) {
        out << j << "/" << (i - trk.FirstClusterRef()) << ": " << (int32_t)mClusters[i].row << "/" << mClusters[i].num << "/" << (int32_t)mClusters[i].state << ", ";
      }
    }
    out << "\n";
  }
}

void GPUTPCGMMerger::DumpFitPrepare(std::ostream& out) const
{
  out << "\nTPC Merger Refit Prepare\n";
  out << "  Sort\n";
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    out << "    " << i << ": " << mTrackOrderAttach[i] << "\n";
  }
  out << "  Track Clusters";
  DumpTrackClusters(out);
  uint32_t j = 0;
  for (uint32_t i = 0; i < mNMaxClusters; i++) {
    if ((mClusterAttachment[i] & attachFlagMask) != 0) {
      if (j++ % 10 == 0) {
        out << "\n    Cluster attachment ";
      }
      out << i << ": " << (mClusterAttachment[i] & attachTrackMask) << " / " << (mClusterAttachment[i] & attachFlagMask) << " - ";
    }
  }
  out << "\n";
}

void GPUTPCGMMerger::DumpRefit(std::ostream& out) const
{
  std::streamsize ss = out.precision();
  out << std::setprecision(6);
  out << "\nTPC Merger Refit\n";
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    const auto& trk = mMergedTracks[i];
    if (trk.NClusters() == 0) {
      continue;
    }
    const auto& p = trk.GetParam();
    const auto& po = trk.OuterParam();
    out << "  Track " << i << ": OK " << trk.OK() << " Alpha " << trk.GetAlpha() << " X " << p.GetX() << " offset " << p.GetTOffset() << " Y " << p.GetY() << " Z " << p.GetZ() << " SPhi " << p.GetSinPhi() << " Tgl " << p.GetDzDs() << " QPt " << p.GetQPt() << " NCl " << trk.NClusters() << " / " << trk.NClustersFitted() << " Cov " << p.GetErr2Y() << "/" << p.GetErr2Z()
        << " dEdx " << (trk.OK() && Param().dodEdxEnabled ? mMergedTracksdEdx[i].dEdxTotTPC : -1.f) << "/" << (trk.OK() && Param().dodEdxEnabled ? mMergedTracksdEdx[i].dEdxMaxTPC : -1.f)
        << " Outer " << po.P[0] << "/" << po.P[1] << "/" << po.P[2] << "/" << po.P[3] << "/" << po.P[4]
        << " NFitted " << trk.NClustersFitted() << " flags " << (int)trk.Flags() << "\n";
  }
  out << std::setprecision(ss);
}

void GPUTPCGMMerger::DumpLoopers(std::ostream& out) const
{
  out << "\n TPC Merger Looper Afterburner\n";
  for (uint32_t i = 0; i < mMemory->nMergedTracks; i++) {
    if (i && i % 100 == 0) {
      out << "\n";
    }
    out << (int)mMergedTracks[i].MergedLooperUnconnected() << " ";
  }
  out << "\n";
}

void GPUTPCGMMerger::DumpFinal(std::ostream& out) const
{
  out << "\nTPC Merger Finalized\n";
  out << "Track Clusters\n";
  DumpTrackClusters(out, true);
  uint32_t j = 0;
  for (uint32_t i = 0; i < mNMaxClusters; i++) {
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
inline void GPUTPCGMMerger::MergedTrackStreamerInternal(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t sector1, int32_t sector2, uint8_t mergeMode, float weight, float frac) const
{
#ifdef DEBUG_STREAMER
  std::vector<int32_t> hits1(GPUTPCGeometry::NROWS), hits2(GPUTPCGeometry::NROWS);
  for (int32_t i = 0; i < GPUTPCGeometry::NROWS; i++) {
    hits1[i] = hits2[i] = -1;
  }
  const GPUTPCTracker& tracker1 = GetConstantMem()->tpcTrackers[sector1];
  const GPUTPCGMSectorTrack& sectorTrack1 = mSectorTrackInfos[b1.TrackID()];
  const GPUTPCTrack& inTrack1 = *sectorTrack1.OrigTrack();
  for (int32_t i = 0; i < inTrack1.NHits(); i++) {
    const GPUTPCHitId& ic1 = tracker1.TrackHits()[inTrack1.FirstHitID() + i];
    int32_t clusterIndex = tracker1.Data().ClusterDataIndex(tracker1.Data().Row(ic1.RowIndex()), ic1.HitIndex());
    hits1[ic1.RowIndex()] = clusterIndex;
  }
  const GPUTPCTracker& tracker2 = GetConstantMem()->tpcTrackers[sector2];
  const GPUTPCGMSectorTrack& sectorTrack2 = mSectorTrackInfos[b2.TrackID()];
  const GPUTPCTrack& inTrack2 = *sectorTrack2.OrigTrack();
  for (int32_t i = 0; i < inTrack2.NHits(); i++) {
    const GPUTPCHitId& ic2 = tracker2.TrackHits()[inTrack2.FirstHitID() + i];
    int32_t clusterIndex = tracker2.Data().ClusterDataIndex(tracker2.Data().Row(ic2.RowIndex()), ic2.HitIndex());
    hits2[ic2.RowIndex()] = clusterIndex;
  }

  std::string debugname = std::string("debug_") + name;
  std::string treename = std::string("tree_") + name;
  o2::utils::DebugStreamer::instance()->getStreamer(debugname.c_str(), "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName(treename.c_str()).data() << "sector1=" << sector1 << "sector2=" << sector2 << "b1=" << b1 << "b2=" << b2 << "clusters1=" << hits1 << "clusters2=" << hits2 << "sectorTrack1=" << sectorTrack1 << "sectorTrack2=" << sectorTrack2 << "mergeMode=" << mergeMode << "weight=" << weight << "fraction=" << frac << "\n";
#endif
}

void GPUTPCGMMerger::MergedTrackStreamer(const GPUTPCGMBorderTrack& b1, const GPUTPCGMBorderTrack& b2, const char* name, int32_t sector1, int32_t sector2, uint8_t mergeMode, float weight, float frac) const
{
#ifdef DEBUG_STREAMER
  if (!(mergeMode & mergeModes::mergeAcrossCE0)) {
    MergedTrackStreamerInternal<0>(b1, b2, name, sector1, sector2, mergeMode, weight, frac);
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
    prop.SetMaxSinPhi(constants::MAX_SIN_PHI);
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
}

std::vector<uint32_t> GPUTPCGMMerger::StreamerOccupancyBin(int32_t iSector, int32_t iRow, float time) const
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

std::vector<float> GPUTPCGMMerger::StreamerUncorrectedZY(int32_t iSector, int32_t iRow, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop) const
{
  std::vector<float> retVal(2);
#ifdef DEBUG_STREAMER
  GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(iSector, iRow, track.GetY(), track.GetZ(), retVal[0], retVal[1]);
#endif
  return retVal;
}

void GPUTPCGMMerger::DebugStreamerUpdate(int32_t iTrk, int32_t ihit, float xx, float yy, float zz, const GPUTPCGMMergedTrackHit& cluster, const o2::tpc::ClusterNative& clusterNative, const GPUTPCGMTrackParam& track, const GPUTPCGMPropagator& prop, const gputpcgmmergertypes::InterpolationErrorHit& interpolation, int8_t rejectChi2, bool refit, int32_t retVal, float avgInvCharge, float posY, float posZ, int16_t clusterState, int32_t retValReject, float err2Y, float err2Z) const
{
#ifdef DEBUG_STREAMER
  float time = clusterNative.getTime();
  auto occupancyBins = StreamerOccupancyBin(cluster.sector, cluster.row, time);
  auto uncorrectedYZ = StreamerUncorrectedZY(cluster.sector, cluster.row, track, prop);
  float invCharge = 1.f / clusterNative.qMax;
  int32_t iRow = cluster.row;
  float unscaledMult = (time >= 0.f ? Param().GetUnscaledMult(time) / GPUTPCGeometry::Row2X(iRow) : 0.f);
  const float clAlpha = Param().Alpha(cluster.sector);
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
