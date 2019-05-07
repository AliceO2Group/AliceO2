// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompressionKernels.cxx
/// \author David Rohr

#include "GPUTPCCompressionKernels.h"
#include "GPUConstantMem.h"
#include "ClusterNativeAccessExt.h"
#include "GPUTPCGMMerger.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::TPC;

#if defined(GPUCA_BUILD_TPCCOMPRESSION) && !defined(GPUCA_ALIROOT_LIB)
template <>
GPUd() void GPUTPCCompressionKernels::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& processors)
{
  GPUTPCGMMerger& merger = processors.tpcMerger;
  const ClusterNativeAccessExt* clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& compressor = processors.tpcCompressor;

  char lastLeg = 0;
  int myTrack = 0;
  for (unsigned int i = get_global_id(0); i < (unsigned int)merger.NOutputTracks(); i += get_global_size(0)) {
    const GPUTPCGMMergedTrack& trk = merger.OutputTracks()[i];
    if (!trk.OK()) {
      continue;
    }
    bool rejectTrk = CAMath::Abs(trk.GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt;
    int nClustersStored = 0;
    CompressedClusters& c = compressor.mPtrs;
    for (unsigned int k = 0; k < trk.NClusters(); k++) {
      const GPUTPCGMMergedTrackHit& hit = merger.Clusters()[trk.FirstClusterRef() + k];
      if (hit.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }

      int hitId = hit.num;
      int attach = merger.ClusterAttachment()[hitId];
      if ((attach & GPUTPCGMMerger::attachTrackMask) != i) {
        continue; // Main attachment to different track
      }
      compressor.mClusterStatus[hitId] = 1;

      if (processors.param.rec.tpcRejectionMode) {
        if (rejectTrk) {
          continue;
        }
        if ((attach & GPUTPCGMMerger::attachGoodLeg) == 0) {
          continue;
        }
        if (attach & GPUTPCGMMerger::attachHighIncl) {
          continue;
        }
      }
      const ClusterNative& orgCl = clusters->clusters[hit.slice][hit.row][hit.num - clusters->clusterOffset[hit.slice][hit.row]];
      int cidx = trk.FirstClusterRef() + nClustersStored++;
      if (nClustersStored == 1) {
        myTrack = CAMath::AtomicAdd(&compressor.mMemory->nStoredTracks, 1);
        compressor.mAttachedClusterFirstIndex[myTrack] = trk.FirstClusterRef();
        lastLeg = hit.leg;
        c.qPtA[myTrack] = fabs(trk.GetParam().GetQPt()) < 127.f / 20.f ? (trk.GetParam().GetQPt() * (20.f / 127.f)) : (trk.GetParam().GetQPt() > 0 ? 127 : -127);
        c.rowA[myTrack] = hit.row;
        c.sliceA[myTrack] = hit.slice;
        c.timeA[myTrack] = orgCl.getTimePacked();
        c.padA[myTrack] = orgCl.padPacked;
      } else {
        c.rowDiffA[cidx] = hit.row;
        c.sliceLegDiffA[cidx] = (hit.leg == lastLeg ? 0 : compressor.NSLICES) + hit.slice;
        c.padResA[cidx] = orgCl.padPacked;
        c.timeResA[cidx] = orgCl.getTimePacked();
      }
      c.qTotA[cidx] = orgCl.qTot;
      c.qMaxA[cidx] = orgCl.qMax;
      c.flagsA[cidx] = orgCl.getFlags();
      c.sigmaPadA[cidx] = orgCl.sigmaPadPacked;
      c.sigmaTimeA[cidx] = orgCl.sigmaTimePacked;
    }
    if (nClustersStored) {
      CAMath::AtomicAdd(&compressor.mMemory->nStoredAttachedClusters, nClustersStored);
      c.nTrackClusters[myTrack] = nClustersStored;
    }
  }
}

template <>
GPUd() void GPUTPCCompressionKernels::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& processors)
{
  GPUTPCGMMerger& merger = processors.tpcMerger;
  const ClusterNativeAccessExt* clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& compressor = processors.tpcCompressor;
  const int iSlice = iBlock / GPUCA_ROW_COUNT;
  const int iRow = iBlock % GPUCA_ROW_COUNT;
  const int idOffset = clusters->clusterOffset[iSlice][iRow];
  GPUshared() unsigned int nCount;
  if (iThread == 0) {
    nCount = 0;
  }
  GPUbarrier();

  CompressedClusters& c = compressor.mPtrs;
  for (unsigned int i = get_local_id(0); i < clusters->nClusters[iSlice][iRow]; i += get_local_size(0)) {
    const int idx = idOffset + i;
    if (compressor.mClusterStatus[idx]) {
      continue;
    }
    int attach = merger.ClusterAttachment()[idx];

    bool unattached = attach == 0;
    if (unattached) {
      if (processors.param.rec.tpcRejectionMode >= 2) {
        continue;
      }
    } else if (processors.param.rec.tpcRejectionMode) {
      if ((attach & GPUTPCGMMerger::attachGoodLeg) == 0) {
        continue;
      }
      if (attach & GPUTPCGMMerger::attachHighIncl) {
        continue;
      }
      int id = attach & GPUTPCGMMerger::attachTrackMask;
      if (CAMath::Abs(merger.OutputTracks()[id].GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt) {
        continue;
      }
    }

    const ClusterNative& orgCl = clusters->clusters[iSlice][iRow][i];
    int cidx = idOffset + CAMath::AtomicAddShared(&nCount, 1);
    c.qTotU[cidx] = orgCl.qTot;
    c.qMaxU[cidx] = orgCl.qMax;
    c.flagsU[cidx] = orgCl.getFlags();
    c.padDiffU[cidx] = orgCl.padPacked;
    c.timeDiffU[cidx] = orgCl.getTimePacked();
    c.sigmaPadU[cidx] = orgCl.sigmaPadPacked;
    c.sigmaTimeU[cidx] = orgCl.sigmaTimePacked;
  }
  GPUbarrier();
  if (iThread == 0) {
    c.nSliceRowClusters[iSlice * GPUCA_ROW_COUNT + iRow] = nCount;
    CAMath::AtomicAdd(&compressor.mMemory->nStoredUnattachedClusters, nCount);
  }
}

#endif
