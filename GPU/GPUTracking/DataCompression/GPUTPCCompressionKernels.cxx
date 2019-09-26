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
#include "GPUO2DataTypes.h"
#include "GPUTPCGMMerger.h"
#include "GPUParam.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUTPCGeometry.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUd() void GPUTPCCompressionKernels::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& processors)
{
  GPUTPCGMMerger& merger = processors.tpcMerger;
  const o2::tpc::ClusterNativeAccess* clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& compressor = processors.tpcCompressor;
  GPUParam& param = processors.param;

  char lastLeg = 0;
  int myTrack = 0;
  for (unsigned int i = get_global_id(0); i < (unsigned int)merger.NOutputTracks(); i += get_global_size(0)) {
    const GPUTPCGMMergedTrack& trk = merger.OutputTracks()[i];
    if (!trk.OK()) {
      continue;
    }
    bool rejectTrk = CAMath::Abs(trk.GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt;
    int nClustersStored = 0;
    CompressedClustersPtrsOnly& c = compressor.mPtrs;
    unsigned int lastRow = 0, lastSlice = 0; // BUG: These should be unsigned char, but then CUDA breaks
    GPUTPCCompressionTrackModel track;
    for (int k = trk.NClusters() - 1; k >= 0; k--) {
      const GPUTPCGMMergedTrackHit& hit = merger.Clusters()[trk.FirstClusterRef() + k];
      if (hit.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }

      int hitId = hit.num;
      int attach = merger.ClusterAttachment()[hitId];
      if ((attach & GPUTPCGMMerger::attachTrackMask) != i) {
        continue; // Main attachment to different track
      }
      bool rejectCluster = processors.param.rec.tpcRejectionMode && (rejectTrk || ((attach & GPUTPCGMMerger::attachGoodLeg) == 0) || (attach & GPUTPCGMMerger::attachHighIncl));
      if (rejectCluster) {
        compressor.mClusterStatus[hitId] = 1; // Cluster rejected, do not store
        continue;
      }

      if (!(param.rec.tpcCompressionModes & GPUSettings::CompressionTrackModel)) {
        continue; // No track model compression
      }
      const ClusterNative& orgCl = clusters->clusters[hit.slice][hit.row][hit.num - clusters->clusterOffset[hit.slice][hit.row]];
      float x = param.tpcGeometry.Row2X(hit.row);
      float y = param.tpcGeometry.LinearPad2Y(hit.slice, hit.row, orgCl.getPad());
      float z = param.tpcGeometry.LinearTime2Z(hit.slice, orgCl.getTime());
      if (nClustersStored) {
        if (lastLeg != hit.leg && track.Mirror()) {
          break;
        }
        if (track.Propagate(param.tpcGeometry.Row2X(hit.row), param.SliceParam[hit.slice].Alpha)) {
          break;
        }
      }

      compressor.mClusterStatus[hitId] = 1; // Cluster compressed in track model, do not store as difference

      int cidx = trk.FirstClusterRef() + nClustersStored++;
      if (nClustersStored == 1) {
        unsigned char qpt = fabs(trk.GetParam().GetQPt()) < 20.f ? (trk.GetParam().GetQPt() * (127.f / 20.f) + 127.5f) : (trk.GetParam().GetQPt() > 0 ? 254 : 0);
        track.Init(x, y, z, param.SliceParam[hit.slice].Alpha, qpt, param);

        myTrack = CAMath::AtomicAdd(&compressor.mMemory->nStoredTracks, 1);
        compressor.mAttachedClusterFirstIndex[myTrack] = trk.FirstClusterRef();
        lastLeg = hit.leg;
        c.qPtA[myTrack] = qpt;
        c.rowA[myTrack] = hit.row;
        c.sliceA[myTrack] = hit.slice;
        c.timeA[myTrack] = orgCl.getTimePacked();
        c.padA[myTrack] = orgCl.padPacked;
      } else {
        unsigned int row = hit.row;
        unsigned int slice = hit.slice;

        if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
          if (lastRow > row) {
            row += GPUCA_ROW_COUNT;
          }
          row -= lastRow;
          if (lastSlice > slice) {
            slice += compressor.NSLICES;
          }
          slice -= lastSlice;
        }
        c.rowDiffA[cidx] = row;
        c.sliceLegDiffA[cidx] = (hit.leg == lastLeg ? 0 : compressor.NSLICES) + slice;
        c.padResA[cidx] = orgCl.padPacked - orgCl.packPad(param.tpcGeometry.LinearY2Pad(hit.slice, hit.row, track.Y()));
        c.timeResA[cidx] = (orgCl.getTimePacked() - orgCl.packTime(param.tpcGeometry.LinearZ2Time(hit.slice, track.Z()))) & 0xFFFFFF;
        lastLeg = hit.leg;
      }
      lastRow = hit.row;
      lastSlice = hit.slice;
      unsigned short qtot = orgCl.qTot, qmax = orgCl.qMax;
      unsigned char sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
      if (param.rec.tpcCompressionModes & GPUSettings::CompressionTruncate) {
        compressor.truncateSignificantBitsCharge(qmax, param);
        compressor.truncateSignificantBitsCharge(qtot, param);
        compressor.truncateSignificantBitsWidth(sigmapad, param);
        compressor.truncateSignificantBitsWidth(sigmatime, param);
      }
      c.qTotA[cidx] = qtot;
      c.qMaxA[cidx] = qmax;
      c.sigmaPadA[cidx] = sigmapad;
      c.sigmaTimeA[cidx] = sigmatime;
      c.flagsA[cidx] = orgCl.getFlags();
      if (k && track.Filter(y, z, hit.row)) {
        break;
      }
    }
    if (nClustersStored) {
      CAMath::AtomicAdd(&compressor.mMemory->nStoredAttachedClusters, nClustersStored);
      c.nTrackClusters[myTrack] = nClustersStored;
    }
  }
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<0>::operator()(unsigned int a, unsigned int b) const
{
  return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<1>::operator()(unsigned int a, unsigned int b) const
{
  return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<2>::operator()(unsigned int a, unsigned int b) const
{
  if (mClsPtr[a].getTimePacked() >> 3 == mClsPtr[b].getTimePacked() >> 3) {
    return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
  }
  return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<3>::operator()(unsigned int a, unsigned int b) const
{
  if (mClsPtr[a].padPacked >> 3 == mClsPtr[b].padPacked >> 3) {
    return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
  }
  return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
}

template <>
GPUd() void GPUTPCCompressionKernels::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUTPCSharedMemory& smem, processorType& processors)
{
  GPUTPCGMMerger& merger = processors.tpcMerger;
  const o2::tpc::ClusterNativeAccess* clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& compressor = processors.tpcCompressor;
  GPUParam& param = processors.param;
  unsigned int* sortBuffer = compressor.mClusterSortBuffer + iBlock * compressor.mNMaxClusterSliceRow;
  for (int iSliceRow = iBlock; iSliceRow < GPUCA_NSLICES * GPUCA_ROW_COUNT; iSliceRow += nBlocks) {
    const int iSlice = iSliceRow / GPUCA_ROW_COUNT;
    const int iRow = iSliceRow % GPUCA_ROW_COUNT;
    const int idOffset = clusters->clusterOffset[iSlice][iRow];
    if (iThread == 0) {
      smem.nCount = 0;
    }
    GPUbarrier();

    CompressedClustersPtrsOnly& c = compressor.mPtrs;
    for (unsigned int i = get_local_id(0); i < clusters->nClusters[iSlice][iRow]; i += get_local_size(0)) {
      const int idx = idOffset + i;
      if (compressor.mClusterStatus[idx]) {
        continue;
      }
      int attach = merger.ClusterAttachment()[idx];

      bool unattached = attach == 0;
      if (unattached) {
        if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyB) {
          continue;
        }
      } else if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyA) {
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
      int cidx = CAMath::AtomicAddShared(&smem.nCount, 1);
      sortBuffer[cidx] = i;
    }
    GPUbarrier();

    if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
      if (param.rec.tpcCompressionSortOrder == GPUSettings::SortZPadTime) {
        CAAlgo::sortInBlock(sortBuffer, sortBuffer + smem.nCount, GPUTPCCompressionKernels_Compare<GPUSettings::SortZPadTime>(clusters->clusters[iSlice][iRow]));
      } else if (param.rec.tpcCompressionSortOrder == GPUSettings::SortZTimePad) {
        CAAlgo::sortInBlock(sortBuffer, sortBuffer + smem.nCount, GPUTPCCompressionKernels_Compare<GPUSettings::SortZTimePad>(clusters->clusters[iSlice][iRow]));
      } else if (param.rec.tpcCompressionSortOrder == GPUSettings::SortPad) {
        CAAlgo::sortInBlock(sortBuffer, sortBuffer + smem.nCount, GPUTPCCompressionKernels_Compare<GPUSettings::SortPad>(clusters->clusters[iSlice][iRow]));
      } else {
        CAAlgo::sortInBlock(sortBuffer, sortBuffer + smem.nCount, GPUTPCCompressionKernels_Compare<GPUSettings::SortTime>(clusters->clusters[iSlice][iRow]));
      }
      GPUbarrier();
    }

    unsigned int lastTime = 0;
    unsigned short lastPad = 0;
    for (unsigned int i = 0; i < smem.nCount; i++) {
      int cidx = idOffset + i;
      const ClusterNative& orgCl = clusters->clusters[iSlice][iRow][sortBuffer[i]];
      c.padDiffU[cidx] = orgCl.padPacked - lastPad;
      c.timeDiffU[cidx] = (orgCl.getTimePacked() - lastTime) & 0xFFFFFF;
      if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
        lastPad = orgCl.padPacked;
        lastTime = orgCl.getTimePacked();
      }

      unsigned short qtot = orgCl.qTot, qmax = orgCl.qMax;
      unsigned char sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
      if (param.rec.tpcCompressionModes & GPUSettings::CompressionTruncate) {
        compressor.truncateSignificantBitsCharge(qmax, param);
        compressor.truncateSignificantBitsCharge(qtot, param);
        compressor.truncateSignificantBitsWidth(sigmapad, param);
        compressor.truncateSignificantBitsWidth(sigmatime, param);
      }
      c.qTotU[cidx] = qtot;
      c.qMaxU[cidx] = qmax;
      c.sigmaPadU[cidx] = sigmapad;
      c.sigmaTimeU[cidx] = sigmatime;
      c.flagsU[cidx] = orgCl.getFlags();
    }
    if (iThread == 0) {
      c.nSliceRowClusters[iSlice * GPUCA_ROW_COUNT + iRow] = smem.nCount;
      CAMath::AtomicAdd(&compressor.mMemory->nStoredUnattachedClusters, smem.nCount);
    }
    GPUbarrier();
  }
}
