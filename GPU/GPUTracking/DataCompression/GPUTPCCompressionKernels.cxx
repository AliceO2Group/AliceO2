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
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step0attached>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const GPUTPCGMMerger& GPUrestrict() merger = processors.tpcMerger;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const GPUParam& GPUrestrict() param = processors.param;

  char lastLeg = 0;
  int myTrack = 0;
  for (unsigned int i = get_global_id(0); i < (unsigned int)merger.NOutputTracks(); i += get_global_size(0)) {
    const GPUTPCGMMergedTrack& GPUrestrict() trk = merger.OutputTracks()[i];
    if (!trk.OK()) {
      continue;
    }
    bool rejectTrk = CAMath::Abs(trk.GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt;
    int nClustersStored = 0;
    CompressedClustersPtrsOnly& GPUrestrict() c = compressor.mPtrs;
    unsigned int lastRow = 0, lastSlice = 0; // BUG: These should be unsigned char, but then CUDA breaks
    GPUTPCCompressionTrackModel track;
    for (int k = trk.NClusters() - 1; k >= 0; k--) {
      const GPUTPCGMMergedTrackHit& GPUrestrict() hit = merger.Clusters()[trk.FirstClusterRef() + k];
      if (hit.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }

      int hitId = hit.num;
      int attach = merger.ClusterAttachment()[hitId];
      if ((attach & GPUTPCGMMergerTypes::attachTrackMask) != i) {
        continue; // Main attachment to different track
      }
      bool rejectCluster = processors.param.rec.tpcRejectionMode && (rejectTrk || ((attach & GPUTPCGMMergerTypes::attachGoodLeg) == 0) || (attach & GPUTPCGMMergerTypes::attachHighIncl));
      if (rejectCluster) {
        compressor.mClusterStatus[hitId] = 1; // Cluster rejected, do not store
        continue;
      }

      if (!(param.rec.tpcCompressionModes & GPUSettings::CompressionTrackModel)) {
        continue; // No track model compression
      }
      const ClusterNative& GPUrestrict() orgCl = clusters->clusters[hit.slice][hit.row][hit.num - clusters->clusterOffset[hit.slice][hit.row]];
      float x = param.tpcGeometry.Row2X(hit.row);
      float y = param.tpcGeometry.LinearPad2Y(hit.slice, hit.row, orgCl.getPad());
      float z = param.tpcGeometry.LinearTime2Z(hit.slice, orgCl.getTime());
      if (nClustersStored) {
        if ((hit.slice < GPUCA_NSLICES) ^ (lastSlice < GPUCA_NSLICES)) {
          break;
        }
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
        track.Init(x, y, z, param.SliceParam[hit.slice].Alpha, qpt, param); // TODO: Compression track model must respect Z offset!

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
        float pad = CAMath::Max(0.f, CAMath::Min((float)param.tpcGeometry.NPads(GPUCA_ROW_COUNT - 1), param.tpcGeometry.LinearY2Pad(hit.slice, hit.row, track.Y())));
        c.padResA[cidx] = orgCl.padPacked - orgCl.packPad(pad);
        float time = CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(hit.slice, track.Z()));
        c.timeResA[cidx] = (orgCl.getTimePacked() - orgCl.packTime(time)) & 0xFFFFFF;
        lastLeg = hit.leg;
      }
      lastRow = hit.row;
      lastSlice = hit.slice;
      unsigned short qtot = orgCl.qTot, qmax = orgCl.qMax;
      unsigned char sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
      if (param.rec.tpcCompressionModes & GPUSettings::CompressionTruncate) {
        compressor.truncateSignificantBitsChargeMax(qmax, param);
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
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step1unattached>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  const GPUTPCGMMerger& GPUrestrict() merger = processors.tpcMerger;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.tpcConverter.getClustersNative();
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  GPUParam& GPUrestrict() param = processors.param;
  unsigned int* sortBuffer = smem.sortBuffer;
  for (int iSliceRow = iBlock; iSliceRow < GPUCA_NSLICES * GPUCA_ROW_COUNT; iSliceRow += nBlocks) {
    const int iSlice = iSliceRow / GPUCA_ROW_COUNT;
    const int iRow = iSliceRow % GPUCA_ROW_COUNT;
    const int idOffset = clusters->clusterOffset[iSlice][iRow];
    if (iThread == nThreads - 1) {
      smem.nCount = 0;
    }
    unsigned int totalCount = 0;
    GPUbarrier();

    CompressedClustersPtrsOnly& GPUrestrict() c = compressor.mPtrs;

    const unsigned int nn = GPUCommonMath::nextMultipleOf<GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached)>(clusters->nClusters[iSlice][iRow]);
    for (unsigned int i = iThread; i < nn + nThreads; i += nThreads) {
      const int idx = idOffset + i;
      int cidx = 0;
      do {
        if (i >= clusters->nClusters[iSlice][iRow]) {
          break;
        }
        if (compressor.mClusterStatus[idx]) {
          break;
        }
        int attach = merger.ClusterAttachment()[idx];
        bool unattached = attach == 0;

        if (unattached) {
          if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyB) {
            break;
          }
        } else if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyA) {
          if ((attach & GPUTPCGMMergerTypes::attachGoodLeg) == 0) {
            break;
          }
          if (attach & GPUTPCGMMergerTypes::attachHighIncl) {
            break;
          }
          int id = attach & GPUTPCGMMergerTypes::attachTrackMask;
          if (CAMath::Abs(merger.OutputTracks()[id].GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt) {
            break;
          }
        }
        cidx = 1;
      } while (false);

      GPUbarrier();
      int myIndex = work_group_scan_inclusive_add(cidx);
      int storeLater = -1;
      if (cidx) {
        if (smem.nCount + myIndex <= GPUCA_TPC_COMP_CHUNK_SIZE) {
          sortBuffer[smem.nCount + myIndex - 1] = i;
        } else {
          storeLater = smem.nCount + myIndex - 1 - GPUCA_TPC_COMP_CHUNK_SIZE;
        }
      }
      GPUbarrier();
      if (iThread == nThreads - 1) {
        smem.nCount += myIndex;
      }
      GPUbarrier();

      if (smem.nCount < GPUCA_TPC_COMP_CHUNK_SIZE && i < nn) {
        continue;
      }

      const unsigned int count = CAMath::Min(smem.nCount, (unsigned int)GPUCA_TPC_COMP_CHUNK_SIZE);
      if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
        if (param.rec.tpcCompressionSortOrder == GPUSettings::SortZPadTime) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortZPadTime>(clusters->clusters[iSlice][iRow]));
        } else if (param.rec.tpcCompressionSortOrder == GPUSettings::SortZTimePad) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortZTimePad>(clusters->clusters[iSlice][iRow]));
        } else if (param.rec.tpcCompressionSortOrder == GPUSettings::SortPad) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortPad>(clusters->clusters[iSlice][iRow]));
        } else if (param.rec.tpcCompressionSortOrder == GPUSettings::SortTime) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortTime>(clusters->clusters[iSlice][iRow]));
        }
        GPUbarrier();
      }

      for (unsigned int j = get_local_id(0); j < count; j += get_local_size(0)) {
        int outidx = idOffset + totalCount + j;
        const ClusterNative& GPUrestrict() orgCl = clusters->clusters[iSlice][iRow][sortBuffer[j]];
        unsigned int lastTime = 0;
        unsigned int lastPad = 0;
        if (j != 0) {
          const ClusterNative& GPUrestrict() orgClPre = clusters->clusters[iSlice][iRow][sortBuffer[j - 1]];
          lastPad = orgClPre.padPacked;
          lastTime = orgClPre.getTimePacked();
        } else if (totalCount != 0) {
          const ClusterNative& GPUrestrict() orgClPre = clusters->clusters[iSlice][iRow][smem.lastIndex];
          lastPad = orgClPre.padPacked;
          lastTime = orgClPre.getTimePacked();
        }

        c.padDiffU[outidx] = orgCl.padPacked - lastPad;
        c.timeDiffU[outidx] = (orgCl.getTimePacked() - lastTime) & 0xFFFFFF;

        unsigned short qtot = orgCl.qTot, qmax = orgCl.qMax;
        unsigned char sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
        if (param.rec.tpcCompressionModes & GPUSettings::CompressionTruncate) {
          compressor.truncateSignificantBitsChargeMax(qmax, param);
          compressor.truncateSignificantBitsCharge(qtot, param);
          compressor.truncateSignificantBitsWidth(sigmapad, param);
          compressor.truncateSignificantBitsWidth(sigmatime, param);
        }
        c.qTotU[outidx] = qtot;
        c.qMaxU[outidx] = qmax;
        c.sigmaPadU[outidx] = sigmapad;
        c.sigmaTimeU[outidx] = sigmatime;
        c.flagsU[outidx] = orgCl.getFlags();
      }

      GPUbarrier();
      if (storeLater > 0) {
        sortBuffer[storeLater] = i;
      }
      totalCount += count;
      if (iThread == nThreads - 1 && count) {
        smem.lastIndex = sortBuffer[count - 1];
        smem.nCount -= count;
      }
    }

    if (iThread == nThreads - 1) {
      c.nSliceRowClusters[iSlice * GPUCA_ROW_COUNT + iRow] = totalCount;
      CAMath::AtomicAdd(&compressor.mMemory->nStoredUnattachedClusters, totalCount);
    }
    GPUbarrier();
  }
}
