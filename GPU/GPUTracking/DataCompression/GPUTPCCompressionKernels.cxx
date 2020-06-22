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
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;
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
    unsigned int nClustersStored = 0;
    CompressedClustersPtrs& GPUrestrict() c = compressor.mPtrs;
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

        myTrack = CAMath::AtomicAdd(&compressor.mMemory->nStoredTracks, 1u);
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
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  GPUParam& GPUrestrict() param = processors.param;
  unsigned int* sortBuffer = smem.step1.sortBuffer;
  for (int iSliceRow = iBlock; iSliceRow < GPUCA_NSLICES * GPUCA_ROW_COUNT; iSliceRow += nBlocks) {
    const int iSlice = iSliceRow / GPUCA_ROW_COUNT;
    const int iRow = iSliceRow % GPUCA_ROW_COUNT;
    const int idOffset = clusters->clusterOffset[iSlice][iRow];
    if (iThread == nThreads - 1) {
      smem.step1.nCount = 0;
    }
    unsigned int totalCount = 0;
    GPUbarrier();

    CompressedClustersPtrs& GPUrestrict() c = compressor.mPtrs;

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
        if (smem.step1.nCount + myIndex <= GPUCA_TPC_COMP_CHUNK_SIZE) {
          sortBuffer[smem.step1.nCount + myIndex - 1] = i;
        } else {
          storeLater = smem.step1.nCount + myIndex - 1 - GPUCA_TPC_COMP_CHUNK_SIZE;
        }
      }
      GPUbarrier();
      if (iThread == nThreads - 1) {
        smem.step1.nCount += myIndex;
      }
      GPUbarrier();

      if (smem.step1.nCount < GPUCA_TPC_COMP_CHUNK_SIZE && i < nn) {
        continue;
      }

      const unsigned int count = CAMath::Min(smem.step1.nCount, (unsigned int)GPUCA_TPC_COMP_CHUNK_SIZE);
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
        if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
          if (j != 0) {
            const ClusterNative& GPUrestrict() orgClPre = clusters->clusters[iSlice][iRow][sortBuffer[j - 1]];
            lastPad = orgClPre.padPacked;
            lastTime = orgClPre.getTimePacked();
          } else if (totalCount != 0) {
            const ClusterNative& GPUrestrict() orgClPre = clusters->clusters[iSlice][iRow][smem.step1.lastIndex];
            lastPad = orgClPre.padPacked;
            lastTime = orgClPre.getTimePacked();
          }

          c.padDiffU[outidx] = orgCl.padPacked - lastPad;
          c.timeDiffU[outidx] = (orgCl.getTimePacked() - lastTime) & 0xFFFFFF;
        } else {
          c.padDiffU[outidx] = orgCl.padPacked;
          c.timeDiffU[outidx] = orgCl.getTimePacked();
        }

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
        smem.step1.lastIndex = sortBuffer[count - 1];
        smem.step1.nCount -= count;
      }
    }

    if (iThread == nThreads - 1) {
      c.nSliceRowClusters[iSlice * GPUCA_ROW_COUNT + iRow] = totalCount;
      CAMath::AtomicAdd(&compressor.mMemory->nStoredUnattachedClusters, totalCount);
    }
    GPUbarrier();
  }
}

template <typename T>
GPUdi() bool GPUTPCCompressionKernels::isAlignedTo(const void* ptr)
{
  return reinterpret_cast<size_t>(ptr) % alignof(T) == 0;
}

template <>
GPUdi() void GPUTPCCompressionKernels::compressorMemcpy<unsigned char>(unsigned char* GPUrestrict() dst, const unsigned char* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
{
  CONSTEXPR int vec128Elems = CpyVector<unsigned char, Vec128>::Size;
  CONSTEXPR int vec64Elems = CpyVector<unsigned char, Vec64>::Size;
  CONSTEXPR int vec32Elems = CpyVector<unsigned char, Vec32>::Size;
  CONSTEXPR int vec16Elems = CpyVector<unsigned char, Vec16>::Size;

  if (size >= uint(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<unsigned char, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<unsigned char, Vec64>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec32Elems)) {
    compressorMemcpyVectorised<unsigned char, Vec32>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec16Elems)) {
    compressorMemcpyVectorised<unsigned char, Vec16>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <>
GPUdi() void GPUTPCCompressionKernels::compressorMemcpy<unsigned short>(unsigned short* GPUrestrict() dst, const unsigned short* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
{
  CONSTEXPR int vec128Elems = CpyVector<unsigned short, Vec128>::Size;
  CONSTEXPR int vec64Elems = CpyVector<unsigned short, Vec64>::Size;
  CONSTEXPR int vec32Elems = CpyVector<unsigned short, Vec32>::Size;

  if (size >= uint(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<unsigned short, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<unsigned short, Vec64>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec32Elems)) {
    compressorMemcpyVectorised<unsigned short, Vec32>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <>
GPUdi() void GPUTPCCompressionKernels::compressorMemcpy<unsigned int>(unsigned int* GPUrestrict() dst, const unsigned int* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
{
  CONSTEXPR int vec128Elems = CpyVector<unsigned int, Vec128>::Size;
  CONSTEXPR int vec64Elems = CpyVector<unsigned int, Vec64>::Size;

  if (size >= uint(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<unsigned int, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<unsigned int, Vec64>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <typename T>
GPUdi() void GPUTPCCompressionKernels::compressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, unsigned int size, int nThreads, int iThread, int nBlocks, int iBlock)
{
  unsigned int start = (size + nBlocks - 1) / nBlocks * iBlock + iThread;
  unsigned int end = CAMath::Min(size, (size + nBlocks - 1) / nBlocks * (iBlock + 1));
  for (unsigned int i = start; i < end; i += nThreads) {
    dst[i] = src[i];
  }
}

template <typename Scalar, typename BaseVector>
GPUdi() void GPUTPCCompressionKernels::compressorMemcpyVectorised(Scalar* dst, const Scalar* src, unsigned int size, int nThreads, int iThread)
{
  if (not isAlignedTo<BaseVector>(dst)) {
    size_t dsti = reinterpret_cast<size_t>(dst);
    int offset = (alignof(BaseVector) - dsti % alignof(BaseVector)) / sizeof(Scalar);
    compressorMemcpyBasic(dst, src, offset, nThreads, iThread);
    src += offset;
    dst += offset;
    size -= offset;
  }

  BaseVector* GPUrestrict() dstAligned = reinterpret_cast<BaseVector*>(dst);

  using CpyVec = CpyVector<Scalar, BaseVector>;
  unsigned int sizeAligned = size / CpyVec::Size;

  if (isAlignedTo<BaseVector>(src)) {
    const BaseVector* GPUrestrict() srcAligned = reinterpret_cast<const BaseVector*>(src);
    compressorMemcpyBasic(dstAligned, srcAligned, sizeAligned, nThreads, iThread);
  } else {
    for (unsigned int i = iThread; i < sizeAligned; i += nThreads) {
      CpyVec buf;
      for (unsigned int j = 0; j < CpyVec::Size; j++) {
        buf.elems[j] = src[i * CpyVec::Size + j];
      }
      dstAligned[i] = buf.all;
    }
  }

  int leftovers = size % CpyVec::Size;
  compressorMemcpyBasic(dst + size - leftovers, src + size - leftovers, leftovers, nThreads, iThread);
}

template <typename T>
GPUdi() unsigned int GPUTPCCompressionKernels::calculateWarpOffsets(GPUSharedMemory& smem, T* nums, unsigned int start, unsigned int end, int iWarp, int nLanes, int iLane)
{
  unsigned int offset = 0;
  if (iWarp > -1) {
    for (unsigned int i = start + iLane; i < end; i += nLanes) {
      offset += nums[i];
    }
  }
  offset = work_group_scan_inclusive_add(int(offset)); // FIXME: use scan with unsigned int
  if (iWarp > -1 && iLane == nLanes - 1) {
    smem.step2.warpOffset[iWarp] = offset;
  }
  GPUbarrier();
  offset = (iWarp <= 0) ? 0 : smem.step2.warpOffset[iWarp - 1];

  return offset;
}
template <>
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step2gather>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;

  int nWarps = nThreads / GPUCA_WARP_SIZE;
  int iWarp = iThread / GPUCA_WARP_SIZE;

  int nLanes = GPUCA_WARP_SIZE;
  int iLane = iThread % GPUCA_WARP_SIZE;

  if (iBlock == 0) {

    unsigned int nRows = compressor.NSLICES * GPUCA_ROW_COUNT;
    unsigned int rowsPerWarp = (nRows + nWarps - 1) / nWarps;
    unsigned int rowStart = rowsPerWarp * iWarp;
    unsigned int rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);

    unsigned int rowsOffset = calculateWarpOffsets(smem, compressor.mPtrs.nSliceRowClusters, rowStart, rowEnd, iWarp, nLanes, iLane);

    compressorMemcpy(compressor.mOutput->nSliceRowClusters, compressor.mPtrs.nSliceRowClusters, compressor.NSLICES * GPUCA_ROW_COUNT, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->nTrackClusters, compressor.mPtrs.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->qPtA, compressor.mPtrs.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->rowA, compressor.mPtrs.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->sliceA, compressor.mPtrs.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->timeA, compressor.mPtrs.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->padA, compressor.mPtrs.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);

    unsigned int sliceStart = rowStart / GPUCA_ROW_COUNT;
    unsigned int sliceEnd = rowEnd / GPUCA_ROW_COUNT;

    unsigned int sliceRowStart = rowStart % GPUCA_ROW_COUNT;
    unsigned int sliceRowEnd = rowEnd % GPUCA_ROW_COUNT;

    for (unsigned int i = sliceStart; i <= sliceEnd && i < compressor.NSLICES; i++) {
      for (unsigned int j = ((i == sliceStart) ? sliceRowStart : 0); j < ((i == sliceEnd) ? sliceRowEnd : GPUCA_ROW_COUNT); j++) {
        unsigned int nClusters = compressor.mPtrs.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
        compressorMemcpy(compressor.mOutput->qTotU + rowsOffset, compressor.mPtrs.qTotU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->qMaxU + rowsOffset, compressor.mPtrs.qMaxU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->flagsU + rowsOffset, compressor.mPtrs.flagsU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->padDiffU + rowsOffset, compressor.mPtrs.padDiffU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->timeDiffU + rowsOffset, compressor.mPtrs.timeDiffU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaPadU + rowsOffset, compressor.mPtrs.sigmaPadU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaTimeU + rowsOffset, compressor.mPtrs.sigmaTimeU + clusters->clusterOffset[i][j], nClusters, nLanes, iLane);
        rowsOffset += nClusters;
      }
    }
  }

  if (iBlock == 1) {
    unsigned int tracksPerWarp = (compressor.mMemory->nStoredTracks + nWarps - 1) / nWarps;
    unsigned int trackStart = tracksPerWarp * iWarp;
    unsigned int trackEnd = CAMath::Min(compressor.mMemory->nStoredTracks, trackStart + tracksPerWarp);

    unsigned int tracksOffset = calculateWarpOffsets(smem, compressor.mPtrs.nTrackClusters, trackStart, trackEnd, iWarp, nLanes, iLane);

    for (unsigned int i = trackStart; i < trackEnd; i += nLanes) {
      unsigned int nTrackClusters = 0;
      unsigned int srcOffset = 0;

      if (i + iLane < trackEnd) {
        nTrackClusters = compressor.mPtrs.nTrackClusters[i + iLane];
        srcOffset = compressor.mAttachedClusterFirstIndex[i + iLane];
      }
      smem.step2.sizes[iWarp][iLane] = nTrackClusters;
      smem.step2.srcOffsets[iWarp][iLane] = srcOffset;

      unsigned int elems = (i + nLanes < trackEnd) ? nLanes : (trackEnd - i);

      for (unsigned int j = 0; j < elems; j++) {
        nTrackClusters = smem.step2.sizes[iWarp][j];
        srcOffset = smem.step2.srcOffsets[iWarp][j];
        unsigned int idx = i + j;
        compressorMemcpy(compressor.mOutput->qTotA + tracksOffset, compressor.mPtrs.qTotA + srcOffset, nTrackClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->qMaxA + tracksOffset, compressor.mPtrs.qMaxA + srcOffset, nTrackClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->flagsA + tracksOffset, compressor.mPtrs.flagsA + srcOffset, nTrackClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaPadA + tracksOffset, compressor.mPtrs.sigmaPadA + srcOffset, nTrackClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaTimeA + tracksOffset, compressor.mPtrs.sigmaTimeA + srcOffset, nTrackClusters, nLanes, iLane);

        // First index stored with track
        compressorMemcpy(compressor.mOutput->rowDiffA + tracksOffset - idx, compressor.mPtrs.rowDiffA + srcOffset + 1, (nTrackClusters - 1), nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sliceLegDiffA + tracksOffset - idx, compressor.mPtrs.sliceLegDiffA + srcOffset + 1, (nTrackClusters - 1), nLanes, iLane);
        compressorMemcpy(compressor.mOutput->padResA + tracksOffset - idx, compressor.mPtrs.padResA + srcOffset + 1, (nTrackClusters - 1), nLanes, iLane);
        compressorMemcpy(compressor.mOutput->timeResA + tracksOffset - idx, compressor.mPtrs.timeResA + srcOffset + 1, (nTrackClusters - 1), nLanes, iLane);

        tracksOffset += nTrackClusters;
      }
    }
  }
}
