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
#include "GPUParam.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUTPCGeometry.h"
#include "GPUTPCClusterRejection.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step0attached>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const GPUParam& GPUrestrict() param = processors.param;

  char lastLeg = 0;
  int myTrack = 0;
  for (unsigned int i = get_global_id(0); i < ioPtrs.nMergedTracks; i += get_global_size(0)) {
    GPUbarrierWarp();
    const GPUTPCGMMergedTrack& GPUrestrict() trk = ioPtrs.mergedTracks[i];
    if (!trk.OK()) {
      continue;
    }
    bool rejectTrk = CAMath::Abs(trk.GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt || trk.MergedLooper();
    unsigned int nClustersStored = 0;
    CompressedClustersPtrs& GPUrestrict() c = compressor.mPtrs;
    unsigned int lastRow = 0, lastSlice = 0; // BUG: These should be unsigned char, but then CUDA breaks
    GPUTPCCompressionTrackModel track;
    float zOffset = 0;
    for (int k = trk.NClusters() - 1; k >= 0; k--) {
      const GPUTPCGMMergedTrackHit& GPUrestrict() hit = ioPtrs.mergedTrackHits[trk.FirstClusterRef() + k];
      if (hit.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }

      int hitId = hit.num;
      int attach = ioPtrs.mergedTrackHitAttachment[hitId];
      if ((attach & gputpcgmmergertypes::attachTrackMask) != i) {
        continue; // Main attachment to different track
      }
      bool rejectCluster = processors.param.rec.tpcRejectionMode && (rejectTrk || GPUTPCClusterRejection::GetIsRejected(attach));
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
        zOffset = z;
        track.Init(x, y, z - zOffset, param.SliceParam[hit.slice].Alpha, qpt, param);

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
        float time = CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(hit.slice, track.Z() + zOffset));
        c.timeResA[cidx] = (orgCl.getTimePacked() - orgCl.packTime(time)) & 0xFFFFFF;
        lastLeg = hit.leg;
      }
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
      if (k && track.Filter(y, z - zOffset, hit.row)) {
        break;
      }
      lastRow = hit.row;
      lastSlice = hit.slice;
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
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
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
        int attach = ioPtrs.mergedTrackHitAttachment[idx];
        bool unattached = attach == 0;

        if (unattached) {
          if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyB) {
            break;
          }
        } else if (processors.param.rec.tpcRejectionMode >= GPUSettings::RejectionStrategyA) {
          if (GPUTPCClusterRejection::GetIsRejected(attach)) {
            break;
          }
          int id = attach & gputpcgmmergertypes::attachTrackMask;
          auto& trk = ioPtrs.mergedTracks[id];
          if (CAMath::Abs(trk.GetParam().GetQPt()) > processors.param.rec.tpcRejectQPt || trk.MergedLooper()) {
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
        if (param.rec.tpcCompressionModes & GPUSettings::CompressionDifferences) {
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

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec32* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec32>(int iWarp)
{
  return buf32[iWarp];
}

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec64* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec64>(int iWarp)
{
  return buf64[iWarp];
}

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec128* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec128>(int iWarp)
{
  return buf128[iWarp];
}

template <typename T, typename S>
GPUdi() bool GPUTPCCompressionGatherKernels::isAlignedTo(const S* ptr)
{
  if CONSTEXPR17 (alignof(S) >= alignof(T)) {
    static_cast<void>(ptr);
    return true;
  } else {
    return reinterpret_cast<size_t>(ptr) % alignof(T) == 0;
  }
  return false; // BUG: Cuda complains about missing return value with constexpr if
}

template <>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<unsigned char>(unsigned char* GPUrestrict() dst, const unsigned char* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
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
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<unsigned short>(unsigned short* GPUrestrict() dst, const unsigned short* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
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
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<unsigned int>(unsigned int* GPUrestrict() dst, const unsigned int* GPUrestrict() src, unsigned int size, int nThreads, int iThread)
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

template <typename Scalar, typename BaseVector>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyVectorised(Scalar* dst, const Scalar* src, unsigned int size, int nThreads, int iThread)
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
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, unsigned int size, int nThreads, int iThread, int nBlocks, int iBlock)
{
  unsigned int start = (size + nBlocks - 1) / nBlocks * iBlock + iThread;
  unsigned int end = CAMath::Min(size, (size + nBlocks - 1) / nBlocks * (iBlock + 1));
  for (unsigned int i = start; i < end; i += nThreads) {
    dst[i] = src[i];
  }
}

template <typename V, typename T, typename S>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyBuffered(V* buf, T* GPUrestrict() dst, const T* GPUrestrict() src, const S* GPUrestrict() nums, const unsigned int* GPUrestrict() srcOffsets, unsigned int nTracks, int nLanes, int iLane, int diff)
{
  int shmPos = 0;
  unsigned int dstOffset = 0;
  V* GPUrestrict() dstAligned = nullptr;

  T* bufT = reinterpret_cast<T*>(buf);
  CONSTEXPR int bufSize = GPUCA_WARP_SIZE;
  CONSTEXPR int bufTSize = bufSize * sizeof(V) / sizeof(T);

  for (unsigned int i = 0; i < nTracks; i++) {
    unsigned int srcPos = 0;
    unsigned int srcOffset = srcOffsets[i] + diff;
    unsigned int srcSize = nums[i] - diff;

    if (dstAligned == nullptr) {
      if (not isAlignedTo<V>(dst)) {
        size_t dsti = reinterpret_cast<size_t>(dst);
        unsigned int offset = (alignof(V) - dsti % alignof(V)) / sizeof(T);
        offset = CAMath::Min<unsigned int>(offset, srcSize);
        compressorMemcpyBasic(dst, src + srcOffset, offset, nLanes, iLane);
        dst += offset;
        srcPos += offset;
      }
      if (isAlignedTo<V>(dst)) {
        dstAligned = reinterpret_cast<V*>(dst);
      }
    }
    while (srcPos < srcSize) {
      unsigned int shmElemsLeft = bufTSize - shmPos;
      unsigned int srcElemsLeft = srcSize - srcPos;
      unsigned int size = CAMath::Min(srcElemsLeft, shmElemsLeft);
      compressorMemcpyBasic(bufT + shmPos, src + srcOffset + srcPos, size, nLanes, iLane);
      srcPos += size;
      shmPos += size;

      if (shmPos >= bufTSize) {
        compressorMemcpyBasic(dstAligned + dstOffset, buf, bufSize, nLanes, iLane);
        dstOffset += bufSize;
        shmPos = 0;
      }
    }
  }

  compressorMemcpyBasic(reinterpret_cast<T*>(dstAligned + dstOffset), bufT, shmPos, nLanes, iLane);
}

template <typename T>
GPUdi() unsigned int GPUTPCCompressionGatherKernels::calculateWarpOffsets(GPUSharedMemory& smem, T* nums, unsigned int start, unsigned int end, int nWarps, int iWarp, int nLanes, int iLane)
{
  unsigned int blockOffset = 0;
  int iThread = nLanes * iWarp + iLane;
  int nThreads = nLanes * nWarps;
  unsigned int blockStart = work_group_broadcast(start, 0);
  for (unsigned int i = iThread; i < blockStart; i += nThreads) {
    blockOffset += nums[i];
  }
  blockOffset = work_group_reduce_add(blockOffset);

  unsigned int offset = 0;
  for (unsigned int i = start + iLane; i < end; i += nLanes) {
    offset += nums[i];
  }
  offset = work_group_scan_inclusive_add(offset);
  if (iWarp > -1 && iLane == nLanes - 1) {
    smem.warpOffset[iWarp] = offset;
  }
  GPUbarrier();
  offset = (iWarp <= 0) ? 0 : smem.warpOffset[iWarp - 1];
  GPUbarrier();

  return offset + blockOffset;
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::unbuffered>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
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
    if (rowStart >= nRows) {
      rowStart = 0;
      rowEnd = 0;
    }

    unsigned int rowsOffset = calculateWarpOffsets(smem, compressor.mPtrs.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);

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
    if (trackStart >= compressor.mMemory->nStoredTracks) {
      trackStart = 0;
      trackEnd = 0;
    }

    unsigned int tracksOffset = calculateWarpOffsets(smem, compressor.mPtrs.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);

    for (unsigned int i = trackStart; i < trackEnd; i += nLanes) {
      unsigned int nTrackClusters = 0;
      unsigned int srcOffset = 0;

      if (i + iLane < trackEnd) {
        nTrackClusters = compressor.mPtrs.nTrackClusters[i + iLane];
        srcOffset = compressor.mAttachedClusterFirstIndex[i + iLane];
      }
      smem.unbuffered.sizes[iWarp][iLane] = nTrackClusters;
      smem.unbuffered.srcOffsets[iWarp][iLane] = srcOffset;

      unsigned int elems = (i + nLanes < trackEnd) ? nLanes : (trackEnd - i);

      for (unsigned int j = 0; j < elems; j++) {
        nTrackClusters = smem.unbuffered.sizes[iWarp][j];
        srcOffset = smem.unbuffered.srcOffsets[iWarp][j];
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

template <typename V>
GPUdii() void GPUTPCCompressionGatherKernels::gatherBuffered(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{

  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;

  int nWarps = nThreads / GPUCA_WARP_SIZE;
  int iWarp = iThread / GPUCA_WARP_SIZE;

  int nGlobalWarps = nWarps * nBlocks;
  int iGlobalWarp = nWarps * iBlock + iWarp;

  int nLanes = GPUCA_WARP_SIZE;
  int iLane = iThread % GPUCA_WARP_SIZE;

  auto& input = compressor.mPtrs;
  auto* output = compressor.mOutput;

  unsigned int nRows = compressor.NSLICES * GPUCA_ROW_COUNT;
  unsigned int rowsPerWarp = (nRows + nGlobalWarps - 1) / nGlobalWarps;
  unsigned int rowStart = rowsPerWarp * iGlobalWarp;
  unsigned int rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);
  if (rowStart >= nRows) {
    rowStart = 0;
    rowEnd = 0;
  }
  rowsPerWarp = rowEnd - rowStart;

  unsigned int rowsOffset = calculateWarpOffsets(smem, input.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);

  unsigned int nStoredTracks = compressor.mMemory->nStoredTracks;
  unsigned int tracksPerWarp = (nStoredTracks + nGlobalWarps - 1) / nGlobalWarps;
  unsigned int trackStart = tracksPerWarp * iGlobalWarp;
  unsigned int trackEnd = CAMath::Min(nStoredTracks, trackStart + tracksPerWarp);
  if (trackStart >= nStoredTracks) {
    trackStart = 0;
    trackEnd = 0;
  }
  tracksPerWarp = trackEnd - trackStart;

  unsigned int tracksOffset = calculateWarpOffsets(smem, input.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);

  if (iBlock == 0) {
    compressorMemcpyBasic(output->nSliceRowClusters, input.nSliceRowClusters, compressor.NSLICES * GPUCA_ROW_COUNT, nThreads, iThread);
    compressorMemcpyBasic(output->nTrackClusters, input.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->qPtA, input.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->rowA, input.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->sliceA, input.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->timeA, input.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->padA, input.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);
  }

  const unsigned int* clusterOffsets = reinterpret_cast<const unsigned int*>(clusters->clusterOffset) + rowStart;
  const unsigned int* nSliceRowClusters = input.nSliceRowClusters + rowStart;

  auto* buf = smem.getBuffer<V>(iWarp);

  compressorMemcpyBuffered(buf, output->qTotU + rowsOffset, input.qTotU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->qMaxU + rowsOffset, input.qMaxU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->flagsU + rowsOffset, input.flagsU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->padDiffU + rowsOffset, input.padDiffU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->timeDiffU + rowsOffset, input.timeDiffU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaPadU + rowsOffset, input.sigmaPadU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaTimeU + rowsOffset, input.sigmaTimeU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);

  const unsigned short* nTrackClustersPtr = input.nTrackClusters + trackStart;
  const unsigned int* aClsFstIdx = compressor.mAttachedClusterFirstIndex + trackStart;

  compressorMemcpyBuffered(buf, output->qTotA + tracksOffset, input.qTotA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->qMaxA + tracksOffset, input.qMaxA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->flagsA + tracksOffset, input.flagsA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaPadA + tracksOffset, input.sigmaPadA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaTimeA + tracksOffset, input.sigmaTimeA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);

  // First index stored with track
  unsigned int tracksOffsetDiff = tracksOffset - trackStart;
  compressorMemcpyBuffered(buf, output->rowDiffA + tracksOffsetDiff, input.rowDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->sliceLegDiffA + tracksOffsetDiff, input.sliceLegDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->padResA + tracksOffsetDiff, input.padResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->timeResA + tracksOffsetDiff, input.timeResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
}

GPUdii() void GPUTPCCompressionGatherKernels::gatherMulti(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;
  const auto& input = compressor.mPtrs;
  auto* output = compressor.mOutput;

  const int nWarps = nThreads / GPUCA_WARP_SIZE;
  const int iWarp = iThread / GPUCA_WARP_SIZE;
  const int nLanes = GPUCA_WARP_SIZE;
  const int iLane = iThread % GPUCA_WARP_SIZE;
  auto* buf = smem.getBuffer<Vec128>(iWarp);

  if (iBlock == 0) {
    compressorMemcpyBasic(output->nSliceRowClusters, input.nSliceRowClusters, compressor.NSLICES * GPUCA_ROW_COUNT, nThreads, iThread);
    compressorMemcpyBasic(output->nTrackClusters, input.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->qPtA, input.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->rowA, input.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->sliceA, input.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->timeA, input.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->padA, input.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);
  } else if (iBlock & 1) {
    const unsigned int nGlobalWarps = nWarps * (nBlocks - 1) / 2;
    const unsigned int iGlobalWarp = nWarps * (iBlock - 1) / 2 + iWarp;

    const unsigned int nRows = compressor.NSLICES * GPUCA_ROW_COUNT;
    unsigned int rowsPerWarp = (nRows + nGlobalWarps - 1) / nGlobalWarps;
    unsigned int rowStart = rowsPerWarp * iGlobalWarp;
    unsigned int rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);
    if (rowStart >= nRows) {
      rowStart = 0;
      rowEnd = 0;
    }
    rowsPerWarp = rowEnd - rowStart;

    const unsigned int rowsOffset = calculateWarpOffsets(smem, input.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);
    const unsigned int* clusterOffsets = reinterpret_cast<const unsigned int*>(clusters->clusterOffset) + rowStart;
    const unsigned int* nSliceRowClusters = input.nSliceRowClusters + rowStart;

    compressorMemcpyBuffered(buf, output->qTotU + rowsOffset, input.qTotU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->qMaxU + rowsOffset, input.qMaxU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->flagsU + rowsOffset, input.flagsU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->padDiffU + rowsOffset, input.padDiffU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->timeDiffU + rowsOffset, input.timeDiffU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaPadU + rowsOffset, input.sigmaPadU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaTimeU + rowsOffset, input.sigmaTimeU, nSliceRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0);
  } else {
    const unsigned int nGlobalWarps = nWarps * (nBlocks - 1) / 2;
    const unsigned int iGlobalWarp = nWarps * (iBlock / 2 - 1) + iWarp;

    const unsigned int nStoredTracks = compressor.mMemory->nStoredTracks;
    unsigned int tracksPerWarp = (nStoredTracks + nGlobalWarps - 1) / nGlobalWarps;
    unsigned int trackStart = tracksPerWarp * iGlobalWarp;
    unsigned int trackEnd = CAMath::Min(nStoredTracks, trackStart + tracksPerWarp);
    if (trackStart >= nStoredTracks) {
      trackStart = 0;
      trackEnd = 0;
    }
    tracksPerWarp = trackEnd - trackStart;

    const unsigned int tracksOffset = calculateWarpOffsets(smem, input.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);
    const unsigned short* nTrackClustersPtr = input.nTrackClusters + trackStart;
    const unsigned int* aClsFstIdx = compressor.mAttachedClusterFirstIndex + trackStart;

    compressorMemcpyBuffered(buf, output->qTotA + tracksOffset, input.qTotA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->qMaxA + tracksOffset, input.qMaxA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->flagsA + tracksOffset, input.flagsA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaPadA + tracksOffset, input.sigmaPadA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaTimeA + tracksOffset, input.sigmaTimeA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);

    // First index stored with track
    unsigned int tracksOffsetDiff = tracksOffset - trackStart;
    compressorMemcpyBuffered(buf, output->rowDiffA + tracksOffsetDiff, input.rowDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->sliceLegDiffA + tracksOffsetDiff, input.sliceLegDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->padResA + tracksOffsetDiff, input.padResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->timeResA + tracksOffsetDiff, input.timeResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  }
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered32>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec32>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered64>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec64>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered128>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec128>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::multiBlock>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& GPUrestrict() smem, processorType& GPUrestrict() processors)
{
  gatherMulti(nBlocks, nThreads, iBlock, iThread, smem, processors);
}
