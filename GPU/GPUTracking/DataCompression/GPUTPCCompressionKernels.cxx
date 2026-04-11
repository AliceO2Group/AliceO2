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

/// \file GPUTPCCompressionKernels.cxx
/// \author David Rohr

#include "GPUTPCCompressionKernels.h"
#include "GPUConstantMem.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUTPCClusterRejection.h"
#include "GPUTPCCompressionKernels.inc"

using namespace o2::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step0attached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const GPUParam& GPUrestrict() param = processors.param;

  int32_t myTrack = 0;
  for (uint32_t i = get_global_id(0); i < ioPtrs.nMergedTracks; i += get_global_size(0)) {
    GPUbarrierWarp();
    const GPUTPCGMMergedTrack& GPUrestrict() trk = ioPtrs.mergedTracks[i];
    if (!trk.OK()) {
      continue;
    }
    bool rejectTrk = GPUTPCClusterRejection::IsTrackRejected(trk, param);
    uint32_t nClustersStored = 0;
    CompressedClustersPtrs& GPUrestrict() c = compressor.mPtrs;
    uint8_t lastRow = 0, lastSector = 0;
    GPUTPCCompressionTrackModel track;
    float zOffset = 0;
    for (int32_t k = trk.NClusters() - 1; k >= 0; k--) {
      const GPUTPCGMMergedTrackHit& GPUrestrict() hit = ioPtrs.mergedTrackHits[trk.FirstClusterRef() + k];
      if (hit.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }

      int32_t hitId = hit.num;
      int32_t attach = ioPtrs.mergedTrackHitAttachment[hitId];
      if ((attach & gputpcgmmergertypes::attachTrackMask) != i) {
        continue; // Main attachment to different track
      }
      bool rejectCluster = processors.param.rec.tpc.rejectionStrategy >= GPUSettings::RejectionStrategyA && !(attach & gputpcgmmergertypes::attachProtect) && (rejectTrk || GPUTPCClusterRejection::GetIsRejected(attach));
      if (rejectCluster) {
        compressor.mClusterStatus[hitId] = 1; // Cluster rejected, do not store
        continue;
      } else if (processors.param.rec.tpc.rejectionStrategy >= GPUSettings::RejectionStrategyA && rejectTrk) {
        continue;
      }

      if (!(param.rec.tpc.compressionTypeMask & GPUSettings::CompressionTrackModel)) {
        continue; // No track model compression
      }
      const ClusterNative& GPUrestrict() orgCl = clusters -> clusters[hit.sector][hit.row][hit.num - clusters->clusterOffset[hit.sector][hit.row]];
      constexpr GPUTPCGeometry geo;
      float x = geo.Row2X(hit.row);
      float y = track.LinearPad2Y(hit.sector, orgCl.getPad(), geo.PadWidth(hit.row), geo.NPads(hit.row));
      float z = geo.LinearTime2Z(hit.sector, orgCl.getTime());
      if (nClustersStored) {
        if ((hit.sector < GPUTPCGeometry::NSECTORS) ^ (lastSector < GPUTPCGeometry::NSECTORS)) {
          break;
        }
        if (track.Propagate(geo.Row2X(hit.row), GPUTPCGeometry::SectorAlpha(hit.sector))) {
          break;
        }
      }

      compressor.mClusterStatus[hitId] = 1; // Cluster compressed in track model, do not store as difference

      int32_t cidx = trk.FirstClusterRef() + nClustersStored++;
      if (nClustersStored == 1) {
        uint8_t qpt = fabs(trk.GetParam().GetQPt()) < 20.f ? (trk.GetParam().GetQPt() * (127.f / 20.f) + 127.5f) : (trk.GetParam().GetQPt() > 0 ? 254 : 0);
        zOffset = z;
        track.Init(x, y, z - zOffset, GPUTPCGeometry::SectorAlpha(hit.sector), qpt, param);

        myTrack = CAMath::AtomicAdd(&compressor.mMemory->nStoredTracks, 1u);
        compressor.mAttachedClusterFirstIndex[myTrack] = trk.FirstClusterRef();
        c.qPtA[myTrack] = qpt;
        c.rowA[myTrack] = hit.row;
        c.sliceA[myTrack] = hit.sector;
        c.timeA[myTrack] = orgCl.getTimePacked();
        c.padA[myTrack] = orgCl.padPacked;
      } else {
        uint32_t row = hit.row;
        uint32_t sector = hit.sector;

        if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionDifferences) {
          if (lastRow > row) {
            row += GPUTPCGeometry::NROWS;
          }
          row -= lastRow;
          if (lastSector > sector) {
            sector += compressor.NSECTORS;
          }
          sector -= lastSector;
        }
        c.rowDiffA[cidx] = row;
        c.sliceLegDiffA[cidx] = sector;
        float pad = CAMath::Max(0.f, CAMath::Min((float)geo.NPads(GPUTPCGeometry::NROWS - 1), track.LinearY2Pad(hit.sector, track.Y(), geo.PadWidth(hit.row), geo.NPads(hit.row))));
        c.padResA[cidx] = orgCl.padPacked - orgCl.packPad(pad);
        float time = CAMath::Max(0.f, geo.LinearZ2Time(hit.sector, track.Z() + zOffset));
        c.timeResA[cidx] = (orgCl.getTimePacked() - orgCl.packTime(time)) & 0xFFFFFF;
      }
      uint16_t qtot = orgCl.qTot, qmax = orgCl.qMax;
      uint8_t sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
      if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionTruncate) {
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
      lastSector = hit.sector;
    }
    if (nClustersStored) {
      CAMath::AtomicAdd(&compressor.mMemory->nStoredAttachedClusters, nClustersStored);
      c.nTrackClusters[myTrack] = nClustersStored;
    }
  }
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<GPUSettings::SortTime>::operator()(uint32_t a, uint32_t b) const
{
  return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<GPUSettings::SortPad>::operator()(uint32_t a, uint32_t b) const
{
  return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<GPUSettings::SortZTimePad>::operator()(uint32_t a, uint32_t b) const
{
  if (mClsPtr[a].getTimePacked() >> 3 == mClsPtr[b].getTimePacked() >> 3) {
    return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
  }
  return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
}

template <>
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<GPUSettings::SortZPadTime>::operator()(uint32_t a, uint32_t b) const
{
  if (mClsPtr[a].padPacked >> 3 == mClsPtr[b].padPacked >> 3) {
    return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
  }
  return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
}

template <> // Deterministic comparison
GPUd() bool GPUTPCCompressionKernels::GPUTPCCompressionKernels_Compare<4>::operator()(uint32_t a, uint32_t b) const
{
  if (mClsPtr[a].getTimePacked() != mClsPtr[b].getTimePacked()) {
    return mClsPtr[a].getTimePacked() < mClsPtr[b].getTimePacked();
  }
  if (mClsPtr[a].padPacked != mClsPtr[b].padPacked) {
    return mClsPtr[a].padPacked < mClsPtr[b].padPacked;
  }
  return mClsPtr[a].qTot < mClsPtr[b].qTot;
}

GPUd() bool GPUTPCCompression::rejectCluster(int32_t idx, const GPUParam& GPUrestrict() param, const GPUTrackingInOutPointers& GPUrestrict() ioPtrs) const
{
  if (mClusterStatus[idx]) {
    return true;
  }
  int32_t attach = ioPtrs.mergedTrackHitAttachment[idx];
  bool unattached = attach == 0;

  if (unattached) {
    if (param.rec.tpc.rejectionStrategy >= GPUSettings::RejectionStrategyB) {
      return true;
    }
  } else if (param.rec.tpc.rejectionStrategy >= GPUSettings::RejectionStrategyA) {
    if (GPUTPCClusterRejection::GetIsRejected(attach)) {
      return true;
    }
    if (attach & gputpcgmmergertypes::attachProtect) {
      return false;
    }
    int32_t id = attach & gputpcgmmergertypes::attachTrackMask;
    auto& trk = ioPtrs.mergedTracks[id];
    if (GPUTPCClusterRejection::IsTrackRejected(trk, param)) {
      return true;
    }
  }
  return false;
}

template <>
GPUdii() void GPUTPCCompressionKernels::Thread<GPUTPCCompressionKernels::step1unattached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  const GPUTrackingInOutPointers& GPUrestrict() ioPtrs = processors.ioPtrs;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = ioPtrs.clustersNative;
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  GPUParam& GPUrestrict() param = processors.param;
  uint32_t* sortBuffer = smem.sortBuffer;
  for (uint32_t iSectorRow = iBlock; iSectorRow < GPUTPCGeometry::NSECTORS * GPUTPCGeometry::NROWS; iSectorRow += nBlocks) {
    const uint32_t iSector = iSectorRow / GPUTPCGeometry::NROWS;
    const uint32_t iRow = iSectorRow % GPUTPCGeometry::NROWS;
    const uint32_t idOffset = clusters->clusterOffset[iSector][iRow];
    const uint32_t idOffsetOut = clusters->clusterOffset[iSector][iRow] * compressor.mMaxClusterFactorBase1024 / 1024;                           // 32 bit enough for number of clusters per row * 1024
    const uint32_t idOffsetOutMax = ((const uint32_t*)clusters->clusterOffset[iSector])[iRow + 1] * compressor.mMaxClusterFactorBase1024 / 1024; // Array out of bounds access is ok, since it goes to the correct nClustersTotal
    if (iThread == nThreads - 1) {
      smem.nCount = 0;
    }
    uint32_t totalCount = 0;
    GPUbarrier();

    CompressedClustersPtrs& GPUrestrict() c = compressor.mPtrs;

    const uint32_t nn = CAMath::nextMultipleOf<GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached)>(clusters->nClusters[iSector][iRow]);
    for (uint32_t i = iThread; i < nn + nThreads; i += nThreads) {
      const int32_t idx = idOffset + i;
      int32_t storeCluster = i < clusters->nClusters[iSector][iRow] && !compressor.rejectCluster(idx, param, ioPtrs);

      GPUbarrier();
      int32_t myIndex = work_group_scan_inclusive_add(storeCluster);
      int32_t storeLater = -1;
      if (storeCluster) {
        if (smem.nCount + myIndex <= constants::TPC_COMP_CHUNK_SIZE) {
          sortBuffer[smem.nCount + myIndex - 1] = i;
        } else {
          storeLater = smem.nCount + myIndex - 1 - constants::TPC_COMP_CHUNK_SIZE;
        }
      }
      GPUbarrier();
      if (iThread == nThreads - 1) {
        smem.nCount += myIndex;
      }
      GPUbarrier();

      if (smem.nCount < constants::TPC_COMP_CHUNK_SIZE && i < nn) {
        continue;
      }

      uint32_t count = CAMath::Min(smem.nCount, (uint32_t)constants::TPC_COMP_CHUNK_SIZE);
      if (idOffsetOut + totalCount + count > idOffsetOutMax) {
        if (iThread == nThreads - 1) {
          compressor.raiseError(GPUErrors::ERROR_COMPRESSION_ROW_HIT_OVERFLOW, iSector * 1000 + iRow, idOffsetOut + totalCount + count, idOffsetOutMax);
        }
        break;
      }
      if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionDifferences) {
#ifdef GPUCA_GPUCODE
        static_assert(GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCompressionKernels_step1unattached) * 2 <= constants::TPC_COMP_CHUNK_SIZE);
#endif
#ifdef GPUCA_DETERMINISTIC_MODE
        CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortZPadTime>(clusters->clusters[iSector][iRow]));
#else  // GPUCA_DETERMINISTIC_MODE
        if (param.rec.tpc.compressionSortOrder == GPUSettings::SortZPadTime) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortZPadTime>(clusters->clusters[iSector][iRow]));
        } else if (param.rec.tpc.compressionSortOrder == GPUSettings::SortZTimePad) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortZTimePad>(clusters->clusters[iSector][iRow]));
        } else if (param.rec.tpc.compressionSortOrder == GPUSettings::SortPad) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortPad>(clusters->clusters[iSector][iRow]));
        } else if (param.rec.tpc.compressionSortOrder == GPUSettings::SortTime) {
          CAAlgo::sortInBlock(sortBuffer, sortBuffer + count, GPUTPCCompressionKernels_Compare<GPUSettings::SortTime>(clusters->clusters[iSector][iRow]));
        }
#endif // GPUCA_DETERMINISTIC_MODE
        GPUbarrier();
      }

      for (uint32_t j = get_local_id(0); j < count; j += get_local_size(0)) {
        int32_t outidx = idOffsetOut + totalCount + j;
        const ClusterNative& GPUrestrict() orgCl = clusters -> clusters[iSector][iRow][sortBuffer[j]];

        int32_t preId = j != 0 ? (int32_t)sortBuffer[j - 1] : (totalCount != 0 ? (int32_t)smem.lastIndex : -1);
        GPUTPCCompression_EncodeUnattached(param.rec.tpc.compressionTypeMask, orgCl, c.timeDiffU[outidx], c.padDiffU[outidx], preId == -1 ? nullptr : &clusters->clusters[iSector][iRow][preId]);

        uint16_t qtot = orgCl.qTot, qmax = orgCl.qMax;
        uint8_t sigmapad = orgCl.sigmaPadPacked, sigmatime = orgCl.sigmaTimePacked;
        if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionTruncate) {
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
      if (storeLater >= 0) {
        sortBuffer[storeLater] = i;
      }
      totalCount += count;
      if (iThread == nThreads - 1 && count) {
        smem.lastIndex = sortBuffer[count - 1];
        smem.nCount -= count;
      }
    }

    if (iThread == nThreads - 1) {
      c.nSliceRowClusters[iSector * GPUTPCGeometry::NROWS + iRow] = totalCount;
      CAMath::AtomicAdd(&compressor.mMemory->nStoredUnattachedClusters, totalCount);
    }
    GPUbarrier();
  }
}

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec32* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec32>(int32_t iWarp)
{
  return buf32[iWarp];
}

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec64* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec64>(int32_t iWarp)
{
  return buf64[iWarp];
}

template <>
GPUdi() GPUTPCCompressionGatherKernels::Vec128* GPUTPCCompressionGatherKernels::GPUSharedMemory::getBuffer<GPUTPCCompressionGatherKernels::Vec128>(int32_t iWarp)
{
  return buf128[iWarp];
}

template <typename T, typename S>
GPUdi() bool GPUTPCCompressionGatherKernels::isAlignedTo(const S* ptr)
{
  if constexpr (alignof(S) >= alignof(T)) {
    static_cast<void>(ptr);
    return true;
  } else {
    return reinterpret_cast<size_t>(ptr) % alignof(T) == 0;
  }
}

template <>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<uint8_t>(uint8_t* GPUrestrict() dst, const uint8_t* GPUrestrict() src, uint32_t size, int32_t nThreads, int32_t iThread)
{
  constexpr const int32_t vec128Elems = CpyVector<uint8_t, Vec128>::Size;
  constexpr const int32_t vec64Elems = CpyVector<uint8_t, Vec64>::Size;
  constexpr const int32_t vec32Elems = CpyVector<uint8_t, Vec32>::Size;
  constexpr const int32_t vec16Elems = CpyVector<uint8_t, Vec16>::Size;

  if (size >= uint32_t(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<uint8_t, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<uint8_t, Vec64>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec32Elems)) {
    compressorMemcpyVectorised<uint8_t, Vec32>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec16Elems)) {
    compressorMemcpyVectorised<uint8_t, Vec16>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<uint16_t>(uint16_t* GPUrestrict() dst, const uint16_t* GPUrestrict() src, uint32_t size, int32_t nThreads, int32_t iThread)
{
  constexpr const int32_t vec128Elems = CpyVector<uint16_t, Vec128>::Size;
  constexpr const int32_t vec64Elems = CpyVector<uint16_t, Vec64>::Size;
  constexpr const int32_t vec32Elems = CpyVector<uint16_t, Vec32>::Size;

  if (size >= uint32_t(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<uint16_t, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<uint16_t, Vec64>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec32Elems)) {
    compressorMemcpyVectorised<uint16_t, Vec32>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpy<uint32_t>(uint32_t* GPUrestrict() dst, const uint32_t* GPUrestrict() src, uint32_t size, int32_t nThreads, int32_t iThread)
{
  constexpr const int32_t vec128Elems = CpyVector<uint32_t, Vec128>::Size;
  constexpr const int32_t vec64Elems = CpyVector<uint32_t, Vec64>::Size;

  if (size >= uint32_t(nThreads * vec128Elems)) {
    compressorMemcpyVectorised<uint32_t, Vec128>(dst, src, size, nThreads, iThread);
  } else if (size >= uint32_t(nThreads * vec64Elems)) {
    compressorMemcpyVectorised<uint32_t, Vec64>(dst, src, size, nThreads, iThread);
  } else {
    compressorMemcpyBasic(dst, src, size, nThreads, iThread);
  }
}

template <typename Scalar, typename BaseVector>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyVectorised(Scalar* dst, const Scalar* src, uint32_t size, int32_t nThreads, int32_t iThread)
{
  if (not isAlignedTo<BaseVector>(dst)) {
    size_t dsti = reinterpret_cast<size_t>(dst);
    int32_t offset = (alignof(BaseVector) - dsti % alignof(BaseVector)) / sizeof(Scalar);
    compressorMemcpyBasic(dst, src, offset, nThreads, iThread);
    src += offset;
    dst += offset;
    size -= offset;
  }

  BaseVector* GPUrestrict() dstAligned = reinterpret_cast<BaseVector*>(dst);

  using CpyVec = CpyVector<Scalar, BaseVector>;
  uint32_t sizeAligned = size / CpyVec::Size;

  if (isAlignedTo<BaseVector>(src)) {
    const BaseVector* GPUrestrict() srcAligned = reinterpret_cast<const BaseVector*>(src);
    compressorMemcpyBasic(dstAligned, srcAligned, sizeAligned, nThreads, iThread);
  } else {
    for (uint32_t i = iThread; i < sizeAligned; i += nThreads) {
      CpyVec buf;
      for (uint32_t j = 0; j < CpyVec::Size; j++) {
        buf.elems[j] = src[i * CpyVec::Size + j];
      }
      dstAligned[i] = buf.all;
    }
  }

  int32_t leftovers = size % CpyVec::Size;
  compressorMemcpyBasic(dst + size - leftovers, src + size - leftovers, leftovers, nThreads, iThread);
}

template <typename T>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, uint32_t size, int32_t nThreads, int32_t iThread, int32_t nBlocks, int32_t iBlock)
{
  uint32_t start = (size + nBlocks - 1) / nBlocks * iBlock + iThread;
  uint32_t end = CAMath::Min(size, (size + nBlocks - 1) / nBlocks * (iBlock + 1));
  for (uint32_t i = start; i < end; i += nThreads) {
    dst[i] = src[i];
  }
}

template <typename V, typename T, typename S>
GPUdi() void GPUTPCCompressionGatherKernels::compressorMemcpyBuffered(V* buf, T* GPUrestrict() dst, const T* GPUrestrict() src, const S* GPUrestrict() nums, const uint32_t* GPUrestrict() srcOffsets, uint32_t nEntries, int32_t nLanes, int32_t iLane, int32_t diff, size_t scaleBase1024)
{
  int32_t shmPos = 0;
  uint32_t dstOffset = 0;
  V* GPUrestrict() dstAligned = nullptr;

  T* bufT = reinterpret_cast<T*>(buf);
  constexpr const int32_t bufSize = GPUCA_WARP_SIZE;
  constexpr const int32_t bufTSize = bufSize * sizeof(V) / sizeof(T);

  for (uint32_t i = 0; i < nEntries; i++) {
    uint32_t srcPos = 0;
    uint32_t srcOffset = (srcOffsets[i] * scaleBase1024 / 1024) + diff;
    uint32_t srcSize = nums[i] - diff;

    if (dstAligned == nullptr) {
      if (not isAlignedTo<V>(dst)) {
        size_t dsti = reinterpret_cast<size_t>(dst);
        uint32_t offset = (alignof(V) - dsti % alignof(V)) / sizeof(T);
        offset = CAMath::Min<uint32_t>(offset, srcSize);
        compressorMemcpyBasic(dst, src + srcOffset, offset, nLanes, iLane);
        dst += offset;
        srcPos += offset;
      }
      if (isAlignedTo<V>(dst)) {
        dstAligned = reinterpret_cast<V*>(dst);
      }
    }
    while (srcPos < srcSize) {
      uint32_t shmElemsLeft = bufTSize - shmPos;
      uint32_t srcElemsLeft = srcSize - srcPos;
      uint32_t size = CAMath::Min(srcElemsLeft, shmElemsLeft);
      compressorMemcpyBasic(bufT + shmPos, src + srcOffset + srcPos, size, nLanes, iLane);
      srcPos += size;
      shmPos += size;
      GPUbarrierWarp();

      if (shmPos >= bufTSize) {
        compressorMemcpyBasic(dstAligned + dstOffset, buf, bufSize, nLanes, iLane);
        dstOffset += bufSize;
        shmPos = 0;
        GPUbarrierWarp();
      }
    }
  }

  compressorMemcpyBasic(reinterpret_cast<T*>(dstAligned + dstOffset), bufT, shmPos, nLanes, iLane);
  GPUbarrierWarp();
}

template <typename T>
GPUdi() uint32_t GPUTPCCompressionGatherKernels::calculateWarpOffsets(GPUSharedMemory& smem, T* nums, uint32_t start, uint32_t end, int32_t nWarps, int32_t iWarp, int32_t nLanes, int32_t iLane)
{
  uint32_t blockOffset = 0;
  int32_t iThread = nLanes * iWarp + iLane;
  int32_t nThreads = nLanes * nWarps;
  uint32_t blockStart = work_group_broadcast(start, 0);
  for (uint32_t i = iThread; i < blockStart; i += nThreads) {
    blockOffset += nums[i];
  }
  blockOffset = work_group_reduce_add(blockOffset);

  uint32_t offset = 0;
  for (uint32_t i = start + iLane; i < end; i += nLanes) {
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
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::unbuffered>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;

  int32_t nWarps = nThreads / GPUCA_WARP_SIZE;
  int32_t iWarp = iThread / GPUCA_WARP_SIZE;

  int32_t nLanes = GPUCA_WARP_SIZE;
  int32_t iLane = iThread % GPUCA_WARP_SIZE;

  if (iBlock == 0) {

    uint32_t nRows = compressor.NSECTORS * GPUTPCGeometry::NROWS;
    uint32_t rowsPerWarp = (nRows + nWarps - 1) / nWarps;
    uint32_t rowStart = rowsPerWarp * iWarp;
    uint32_t rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);
    if (rowStart >= nRows) {
      rowStart = 0;
      rowEnd = 0;
    }

    uint32_t rowsOffset = calculateWarpOffsets(smem, compressor.mPtrs.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);

    compressorMemcpy(compressor.mOutput->nSliceRowClusters, compressor.mPtrs.nSliceRowClusters, compressor.NSECTORS * GPUTPCGeometry::NROWS, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->nTrackClusters, compressor.mPtrs.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->qPtA, compressor.mPtrs.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->rowA, compressor.mPtrs.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->sliceA, compressor.mPtrs.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->timeA, compressor.mPtrs.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpy(compressor.mOutput->padA, compressor.mPtrs.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);

    uint32_t sectorStart = rowStart / GPUTPCGeometry::NROWS;
    uint32_t sectorEnd = rowEnd / GPUTPCGeometry::NROWS;

    uint32_t sectorRowStart = rowStart % GPUTPCGeometry::NROWS;
    uint32_t sectorRowEnd = rowEnd % GPUTPCGeometry::NROWS;

    for (uint32_t i = sectorStart; i <= sectorEnd && i < compressor.NSECTORS; i++) {
      for (uint32_t j = ((i == sectorStart) ? sectorRowStart : 0); j < ((i == sectorEnd) ? sectorRowEnd : GPUTPCGeometry::NROWS); j++) {
        uint32_t nClusters = compressor.mPtrs.nSliceRowClusters[i * GPUTPCGeometry::NROWS + j];
        uint32_t clusterOffsetInCache = clusters->clusterOffset[i][j] * compressor.mMaxClusterFactorBase1024 / 1024;
        compressorMemcpy(compressor.mOutput->qTotU + rowsOffset, compressor.mPtrs.qTotU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->qMaxU + rowsOffset, compressor.mPtrs.qMaxU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->flagsU + rowsOffset, compressor.mPtrs.flagsU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->padDiffU + rowsOffset, compressor.mPtrs.padDiffU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->timeDiffU + rowsOffset, compressor.mPtrs.timeDiffU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaPadU + rowsOffset, compressor.mPtrs.sigmaPadU + clusterOffsetInCache, nClusters, nLanes, iLane);
        compressorMemcpy(compressor.mOutput->sigmaTimeU + rowsOffset, compressor.mPtrs.sigmaTimeU + clusterOffsetInCache, nClusters, nLanes, iLane);
        rowsOffset += nClusters;
      }
    }
  }

  if (iBlock == 1) {
    uint32_t tracksPerWarp = (compressor.mMemory->nStoredTracks + nWarps - 1) / nWarps;
    uint32_t trackStart = tracksPerWarp * iWarp;
    uint32_t trackEnd = CAMath::Min(compressor.mMemory->nStoredTracks, trackStart + tracksPerWarp);
    if (trackStart >= compressor.mMemory->nStoredTracks) {
      trackStart = 0;
      trackEnd = 0;
    }

    uint32_t tracksOffset = calculateWarpOffsets(smem, compressor.mPtrs.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);

    for (uint32_t i = trackStart; i < trackEnd; i += nLanes) {
      uint32_t nTrackClusters = 0;
      uint32_t srcOffset = 0;

      if (i + iLane < trackEnd) {
        nTrackClusters = compressor.mPtrs.nTrackClusters[i + iLane];
        srcOffset = compressor.mAttachedClusterFirstIndex[i + iLane];
      }
      smem.unbuffered.sizes[iWarp][iLane] = nTrackClusters;
      smem.unbuffered.srcOffsets[iWarp][iLane] = srcOffset;

      uint32_t elems = (i + nLanes < trackEnd) ? nLanes : (trackEnd - i);

      for (uint32_t j = 0; j < elems; j++) {
        nTrackClusters = smem.unbuffered.sizes[iWarp][j];
        srcOffset = smem.unbuffered.srcOffsets[iWarp][j];
        uint32_t idx = i + j;
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
GPUdii() void GPUTPCCompressionGatherKernels::gatherBuffered(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{

  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;

  int32_t nWarps = nThreads / GPUCA_WARP_SIZE;
  int32_t iWarp = iThread / GPUCA_WARP_SIZE;

  int32_t nGlobalWarps = nWarps * nBlocks;
  int32_t iGlobalWarp = nWarps * iBlock + iWarp;

  int32_t nLanes = GPUCA_WARP_SIZE;
  int32_t iLane = iThread % GPUCA_WARP_SIZE;

  auto& input = compressor.mPtrs;
  auto* output = compressor.mOutput;

  uint32_t nRows = compressor.NSECTORS * GPUTPCGeometry::NROWS;
  uint32_t rowsPerWarp = (nRows + nGlobalWarps - 1) / nGlobalWarps;
  uint32_t rowStart = rowsPerWarp * iGlobalWarp;
  uint32_t rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);
  if (rowStart >= nRows) {
    rowStart = 0;
    rowEnd = 0;
  }
  rowsPerWarp = rowEnd - rowStart;

  uint32_t rowsOffset = calculateWarpOffsets(smem, input.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);

  uint32_t nStoredTracks = compressor.mMemory->nStoredTracks;
  uint32_t tracksPerWarp = (nStoredTracks + nGlobalWarps - 1) / nGlobalWarps;
  uint32_t trackStart = tracksPerWarp * iGlobalWarp;
  uint32_t trackEnd = CAMath::Min(nStoredTracks, trackStart + tracksPerWarp);
  if (trackStart >= nStoredTracks) {
    trackStart = 0;
    trackEnd = 0;
  }
  tracksPerWarp = trackEnd - trackStart;

  uint32_t tracksOffset = calculateWarpOffsets(smem, input.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);

  if (iBlock == 0) {
    compressorMemcpyBasic(output->nSliceRowClusters, input.nSliceRowClusters, compressor.NSECTORS * GPUTPCGeometry::NROWS, nThreads, iThread);
    compressorMemcpyBasic(output->nTrackClusters, input.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->qPtA, input.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->rowA, input.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->sliceA, input.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->timeA, input.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->padA, input.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);
  }

  const uint32_t* clusterOffsets = &clusters->clusterOffset[0][0] + rowStart;
  const uint32_t* nSectorRowClusters = input.nSliceRowClusters + rowStart;

  auto* buf = smem.getBuffer<V>(iWarp);

  compressorMemcpyBuffered(buf, output->qTotU + rowsOffset, input.qTotU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->qMaxU + rowsOffset, input.qMaxU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->flagsU + rowsOffset, input.flagsU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->padDiffU + rowsOffset, input.padDiffU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->timeDiffU + rowsOffset, input.timeDiffU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->sigmaPadU + rowsOffset, input.sigmaPadU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  compressorMemcpyBuffered(buf, output->sigmaTimeU + rowsOffset, input.sigmaTimeU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);

  const uint16_t* nTrackClustersPtr = input.nTrackClusters + trackStart;
  const uint32_t* aClsFstIdx = compressor.mAttachedClusterFirstIndex + trackStart;

  compressorMemcpyBuffered(buf, output->qTotA + tracksOffset, input.qTotA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->qMaxA + tracksOffset, input.qMaxA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->flagsA + tracksOffset, input.flagsA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaPadA + tracksOffset, input.sigmaPadA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
  compressorMemcpyBuffered(buf, output->sigmaTimeA + tracksOffset, input.sigmaTimeA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);

  // First index stored with track
  uint32_t tracksOffsetDiff = tracksOffset - trackStart;
  compressorMemcpyBuffered(buf, output->rowDiffA + tracksOffsetDiff, input.rowDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->sliceLegDiffA + tracksOffsetDiff, input.sliceLegDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->padResA + tracksOffsetDiff, input.padResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  compressorMemcpyBuffered(buf, output->timeResA + tracksOffsetDiff, input.timeResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
}

GPUdii() void GPUTPCCompressionGatherKernels::gatherMulti(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  GPUTPCCompression& GPUrestrict() compressor = processors.tpcCompressor;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() clusters = processors.ioPtrs.clustersNative;
  const auto& input = compressor.mPtrs;
  auto* output = compressor.mOutput;

  const int32_t nWarps = nThreads / GPUCA_WARP_SIZE;
  const int32_t iWarp = iThread / GPUCA_WARP_SIZE;
  const int32_t nLanes = GPUCA_WARP_SIZE;
  const int32_t iLane = iThread % GPUCA_WARP_SIZE;
  auto* buf = smem.getBuffer<Vec128>(iWarp);

  if (iBlock == 0) {
    compressorMemcpyBasic(output->nSliceRowClusters, input.nSliceRowClusters, compressor.NSECTORS * GPUTPCGeometry::NROWS, nThreads, iThread);
    compressorMemcpyBasic(output->nTrackClusters, input.nTrackClusters, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->qPtA, input.qPtA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->rowA, input.rowA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->sliceA, input.sliceA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->timeA, input.timeA, compressor.mMemory->nStoredTracks, nThreads, iThread);
    compressorMemcpyBasic(output->padA, input.padA, compressor.mMemory->nStoredTracks, nThreads, iThread);
  } else if (iBlock & 1) {
    const uint32_t nGlobalWarps = nWarps * (nBlocks - 1) / 2;
    const uint32_t iGlobalWarp = nWarps * (iBlock - 1) / 2 + iWarp;

    const uint32_t nRows = compressor.NSECTORS * GPUTPCGeometry::NROWS;
    uint32_t rowsPerWarp = (nRows + nGlobalWarps - 1) / nGlobalWarps;
    uint32_t rowStart = rowsPerWarp * iGlobalWarp;
    uint32_t rowEnd = CAMath::Min(nRows, rowStart + rowsPerWarp);
    if (rowStart >= nRows) {
      rowStart = 0;
      rowEnd = 0;
    }
    rowsPerWarp = rowEnd - rowStart;

    const uint32_t rowsOffset = calculateWarpOffsets(smem, input.nSliceRowClusters, rowStart, rowEnd, nWarps, iWarp, nLanes, iLane);
    const uint32_t* clusterOffsets = &clusters->clusterOffset[0][0] + rowStart;
    const uint32_t* nSectorRowClusters = input.nSliceRowClusters + rowStart;

    compressorMemcpyBuffered(buf, output->qTotU + rowsOffset, input.qTotU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->qMaxU + rowsOffset, input.qMaxU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->flagsU + rowsOffset, input.flagsU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->padDiffU + rowsOffset, input.padDiffU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->timeDiffU + rowsOffset, input.timeDiffU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->sigmaPadU + rowsOffset, input.sigmaPadU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
    compressorMemcpyBuffered(buf, output->sigmaTimeU + rowsOffset, input.sigmaTimeU, nSectorRowClusters, clusterOffsets, rowsPerWarp, nLanes, iLane, 0, compressor.mMaxClusterFactorBase1024);
  } else {
    const uint32_t nGlobalWarps = nWarps * (nBlocks - 1) / 2;
    const uint32_t iGlobalWarp = nWarps * (iBlock / 2 - 1) + iWarp;

    const uint32_t nStoredTracks = compressor.mMemory->nStoredTracks;
    uint32_t tracksPerWarp = (nStoredTracks + nGlobalWarps - 1) / nGlobalWarps;
    uint32_t trackStart = tracksPerWarp * iGlobalWarp;
    uint32_t trackEnd = CAMath::Min(nStoredTracks, trackStart + tracksPerWarp);
    if (trackStart >= nStoredTracks) {
      trackStart = 0;
      trackEnd = 0;
    }
    tracksPerWarp = trackEnd - trackStart;

    const uint32_t tracksOffset = calculateWarpOffsets(smem, input.nTrackClusters, trackStart, trackEnd, nWarps, iWarp, nLanes, iLane);
    const uint16_t* nTrackClustersPtr = input.nTrackClusters + trackStart;
    const uint32_t* aClsFstIdx = compressor.mAttachedClusterFirstIndex + trackStart;

    compressorMemcpyBuffered(buf, output->qTotA + tracksOffset, input.qTotA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->qMaxA + tracksOffset, input.qMaxA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->flagsA + tracksOffset, input.flagsA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaPadA + tracksOffset, input.sigmaPadA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);
    compressorMemcpyBuffered(buf, output->sigmaTimeA + tracksOffset, input.sigmaTimeA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 0);

    // First index stored with track
    uint32_t tracksOffsetDiff = tracksOffset - trackStart;
    compressorMemcpyBuffered(buf, output->rowDiffA + tracksOffsetDiff, input.rowDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->sliceLegDiffA + tracksOffsetDiff, input.sliceLegDiffA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->padResA + tracksOffsetDiff, input.padResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
    compressorMemcpyBuffered(buf, output->timeResA + tracksOffsetDiff, input.timeResA, nTrackClustersPtr, aClsFstIdx, tracksPerWarp, nLanes, iLane, 1);
  }
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered32>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec32>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered64>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec64>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::buffered128>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  gatherBuffered<Vec128>(nBlocks, nThreads, iBlock, iThread, smem, processors);
}

template <>
GPUdii() void GPUTPCCompressionGatherKernels::Thread<GPUTPCCompressionGatherKernels::multiBlock>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  gatherMulti(nBlocks, nThreads, iBlock, iThread, smem, processors);
}
