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

/// \file GPUTPCDecompressionKernels.cxx
/// \author Gabriele Cimador

#include "GPUTPCDecompressionKernels.h"
#include "GPULogging.h"
#include "GPUConstantMem.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUCommonAlgorithm.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step0attached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t trackStart, int32_t trackEnd)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  const GPUParam& GPUrestrict() param = processors.param;

  const uint32_t maxTime = (param.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;

  for (int32_t i = trackStart + get_global_id(0); i < trackEnd; i += get_global_size(0)) {
    decompressTrack(cmprClusters, param, maxTime, i, decompressor.mAttachedClustersOffsets[i], decompressor);
  }
}

GPUdii() void GPUTPCDecompressionKernels::decompressTrack(CompressedClusters& cmprClusters, const GPUParam& param, const uint32_t maxTime, const uint32_t trackIndex, uint32_t clusterOffset, GPUTPCDecompression& decompressor)
{
  float zOffset = 0;
  uint32_t slice = cmprClusters.sliceA[trackIndex];
  uint32_t row = cmprClusters.rowA[trackIndex];
  GPUTPCCompressionTrackModel track;
  uint32_t clusterIndex;
  for (clusterIndex = 0; clusterIndex < cmprClusters.nTrackClusters[trackIndex]; clusterIndex++) {
    uint32_t pad = 0, time = 0;
    if (clusterIndex != 0) {
      uint8_t tmpSlice = cmprClusters.sliceLegDiffA[clusterOffset - trackIndex - 1];
      bool changeLeg = (tmpSlice >= GPUCA_NSLICES);
      if (changeLeg) {
        tmpSlice -= GPUCA_NSLICES;
      }
      if (cmprClusters.nComppressionModes & GPUSettings::CompressionDifferences) {
        slice += tmpSlice;
        if (slice >= GPUCA_NSLICES) {
          slice -= GPUCA_NSLICES;
        }
        row += cmprClusters.rowDiffA[clusterOffset - trackIndex - 1];
        if (row >= GPUCA_ROW_COUNT) {
          row -= GPUCA_ROW_COUNT;
        }
      } else {
        slice = tmpSlice;
        row = cmprClusters.rowDiffA[clusterOffset - trackIndex - 1];
      }
      if (changeLeg && track.Mirror()) {
        break;
      }
      if (track.Propagate(param.tpcGeometry.Row2X(row), param.SliceParam[slice].Alpha)) {
        break;
      }
      uint32_t timeTmp = cmprClusters.timeResA[clusterOffset - trackIndex - 1];
      if (timeTmp & 800000) {
        timeTmp |= 0xFF000000;
      }
      time = timeTmp + ClusterNative::packTime(CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(slice, track.Z() + zOffset)));
      float tmpPad = CAMath::Max(0.f, CAMath::Min((float)param.tpcGeometry.NPads(GPUCA_ROW_COUNT - 1), param.tpcGeometry.LinearY2Pad(slice, row, track.Y())));
      pad = cmprClusters.padResA[clusterOffset - trackIndex - 1] + ClusterNative::packPad(tmpPad);
      time = time & 0xFFFFFF;
      pad = (uint16_t)pad;
      if (pad >= param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked) {
        if (pad >= 0xFFFF - 11968) { // Constant 11968 = (2^15 - MAX_PADS(138) * scalePadPacked(64)) / 2
          pad = 0;
        } else {
          pad = param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked - 1;
        }
      }
      if (param.continuousMaxTimeBin > 0 && time >= maxTime) {
        if (time >= 0xFFFFFF - 544768) { // Constant 544768 = (2^23 - LHCMAXBUNCHES(3564) * MAXORBITS(256) * scaleTimePacked(64) / BCPERTIMEBIN(8)) / 2)
          time = 0;
        } else {
          time = maxTime;
        }
      }
    } else {
      time = cmprClusters.timeA[trackIndex];
      pad = cmprClusters.padA[trackIndex];
    }
    const auto cluster = decompressTrackStore(cmprClusters, clusterOffset, slice, row, pad, time, decompressor);
    float y = param.tpcGeometry.LinearPad2Y(slice, row, cluster.getPad());
    float z = param.tpcGeometry.LinearTime2Z(slice, cluster.getTime());
    if (clusterIndex == 0) {
      zOffset = z;
      track.Init(param.tpcGeometry.Row2X(row), y, z - zOffset, param.SliceParam[slice].Alpha, cmprClusters.qPtA[trackIndex], param);
    }
    if (clusterIndex + 1 < cmprClusters.nTrackClusters[trackIndex] && track.Filter(y, z - zOffset, row)) {
      break;
    }
    clusterOffset++;
  }
  clusterOffset += cmprClusters.nTrackClusters[trackIndex] - clusterIndex;
}

GPUdii() ClusterNative GPUTPCDecompressionKernels::decompressTrackStore(const o2::tpc::CompressedClusters& cmprClusters, const uint32_t clusterOffset, uint32_t slice, uint32_t row, uint32_t pad, uint32_t time, GPUTPCDecompression& decompressor)
{
  uint32_t tmpBufferIndex = computeLinearTmpBufferIndex(slice, row, decompressor.mMaxNativeClustersPerBuffer);
  uint32_t currentClusterIndex = CAMath::AtomicAdd(decompressor.mNativeClustersIndex + (slice * GPUCA_ROW_COUNT + row), 1u);
  const ClusterNative c(time, cmprClusters.flagsA[clusterOffset], pad, cmprClusters.sigmaTimeA[clusterOffset], cmprClusters.sigmaPadA[clusterOffset], cmprClusters.qMaxA[clusterOffset], cmprClusters.qTotA[clusterOffset]);
  if (currentClusterIndex < decompressor.mMaxNativeClustersPerBuffer) {
    decompressor.mTmpNativeClusters[tmpBufferIndex + currentClusterIndex] = c;
  } else {
    decompressor.raiseError(GPUErrors::ERROR_DECOMPRESSION_ATTACHED_CLUSTER_OVERFLOW, slice * 1000 + row, currentClusterIndex, decompressor.mMaxNativeClustersPerBuffer);
    CAMath::AtomicExch(decompressor.mNativeClustersIndex + (slice * GPUCA_ROW_COUNT + row), decompressor.mMaxNativeClustersPerBuffer);
  }
  return c;
}

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step1unattached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t sliceStart, int32_t nSlices)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  ClusterNative* GPUrestrict() clusterBuffer = decompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  uint32_t* offsets = decompressor.mUnattachedClustersOffsets;
  for (int32_t i = get_global_id(0); i < GPUCA_ROW_COUNT * nSlices; i += get_global_size(0)) {
    uint32_t iRow = i % GPUCA_ROW_COUNT;
    uint32_t iSlice = sliceStart + (i / GPUCA_ROW_COUNT);
    const uint32_t linearIndex = iSlice * GPUCA_ROW_COUNT + iRow;
    uint32_t tmpBufferIndex = computeLinearTmpBufferIndex(iSlice, iRow, decompressor.mMaxNativeClustersPerBuffer);
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[iSlice][iRow];
    if (decompressor.mNativeClustersIndex[linearIndex] != 0) {
      decompressorMemcpyBasic(buffer, decompressor.mTmpNativeClusters + tmpBufferIndex, decompressor.mNativeClustersIndex[linearIndex]);
    }
    ClusterNative* clout = buffer + decompressor.mNativeClustersIndex[linearIndex];
    uint32_t end = offsets[linearIndex] + ((linearIndex >= decompressor.mInputGPU.nSliceRows) ? 0 : decompressor.mInputGPU.nSliceRowClusters[linearIndex]);
    decompressHits(cmprClusters, offsets[linearIndex], end, clout);
    if (processors.param.rec.tpc.clustersShiftTimebins != 0.f) {
      for (uint32_t k = 0; k < outputAccess->nClusters[iSlice][iRow]; k++) {
        auto& cl = buffer[k];
        float t = cl.getTime() + processors.param.rec.tpc.clustersShiftTimebins;
        if (t < 0) {
          t = 0;
        }
        if (processors.param.continuousMaxTimeBin > 0 && t > processors.param.continuousMaxTimeBin) {
          t = processors.param.continuousMaxTimeBin;
        }
        cl.setTime(t);
      }
    }
  }
}

GPUdii() void GPUTPCDecompressionKernels::decompressHits(const o2::tpc::CompressedClusters& cmprClusters, const uint32_t start, const uint32_t end, ClusterNative* clusterNativeBuffer)
{
  uint32_t time = 0;
  uint16_t pad = 0;
  for (uint32_t k = start; k < end; k++) {
    if (cmprClusters.nComppressionModes & GPUSettings::CompressionDifferences) {
      uint32_t timeTmp = cmprClusters.timeDiffU[k];
      if (timeTmp & 800000) {
        timeTmp |= 0xFF000000;
      }
      time += timeTmp;
      pad += cmprClusters.padDiffU[k];
    } else {
      time = cmprClusters.timeDiffU[k];
      pad = cmprClusters.padDiffU[k];
    }
    *(clusterNativeBuffer++) = ClusterNative(time, cmprClusters.flagsU[k], pad, cmprClusters.sigmaTimeU[k], cmprClusters.sigmaPadU[k], cmprClusters.qMaxU[k], cmprClusters.qTotU[k]);
  }
}

template <typename T>
GPUdi() void GPUTPCDecompressionKernels::decompressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, uint32_t size)
{
  for (uint32_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::sortPerSectorRow>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  ClusterNative* GPUrestrict() clusterBuffer = processors.tpcDecompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  for (uint32_t i = get_global_id(0); i < GPUCA_NSLICES * GPUCA_ROW_COUNT; i += get_global_size(0)) {
    uint32_t slice = i / GPUCA_ROW_COUNT;
    uint32_t row = i % GPUCA_ROW_COUNT;
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[slice][row];
    GPUCommonAlgorithm::sort(buffer, buffer + outputAccess->nClusters[slice][row]);
  }
}
