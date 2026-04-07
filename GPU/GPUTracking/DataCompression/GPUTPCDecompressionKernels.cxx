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
#include "TPCClusterDecompressionCore.inc"

using namespace o2::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step0attached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t trackStart, int32_t trackEnd)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  const GPUParam& GPUrestrict() param = processors.param;

  const uint32_t maxTime = (param.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;

  for (int32_t i = trackStart + get_global_id(0); i < trackEnd; i += get_global_size(0)) {
    uint32_t offset = decompressor.mAttachedClustersOffsets[i];
    TPCClusterDecompressionCore::decompressTrack(cmprClusters, param, maxTime, i, offset, decompressor);
  }
}

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step1unattached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t sectorStart, int32_t nSectors)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  ClusterNative* GPUrestrict() clusterBuffer = decompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = decompressor.mClusterNativeAccess;
  uint32_t* offsets = decompressor.mUnattachedClustersOffsets;
  for (int32_t i = get_global_id(0); i < GPUCA_NROWS * nSectors; i += get_global_size(0)) {
    uint32_t iRow = i % GPUCA_NROWS;
    uint32_t iSector = sectorStart + (i / GPUCA_NROWS);
    const uint32_t linearIndex = iSector * GPUCA_NROWS + iRow;
    uint32_t tmpBufferIndex = computeLinearTmpBufferIndex(iSector, iRow, decompressor.mMaxNativeClustersPerBuffer);
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[iSector][iRow];
    if (decompressor.mNativeClustersIndex[linearIndex] != 0) {
      decompressorMemcpyBasic(buffer, decompressor.mTmpNativeClusters + tmpBufferIndex, decompressor.mNativeClustersIndex[linearIndex]);
    }
    ClusterNative* clout = buffer + decompressor.mNativeClustersIndex[linearIndex];
    uint32_t end = offsets[linearIndex] + ((linearIndex >= decompressor.mInputGPU.nSliceRows) ? 0 : decompressor.mInputGPU.nSliceRowClusters[linearIndex]);
    TPCClusterDecompressionCore::decompressHits(cmprClusters, offsets[linearIndex], end, clout);
    if (processors.param.rec.tpc.clustersEdgeFixDistance > 0.f) {
      constexpr GPUTPCGeometry geo;
      for (uint32_t k = 0; k < outputAccess->nClusters[iSector][iRow]; k++) {
        auto& cluster = buffer[k];
        if (cluster.getFlags() & ClusterNative::flagEdge) {
          auto padF = cluster.getPad();
          float distEdge = padF < geo.NPads(iRow) / 2 ? padF : geo.NPads(iRow) - 1 - padF;
          if (distEdge > processors.param.rec.tpc.clustersEdgeFixDistance) {
            cluster.setFlags(cluster.getFlags() ^ ClusterNative::flagEdge);
          }
        }
      }
    }
    if (processors.param.rec.tpc.clustersShiftTimebins != 0.f) {
      for (uint32_t k = 0; k < outputAccess->nClusters[iSector][iRow]; k++) {
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

template <typename T>
GPUdi() void GPUTPCDecompressionKernels::decompressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, uint32_t size)
{
  for (uint32_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

GPUdi() bool GPUTPCDecompressionUtilKernels::isClusterKept(const o2::tpc::ClusterNative& cl, const GPUParam& GPUrestrict() param)
{
  return param.tpcCutTimeBin > 0 ? cl.getTime() < param.tpcCutTimeBin : true;
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::countFilteredClusters>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const GPUParam& GPUrestrict() param = processors.param;
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  const ClusterNativeAccess* clusterAccess = decompressor.mClusterNativeAccess;
  for (uint32_t i = get_global_id(0); i < GPUCA_NSECTORS * GPUCA_NROWS; i += get_global_size(0)) {
    uint32_t sector = i / GPUCA_NROWS;
    uint32_t row = i % GPUCA_NROWS;
    for (uint32_t k = 0; k < clusterAccess->nClusters[sector][row]; k++) {
      ClusterNative cl = clusterAccess->clusters[sector][row][k];
      if (isClusterKept(cl, param)) {
        decompressor.mNClusterPerSectorRow[i]++;
      }
    }
  }
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::storeFilteredClusters>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const GPUParam& GPUrestrict() param = processors.param;
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  ClusterNative* GPUrestrict() clusterBuffer = decompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* clusterAccess = decompressor.mClusterNativeAccess;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  for (uint32_t i = get_global_id(0); i < GPUCA_NSECTORS * GPUCA_NROWS; i += get_global_size(0)) {
    uint32_t sector = i / GPUCA_NROWS;
    uint32_t row = i % GPUCA_NROWS;
    uint32_t count = 0;
    for (uint32_t k = 0; k < clusterAccess->nClusters[sector][row]; k++) {
      const ClusterNative cl = clusterAccess->clusters[sector][row][k];
      if (isClusterKept(cl, param)) {
        clusterBuffer[outputAccess->clusterOffset[sector][row] + count] = cl;
        count++;
      }
    }
  }
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::sortPerSectorRow>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  ClusterNative* GPUrestrict() clusterBuffer = processors.tpcDecompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  for (uint32_t i = get_global_id(0); i < GPUCA_NSECTORS * GPUCA_NROWS; i += get_global_size(0)) {
    uint32_t sector = i / GPUCA_NROWS;
    uint32_t row = i % GPUCA_NROWS;
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[sector][row];
    GPUCommonAlgorithm::sort(buffer, buffer + outputAccess->nClusters[sector][row]);
  }
}
