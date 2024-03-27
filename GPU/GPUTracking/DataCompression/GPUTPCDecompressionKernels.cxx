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
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step0attached>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int trackStart, int trackEnd)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  const GPUParam& GPUrestrict() param = processors.param;

  const unsigned int maxTime = (param.par.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;

  for (unsigned int i = trackStart + get_global_id(0); i < trackEnd; i += get_global_size(0)) {
    decompressTrack(cmprClusters, param, maxTime, i, decompressor.mAttachedClustersOffsets[i], decompressor);
  }
}

GPUdii() void GPUTPCDecompressionKernels::decompressTrack(CompressedClusters& cmprClusters, const GPUParam& param, const unsigned int maxTime, const unsigned int trackIndex, unsigned int clusterOffset, GPUTPCDecompression& decompressor)
{
  float zOffset = 0;
  unsigned int slice = cmprClusters.sliceA[trackIndex];
  unsigned int row = cmprClusters.rowA[trackIndex];
  GPUTPCCompressionTrackModel track;
  unsigned int clusterIndex;
  for (clusterIndex = 0; clusterIndex < cmprClusters.nTrackClusters[trackIndex]; clusterIndex++) {
    unsigned int pad = 0, time = 0;
    if (clusterIndex != 0) {
      unsigned char tmpSlice = cmprClusters.sliceLegDiffA[clusterOffset - trackIndex - 1];
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
      unsigned int timeTmp = cmprClusters.timeResA[clusterOffset - trackIndex - 1];
      if (timeTmp & 800000) {
        timeTmp |= 0xFF000000;
      }
      time = timeTmp + ClusterNative::packTime(CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(slice, track.Z() + zOffset)));
      float tmpPad = CAMath::Max(0.f, CAMath::Min((float)param.tpcGeometry.NPads(GPUCA_ROW_COUNT - 1), param.tpcGeometry.LinearY2Pad(slice, row, track.Y())));
      pad = cmprClusters.padResA[clusterOffset - trackIndex - 1] + ClusterNative::packPad(tmpPad);
      time = time & 0xFFFFFF;
      pad = (unsigned short)pad;
      if (pad >= param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked) {
        if (pad >= 0xFFFF - 11968) { // Constant 11968 = (2^15 - MAX_PADS(138) * scalePadPacked(64)) / 2
          pad = 0;
        } else {
          pad = param.tpcGeometry.NPads(row) * ClusterNative::scalePadPacked - 1;
        }
      }
      if (param.par.continuousMaxTimeBin > 0 && time >= maxTime) {
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

GPUdii() ClusterNative GPUTPCDecompressionKernels::decompressTrackStore(const o2::tpc::CompressedClusters& cmprClusters, const unsigned int clusterOffset, unsigned int slice, unsigned int row, unsigned int pad, unsigned int time, GPUTPCDecompression& decompressor)
{
  unsigned int tmpBufferIndex = computeLinearTmpBufferIndex(slice, row, decompressor.mMaxNativeClustersPerBuffer);
  unsigned int currentClusterIndex = CAMath::AtomicAdd(decompressor.mNativeClustersIndex + (slice * GPUCA_ROW_COUNT + row), 1u);
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
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step1unattached>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  ClusterNative* GPUrestrict() clusterBuffer = decompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;

  unsigned int* offsets = decompressor.mUnattachedClustersOffsets;
  for (unsigned int i = get_global_id(0); i < GPUCA_NSLICES * GPUCA_ROW_COUNT; i += get_global_size(0)) {
    unsigned int slice = i / GPUCA_ROW_COUNT;
    unsigned int row = i % GPUCA_ROW_COUNT;
    unsigned int tmpBufferIndex = computeLinearTmpBufferIndex(slice, row, decompressor.mMaxNativeClustersPerBuffer);
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[slice][row];
    if (decompressor.mNativeClustersIndex[i] != 0) {
      decompressorMemcpyBasic(buffer, decompressor.mTmpNativeClusters + tmpBufferIndex, decompressor.mNativeClustersIndex[i]);
    }
    ClusterNative* clout = buffer + decompressor.mNativeClustersIndex[i];
    unsigned int end = offsets[i] + ((i >= decompressor.mInputGPU.nSliceRows) ? 0 : decompressor.mInputGPU.nSliceRowClusters[i]);
    decompressHits(cmprClusters, offsets[i], end, clout);
    if (processors.param.rec.tpc.clustersShiftTimebins != 0.f) {
      for (unsigned int k = 0; k < outputAccess->nClusters[slice][row]; k++) {
        auto& cl = buffer[k];
        float t = cl.getTime() + processors.param.rec.tpc.clustersShiftTimebins;
        if (t < 0) {
          t = 0;
        }
        if (processors.param.par.continuousMaxTimeBin > 0 && t > processors.param.par.continuousMaxTimeBin) {
          t = processors.param.par.continuousMaxTimeBin;
        }
        cl.setTime(t);
      }
    }
  }
}

GPUdii() void GPUTPCDecompressionKernels::decompressHits(const o2::tpc::CompressedClusters& cmprClusters, const unsigned int start, const unsigned int end, ClusterNative* clusterNativeBuffer)
{
  unsigned int time = 0;
  unsigned short pad = 0;
  for (unsigned int k = start; k < end; k++) {
    if (cmprClusters.nComppressionModes & GPUSettings::CompressionDifferences) {
      unsigned int timeTmp = cmprClusters.timeDiffU[k];
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
GPUdi() void GPUTPCDecompressionKernels::decompressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, unsigned int size)
{
  for (unsigned int i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::sortPerSectorRow>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  ClusterNative* GPUrestrict() clusterBuffer = processors.tpcDecompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  for (unsigned int i = get_global_id(0); i < GPUCA_NSLICES * GPUCA_ROW_COUNT; i += get_global_size(0)) {
    unsigned int slice = i / GPUCA_ROW_COUNT;
    unsigned int row = i % GPUCA_ROW_COUNT;
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[slice][row];
    GPUCommonAlgorithm::sort(buffer, buffer + outputAccess->nClusters[slice][row]);
  }
}
