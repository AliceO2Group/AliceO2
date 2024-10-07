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

/// \file GPUTPCGlobalDebugSortKernels.cxx
/// \author David Rohr

#include "GPUParam.h"
#include "GPUProcessor.h"
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCGlobalDebugSortKernels.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <stdexcept>
#endif

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::clearIds>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t)
{
  for (int32_t i = iBlock * nThreads + iThread; i < GPUCA_NSLICES * merger.NMaxSingleSliceTracks(); i++) {
    merger.TrackIDs()[i] = -1;
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::sectorTracks>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iThread) {
    return;
  }
  int32_t iStart = parameter ? GPUCA_NSLICES : 0;
  int32_t iEnd = iStart + GPUCA_NSLICES;
  for (int32_t i = iStart + iBlock; i < iEnd; i += nBlocks) {
    const int32_t offset = merger.SliceTrackInfoFirst(i);
    int32_t* GPUrestrict() tmp = merger.TmpSortMemory() + offset;
    const int32_t n = merger.SliceTrackInfoLast(i) - merger.SliceTrackInfoFirst(i);
    if (n < 2) {
      continue;
    }
    for (int32_t j = 0; j < n; j++) {
      tmp[j] = j;
    }
    GPUCommonAlgorithm::sort(tmp, tmp + n, [&merger, offset](const int32_t& aa, const int32_t& bb) {
      const auto& a = merger.SliceTrackInfos()[offset + aa];
      const auto& b = merger.SliceTrackInfos()[offset + bb];
      return (a.X() != b.X()) ? (a.X() < b.X()) : (a.Y() != b.Y()) ? (a.Y() < b.Y())
                                                                   : (a.Z() < b.Z());
    });
    for (int32_t j = 0; j < n; j++) {
      if (tmp[j] >= 0 && tmp[j] != j) {
        auto getTrackIDIndex = [&merger](const int32_t iSlice, const int32_t iTrack) {
          const int32_t kEnd = merger.NMaxSingleSliceTracks();
          for (int32_t k = 0; k < kEnd; k++) {
            if (merger.TrackIDs()[iSlice * merger.NMaxSingleSliceTracks() + k] == iTrack) {
              return k;
            }
          }
#ifndef GPUCA_GPUCODE_DEVICE
          throw std::runtime_error("Internal error, track id missing");
#endif
          return -1;
        };
        int32_t firstIdx = j;
        auto firstItem = merger.SliceTrackInfos()[offset + firstIdx];
        int32_t firstTrackIDIndex = parameter ? 0 : getTrackIDIndex(i, offset + firstIdx);
        int32_t currIdx = firstIdx;
        int32_t sourceIdx = tmp[currIdx];
        do {
          tmp[currIdx] = -1;
          merger.SliceTrackInfos()[offset + currIdx] = merger.SliceTrackInfos()[offset + sourceIdx];
          if (!parameter) {
            merger.TrackIDs()[i * merger.NMaxSingleSliceTracks() + getTrackIDIndex(i, offset + sourceIdx)] = offset + currIdx;
          }
          currIdx = sourceIdx;
          sourceIdx = tmp[currIdx];
        } while (sourceIdx != firstIdx);
        tmp[currIdx] = -1;
        merger.SliceTrackInfos()[offset + currIdx] = firstItem;
        if (!parameter) {
          merger.TrackIDs()[i * merger.NMaxSingleSliceTracks() + firstTrackIDIndex] = offset + currIdx;
        }
      }
    }
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::globalTracks1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iThread || iBlock) {
    return;
  }
  int32_t* GPUrestrict() tmp = merger.TmpSortMemory();
  const int32_t n = merger.NOutputTracks();
  for (int32_t j = 0; j < n; j++) {
    tmp[j] = j;
  }
  GPUCommonAlgorithm::sortDeviceDynamic(tmp, tmp + n, [&merger](const int32_t& aa, const int32_t& bb) {
    const GPUTPCGMMergedTrack& a = merger.OutputTracks()[aa];
    const GPUTPCGMMergedTrack& b = merger.OutputTracks()[bb];
    return (a.GetAlpha() != b.GetAlpha()) ? (a.GetAlpha() < b.GetAlpha()) : (a.GetParam().GetX() != b.GetParam().GetX()) ? (a.GetParam().GetX() < b.GetParam().GetX()) : (a.GetParam().GetY() != b.GetParam().GetY()) ? (a.GetParam().GetY() < b.GetParam().GetY()) : (a.GetParam().GetZ() < b.GetParam().GetZ());
  });
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::globalTracks2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iBlock) {
    return;
  }
  const int32_t n = merger.NOutputTracks();
  int32_t* GPUrestrict() tmp = merger.TmpSortMemory();
  int32_t* GPUrestrict() tmp2 = tmp + n;
  if (iThread == 0) {
    for (int32_t j = 0; j < n; j++) {
      if (tmp[j] == j) {
        tmp2[j] = j;
      } else if (tmp[j] >= 0) {
        int32_t firstIdx = j;
        auto firstItem = merger.OutputTracks()[firstIdx];
        int32_t currIdx = firstIdx;
        int32_t sourceIdx = tmp[currIdx];
        tmp2[sourceIdx] = currIdx;
        do {
          tmp[currIdx] = -1;
          merger.OutputTracks()[currIdx] = merger.OutputTracks()[sourceIdx];
          currIdx = sourceIdx;
          sourceIdx = tmp[currIdx];
          tmp2[sourceIdx] = currIdx;
        } while (sourceIdx != firstIdx);
        tmp[currIdx] = -1;
        merger.OutputTracks()[currIdx] = firstItem;
      }
    }
  }
  GPUbarrier();
  for (int32_t i = 0; i < 2 * GPUCA_NSLICES; i++) {
    for (uint32_t k = iThread; k < merger.TmpCounter()[i]; k += nThreads) {
      merger.BorderTracks(i)[k].SetTrackID(tmp2[merger.BorderTracks(i)[k].TrackID()]);
    }
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::borderTracks>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iThread) {
    return;
  }
  auto* borderTracks = merger.BorderTracks(iBlock);
  const uint32_t n = merger.TmpCounter()[iBlock];
  GPUCommonAlgorithm::sortDeviceDynamic(borderTracks, borderTracks + n, [](const GPUTPCGMBorderTrack& a, const GPUTPCGMBorderTrack& b) {
    return (a.TrackID() < b.TrackID());
  });
}
