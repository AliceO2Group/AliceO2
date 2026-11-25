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

using namespace o2::gpu;

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::clearIds>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t)
{
  for (uint32_t i = iBlock * nThreads + iThread; i < GPUTPCGeometry::NSECTORS * merger.NMaxSingleSectorTracks(); i++) {
    merger.TrackIDs()[i] = -1;
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::sectorTracks1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iThread) {
    return;
  }
  const uint32_t offset = parameter ? merger.Memory()->nUnpackedTracks : 0;
  const uint32_t n = parameter ? merger.Memory()->nUnpackedTracksGlobal : merger.Memory()->nUnpackedTracks;
  const GPUTPCGMSectorTrack* GPUrestrict() tracks = merger.SectorTrackInfos() + offset;
  for (uint32_t iSector = iBlock; iSector < GPUTPCGeometry::NSECTORS; iSector += nBlocks) {
    uint32_t sectorOffset = 0;
    for (uint32_t j = 0; j < n; j++) {
      sectorOffset += tracks[j].Sector() < iSector;
    }
    int32_t* GPUrestrict() tmp = merger.TmpSortMemory() + sectorOffset; // Slice of this sector, so that tmp is sorted by sector over the full range
    uint32_t nSector = 0;
    for (uint32_t j = 0; j < n; j++) {
      if (tracks[j].Sector() == iSector) {
        tmp[nSector++] = j;
      }
    }
    GPUCommonAlgorithm::sort(tmp, tmp + nSector, [tracks, parameter](const int32_t& aa, const int32_t& bb) {
      const auto& a = tracks[aa];
      const auto& b = tracks[bb];
      return !parameter         ? (a.OrigTrackId() < b.OrigTrackId()) // Local tracks: order of the sector tracker, which is sorted in deterministic mode
             : (a.X() != b.X()) ? (a.X() < b.X())
             : (a.Y() != b.Y()) ? (a.Y() < b.Y())
                                : (a.Z() < b.Z());
    });
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::sectorTracks2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iBlock) {
    return;
  }
  const uint32_t offset = parameter ? merger.Memory()->nUnpackedTracks : 0;
  const uint32_t n = parameter ? merger.Memory()->nUnpackedTracksGlobal : merger.Memory()->nUnpackedTracks;
  int32_t* GPUrestrict() tmp = merger.TmpSortMemory();
  GPUTPCGMSectorTrack* GPUrestrict() tracks = merger.SectorTrackInfos() + offset;
  if (iThread == 0) {
    for (uint32_t j = 0; j < n; j++) {
      if (tmp[j] >= 0 && tmp[j] != (int32_t)j) {
        int32_t firstIdx = j;
        auto firstItem = tracks[firstIdx];
        int32_t currIdx = firstIdx;
        int32_t sourceIdx = tmp[currIdx];
        do {
          tmp[currIdx] = -1;
          tracks[currIdx] = tracks[sourceIdx];
          currIdx = sourceIdx;
          sourceIdx = tmp[currIdx];
        } while (sourceIdx != firstIdx);
        tmp[currIdx] = -1;
        tracks[currIdx] = firstItem;
      }
    }
  }
  if (parameter) {
    return;
  }
  GPUbarrier();
  for (uint32_t j = iThread; j < n; j += nThreads) {
    merger.TrackIDs()[tracks[j].Sector() * merger.NMaxSingleSectorTracks() + tracks[j].OrigTrack(&merger)->LocalTrackId()] = j;
  }
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::mergedTracks1>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iThread || iBlock) {
    return;
  }
  int32_t* GPUrestrict() tmp = merger.TmpSortMemory();
  const int32_t n = merger.NMergedTracks();
  for (int32_t j = 0; j < n; j++) {
    tmp[j] = j;
  }
  GPUCommonAlgorithm::sortDeviceDynamic(tmp, tmp + n, [&merger](const int32_t& aa, const int32_t& bb) {
    const GPUTPCGMMergedTrack& a = merger.MergedTracks()[aa];
    const GPUTPCGMMergedTrack& b = merger.MergedTracks()[bb];
    return (a.GetAlpha() != b.GetAlpha()) ? (a.GetAlpha() < b.GetAlpha()) : ((a.GetParam().GetX() != b.GetParam().GetX()) ? (a.GetParam().GetX() < b.GetParam().GetX()) : ((a.GetParam().GetY() != b.GetParam().GetY()) ? (a.GetParam().GetY() < b.GetParam().GetY()) : (a.GetParam().GetZ() < b.GetParam().GetZ())));
  });
}

template <>
GPUdii() void GPUTPCGlobalDebugSortKernels::Thread<GPUTPCGlobalDebugSortKernels::mergedTracks2>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() merger, int8_t parameter)
{
  if (iBlock) {
    return;
  }
  const int32_t n = merger.NMergedTracks();
  int32_t* GPUrestrict() tmp = merger.TmpSortMemory();
  int32_t* GPUrestrict() tmp2 = tmp + n;
  if (iThread == 0) {
    for (int32_t j = 0; j < n; j++) {
      if (tmp[j] == j) {
        tmp2[j] = j;
      } else if (tmp[j] >= 0) {
        int32_t firstIdx = j;
        auto firstItem = merger.MergedTracks()[firstIdx];
        int32_t currIdx = firstIdx;
        int32_t sourceIdx = tmp[currIdx];
        tmp2[sourceIdx] = currIdx;
        do {
          tmp[currIdx] = -1;
          merger.MergedTracks()[currIdx] = merger.MergedTracks()[sourceIdx];
          currIdx = sourceIdx;
          sourceIdx = tmp[currIdx];
          tmp2[sourceIdx] = currIdx;
        } while (sourceIdx != firstIdx);
        tmp[currIdx] = -1;
        merger.MergedTracks()[currIdx] = firstItem;
      }
    }
    for (int32_t j = 0; j < n; j++) {
      if (merger.MergedTracks()[j].PrevSegment() >= 0) {
        merger.MergedTracks()[j].SetPrevSegment(tmp2[merger.MergedTracks()[j].PrevSegment()]);
      }
    }
  }
  GPUbarrier();
  for (uint32_t i = 0; i < 2 * GPUTPCGeometry::NSECTORS; i++) {
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
