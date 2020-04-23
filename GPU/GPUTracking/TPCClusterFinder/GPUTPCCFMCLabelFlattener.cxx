// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCFMCLabelFlattener.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFMCLabelFlattener.h"

#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

#if !defined(GPUCA_GPUCODE)
void GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(
  GPUTPCClusterFinder& cls,
  GPUTPCLinearLabels& labels)
{
  uint headerOffset = labels.header.size();
  uint dataOffset = labels.data.size();

  for (Row row = 0; row < GPUCA_ROW_COUNT; row++) {
    cls.mPlabelHeaderOffset[row] = headerOffset;
    headerOffset += cls.mPclusterInRow[row];

    cls.mPlabelDataOffset[row] = dataOffset;
    if (cls.mPclusterInRow[row] > 0) {
      auto& lastInterim = cls.mPlabelsByRow[cls.mNMaxClusterPerRow * row + cls.mPclusterInRow[row] - 1];
      uint labelsInRow = lastInterim.offset + lastInterim.labels.size();
      dataOffset += labelsInRow;
    }
  }

  labels.header.resize(headerOffset);
  labels.data.resize(dataOffset);
}
#endif

template <>
GPUd() void GPUTPCCFMCLabelFlattener::Thread<GPUTPCCFMCLabelFlattener::setRowOffsets>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory&, processorType& clusterer)
{
#if !defined(GPUCA_GPUCODE)
  Row row = get_global_id(0);

  uint clusterInRow = clusterer.mPclusterInRow[row];
  uint offset = 0;

  for (uint i = 0; i < clusterInRow; i++) {
    auto& interim = clusterer.mPlabelsByRow[row * clusterer.mNMaxClusterPerRow + i];
    interim.offset = offset;
    offset += interim.labels.size();
  }
#endif
}

template <>
GPUd() void GPUTPCCFMCLabelFlattener::Thread<GPUTPCCFMCLabelFlattener::flatten>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory&, processorType& clusterer, uint row, GPUTPCLinearLabels* out)
{
#if !defined(GPUCA_GPUCODE)
  uint idx = get_global_id(0);

  GPUTPCClusterMCInterim& interim = clusterer.mPlabelsByRow[row * clusterer.mNMaxClusterPerRow + idx];

  uint headerOffset = clusterer.mPlabelHeaderOffset[row] + idx;
  uint dataOffset = clusterer.mPlabelDataOffset[row] + interim.offset;

  assert(dataOffset + interim.labels.size() <= out->data.size());

  out->header[headerOffset] = dataOffset;
  std::copy(interim.labels.cbegin(), interim.labels.cend(), out->data.begin() + dataOffset);

  interim = {}; // ensure interim labels are destroyed to prevent memleak
#endif
}
