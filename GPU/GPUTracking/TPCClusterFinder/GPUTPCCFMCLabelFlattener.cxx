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
  uint32_t headerOffset = labels.header.size();
  uint32_t dataOffset = labels.data.size();

  cls.mPlabelsHeaderGlobalOffset = headerOffset;
  cls.mPlabelsDataGlobalOffset = dataOffset;

  for (Row row = 0; row < GPUCA_ROW_COUNT; row++) {
    headerOffset += cls.mPclusterInRow[row];
    dataOffset += cls.mPlabelsInRow[row];
  }

  labels.header.resize(headerOffset);
  labels.data.resize(dataOffset);
}
#endif

template <>
GPUd() void GPUTPCCFMCLabelFlattener::Thread<GPUTPCCFMCLabelFlattener::setRowOffsets>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory&, processorType& clusterer)
{
#if !defined(GPUCA_GPUCODE)
  Row row = get_global_id(0);

  uint32_t clusterInRow = clusterer.mPclusterInRow[row];
  uint32_t labelCount = 0;

  for (uint32_t i = 0; i < clusterInRow; i++) {
    auto& interim = clusterer.mPlabelsByRow[row].data[i];
    labelCount += interim.labels.size();
  }

  clusterer.mPlabelsInRow[row] = labelCount;
#endif
}

template <>
GPUd() void GPUTPCCFMCLabelFlattener::Thread<GPUTPCCFMCLabelFlattener::flatten>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory&, processorType& clusterer, GPUTPCLinearLabels* out)
{
#if !defined(GPUCA_GPUCODE)
  uint32_t row = get_global_id(0);

  uint32_t headerOffset = clusterer.mPlabelsHeaderGlobalOffset;
  uint32_t dataOffset = clusterer.mPlabelsDataGlobalOffset;
  for (uint32_t r = 0; r < row; r++) {
    headerOffset += clusterer.mPclusterInRow[r];
    dataOffset += clusterer.mPlabelsInRow[r];
  }

  auto* labels = clusterer.mPlabelsByRow[row].data.data();
  for (uint32_t c = 0; c < clusterer.mPclusterInRow[row]; c++) {
    GPUTPCClusterMCInterim& interim = labels[c];
    assert(dataOffset + interim.labels.size() <= out->data.size());
    out->header[headerOffset] = dataOffset;
    std::copy(interim.labels.cbegin(), interim.labels.cend(), out->data.begin() + dataOffset);

    headerOffset++;
    dataOffset += interim.labels.size();
    interim = {}; // ensure interim labels are destroyed to prevent memleak
  }
#endif
}
