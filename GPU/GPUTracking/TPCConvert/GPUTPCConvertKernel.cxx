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

/// \file GPUTPCConvertKernel.cxx
/// \author David Rohr

#include "GPUTPCConvertKernel.h"
#include "GPUConstantMem.h"
#include "TPCFastTransform.h"
#include "GPUTPCClusterData.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCConvertImpl.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCConvertKernel::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& GPUrestrict() processors)
{
  const int32_t iSlice = iBlock / GPUCA_ROW_COUNT;
  const int32_t iRow = iBlock % GPUCA_ROW_COUNT;
  GPUTPCConvert& GPUrestrict() convert = processors.tpcConverter;
  const o2::tpc::ClusterNativeAccess* GPUrestrict() native = processors.ioPtrs.clustersNative;
  GPUTPCClusterData* GPUrestrict() clusters = convert.mMemory->clusters[iSlice];
  const int32_t idOffset = native->clusterOffset[iSlice][iRow];
  const int32_t indexOffset = native->clusterOffset[iSlice][iRow] - native->clusterOffset[iSlice][0];

  for (uint32_t k = get_local_id(0); k < native->nClusters[iSlice][iRow]; k += get_local_size(0)) {
    const auto& GPUrestrict() clin = native->clusters[iSlice][iRow][k];
    float x, y, z;
    GPUTPCConvertImpl::convert(processors, iSlice, iRow, clin.getPad(), clin.getTime(), x, y, z);
    auto& GPUrestrict() clout = clusters[indexOffset + k];
    clout.x = x;
    clout.y = y;
    clout.z = z;
    clout.row = iRow;
    clout.amp = clin.qTot;
    clout.flags = clin.getFlags();
    clout.id = idOffset + k;
#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
    clout.pad = clin.getPad();
    clout.time = clin.getTime();
#endif
  }
}
