// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
GPUdii() void GPUTPCConvertKernel::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  const int iSlice = iBlock / GPUCA_ROW_COUNT;
  const int iRow = iBlock % GPUCA_ROW_COUNT;
  GPUTPCConvert& convert = processors.tpcConverter;
  const o2::tpc::ClusterNativeAccess* native = convert.mClustersNative;
  GPUTPCClusterData* clusters = convert.mMemory->clusters[iSlice];
  const int idOffset = native->clusterOffset[iSlice][iRow];
  const int indexOffset = native->clusterOffset[iSlice][iRow] - native->clusterOffset[iSlice][0];

  for (unsigned int k = get_local_id(0); k < native->nClusters[iSlice][iRow]; k += get_local_size(0)) {
    const auto& clin = native->clusters[iSlice][iRow][k];
    float x, y, z;
    GPUTPCConvertImpl::convert(processors, iSlice, iRow, clin.getPad(), clin.getTime(), x, y, z);
    auto& clout = clusters[indexOffset + k];
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
