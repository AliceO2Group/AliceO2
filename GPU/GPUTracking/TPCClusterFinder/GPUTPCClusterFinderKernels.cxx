// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterFinderKernels.cxx
/// \author David Rohr

#include "GPUTPCClusterFinderKernels.h"
#include "GPUConstantMem.h"
#include "GPUO2DataTypes.h"
#include "GPUCommonAlgorithm.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "CfConsts.h"
#include "CfUtils.h"
#include "ChargeMapFiller.h"
#include "PeakFinder.h"
#include "ClusterAccumulator.h"
#include "Clusterizer.h"
#include "NoiseSuppression.h"
#include "Deconvolution.h"
#include "StreamCompaction.h"
#include "DecodeZS.h"

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::fillChargeMap>(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  ChargeMapFiller::fillChargeMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, chargeMap, clusterer.mPmemory->counters.nDigits);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::resetMaps>(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  ChargeMapFiller::resetMapsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPdigits, chargeMap, isPeakMap);
}

template <>
GPUd() void GPUTPCClusterFinderKernels::Thread<GPUTPCClusterFinderKernels::decodeZS>(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCSharedMemory& smem, processorType& clusterer)
{
  DecodeZS::decode(clusterer, smem, nBlocks, nThreads, iBlock, iThread);
}
