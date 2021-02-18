// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCFClusterizer.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFClusterizer.h"

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFClusterizer::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(
    MCLabelAccumulator labelAcc(clusterer));

  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterer.mPclusterByRow);
}

GPUdii() void GPUTPCCFClusterizer::computeClustersImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                       const CfFragment& fragment,
                                                       GPUSharedMemory& smem,
                                                       const Array2D<PackedCharge>& chargeMap,
                                                       const ChargePos* filteredPeakPositions,
                                                       MCLabelAccumulator* labelAcc,
                                                       uint clusternum,
                                                       uint maxClusterPerRow,
                                                       uint* clusterInRow,
                                                       tpc::ClusterNative* clusterByRow)
{
  uint idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last cluster but discard the result.
  ChargePos pos = filteredPeakPositions[CAMath::Min(idx, clusternum - 1)];
  Charge charge = chargeMap[pos].unpack();

  ClusterAccumulator pc;
  CPU_ONLY(labelAcc->collect(pos, charge));

  buildCluster(
    chargeMap,
    pos,
    smem.posBcast,
    smem.buf,
    smem.innerAboveThreshold,
    &pc,
    labelAcc);

  if (idx >= clusternum || fragment.isOverlap(pos.time())) {
    return;
  }
  pc.finalize(pos, charge, fragment.start);

  tpc::ClusterNative myCluster;
  pc.toNative(pos, charge, myCluster);

#if defined(CUT_QTOT)
  bool aboveQTotCutoff = (pc.Q > QTOT_CUTOFF);
#else
  bool aboveQTotCutoff = true;
#endif

  if (!aboveQTotCutoff) {
    return;
  }

  uint rowIndex = sortIntoBuckets(
    myCluster,
    pos.row(),
    maxClusterPerRow,
    clusterInRow,
    clusterByRow);
  static_cast<void>(rowIndex); // Avoid unused varible warning on GPU.

  CPU_ONLY(labelAcc->commit(pos.row(), rowIndex, maxClusterPerRow));
}

GPUdii() void GPUTPCCFClusterizer::updateClusterInner(
  ushort lid,
  ushort N,
  const PackedCharge* buf,
  const ChargePos& pos,
  ClusterAccumulator* cluster,
  MCLabelAccumulator* labelAcc,
  uchar* innerAboveThreshold)
{
  uchar aboveThreshold = 0;

  GPUCA_UNROLL(U(), U())
  for (ushort i = 0; i < N; i++) {
    Delta2 d = CfConsts::InnerNeighbors[i];

    PackedCharge p = buf[N * lid + i];

    Charge q = cluster->updateInner(p, d);

    CPU_ONLY(
      labelAcc->collect(pos.delta(d), q));

    aboveThreshold |= (uchar(q > CHARGE_THRESHOLD) << i);
  }

  innerAboveThreshold[lid] = aboveThreshold;

  GPUbarrier();
}

GPUdii() void GPUTPCCFClusterizer::updateClusterOuter(
  ushort lid,
  ushort N,
  ushort M,
  ushort offset,
  const PackedCharge* buf,
  const ChargePos& pos,
  ClusterAccumulator* cluster,
  MCLabelAccumulator* labelAcc)
{
  GPUCA_UNROLL(U(), U())
  for (ushort i = offset; i < M + offset; i++) {
    PackedCharge p = buf[N * lid + i];

    Delta2 d = CfConsts::OuterNeighbors[i];

    Charge q = cluster->updateOuter(p, d);
    static_cast<void>(q); // Avoid unused varible warning on GPU.

    CPU_ONLY(
      labelAcc->collect(pos.delta(d), q));
  }
}

GPUdii() void GPUTPCCFClusterizer::buildCluster(
  const Array2D<PackedCharge>& chargeMap,
  ChargePos pos,
  ChargePos* posBcast,
  PackedCharge* buf,
  uchar* innerAboveThreshold,
  ClusterAccumulator* myCluster,
  MCLabelAccumulator* labelAcc)
{
  ushort ll = get_local_id(0);

  posBcast[ll] = pos;
  GPUbarrier();

  CfUtils::blockLoad<PackedCharge>(
    chargeMap,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    8,
    CfConsts::InnerNeighbors,
    posBcast,
    buf);
  updateClusterInner(
    ll,
    8,
    buf,
    pos,
    myCluster,
    labelAcc,
    innerAboveThreshold);

  ushort wgSizeHalf = (SCRATCH_PAD_WORK_GROUP_SIZE + 1) / 2;

  bool inGroup1 = ll < wgSizeHalf;

  ushort llhalf = (inGroup1) ? ll : (ll - wgSizeHalf);

  CfUtils::condBlockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    CfConsts::OuterNeighbors,
    posBcast,
    innerAboveThreshold,
    buf);

  if (inGroup1) {
    updateClusterOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      pos,
      myCluster,
      labelAcc);
  }

#if defined(GPUCA_GPUCODE)
  CfUtils::condBlockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    CfConsts::OuterNeighbors,
    posBcast + wgSizeHalf,
    innerAboveThreshold + wgSizeHalf,
    buf);
  if (!inGroup1) {
    updateClusterOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      pos,
      myCluster,
      labelAcc);
  }
#endif
}

GPUd() uint GPUTPCCFClusterizer::sortIntoBuckets(const tpc::ClusterNative& cluster, uint row, uint maxElemsPerBucket, uint* elemsInBucket, tpc::ClusterNative* buckets)
{
  uint index = CAMath::AtomicAdd(&elemsInBucket[row], 1u);
  if (index < maxElemsPerBucket) {
    buckets[maxElemsPerBucket * row + index] = cluster;
  }
  return index;
}
