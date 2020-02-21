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

using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUTPCCFClusterizer::Thread<GPUTPCCFClusterizer::computeClusters>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, clusterer.mPfilteredPeaks, clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterer.mPclusterByRow);
}

GPUd() void GPUTPCCFClusterizer::computeClustersImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                     const Array2D<PackedCharge>& chargeMap,
                                                     const deprecated::Digit* digits,
                                                     uint clusternum,
                                                     uint maxClusterPerRow,
                                                     uint* clusterInRow,
                                                     tpc::ClusterNative* clusterByRow)
{
  uint idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last cluster but discard the result.
  deprecated::Digit myDigit = digits[CAMath::Min(idx, clusternum - 1)];

  ChargePos pos(myDigit);

  ClusterAccumulator pc;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  /* #if defined(BUILD_CLUSTER_SCRATCH_PAD) && defined(GPUCA_GPUCODE) */
  /* #if 0 */
  buildClusterScratchPad(
    chargeMap,
    pos,
    smem.posBcast,
    smem.buf,
    smem.innerAboveThreshold,
    &pc);
#else
  buildClusterNaive(chargeMap, &pc, pos);
#endif

  if (idx >= clusternum) {
    return;
  }
  pc.finalize(myDigit);

  tpc::ClusterNative myCluster;
  pc.toNative(myDigit, myCluster);

#if defined(CUT_QTOT)
  bool aboveQTotCutoff = (pc.Q > QTOT_CUTOFF);
#else
  bool aboveQTotCutoff = true;
#endif

  if (aboveQTotCutoff) {
    sortIntoBuckets(
      myCluster,
      myDigit.row,
      maxClusterPerRow,
      clusterInRow,
      clusterByRow);
  }
}

GPUd() void GPUTPCCFClusterizer::addOuterCharge(
  const Array2D<PackedCharge>& chargeMap,
  ClusterAccumulator* cluster,
  const ChargePos& pos,
  Delta2 d)
{
  PackedCharge p = chargeMap[pos.delta(d)];
  cluster->updateOuter(p, d);
}

GPUd() Charge GPUTPCCFClusterizer::addInnerCharge(
  const Array2D<PackedCharge>& chargeMap,
  ClusterAccumulator* cluster,
  const ChargePos& pos,
  Delta2 d)
{
  PackedCharge p = chargeMap[pos.delta(d)];
  return cluster->updateInner(p, d);
}

GPUd() void GPUTPCCFClusterizer::addCorner(
  const Array2D<PackedCharge>& chargeMap,
  ClusterAccumulator* myCluster,
  const ChargePos& pos,
  Delta2 d)
{
  Charge q = addInnerCharge(chargeMap, myCluster, pos, d);

  if (q > CHARGE_THRESHOLD) {
    addOuterCharge(chargeMap, myCluster, pos, {GlobalPad(2 * d.x), d.y});
    addOuterCharge(chargeMap, myCluster, pos, {d.x, Timestamp(2 * d.y)});
    addOuterCharge(chargeMap, myCluster, pos, {GlobalPad(2 * d.x), Timestamp(2 * d.y)});
  }
}

GPUd() void GPUTPCCFClusterizer::addLine(
  const Array2D<PackedCharge>& chargeMap,
  ClusterAccumulator* myCluster,
  const ChargePos& pos,
  Delta2 d)
{
  Charge q = addInnerCharge(chargeMap, myCluster, pos, d);

  if (q > CHARGE_THRESHOLD) {
    addOuterCharge(chargeMap, myCluster, pos, {GlobalPad(2 * d.x), Timestamp(2 * d.y)});
  }
}

GPUd() void GPUTPCCFClusterizer::updateClusterScratchpadInner(
  ushort lid,
  ushort N,
  const PackedCharge* buf,
  ClusterAccumulator* cluster,
  uchar* innerAboveThreshold)
{
  uchar aboveThreshold = 0;

  LOOP_UNROLL_ATTR for (ushort i = 0; i < N; i++)
  {
    Delta2 d = CfConsts::InnerNeighbors[i];

    PackedCharge p = buf[N * lid + i];

    Charge q = cluster->updateInner(p, d);

    aboveThreshold |= (uchar(q > CHARGE_THRESHOLD) << i);
  }

  innerAboveThreshold[lid] = aboveThreshold;

  GPUbarrier();
}

GPUd() void GPUTPCCFClusterizer::updateClusterScratchpadOuter(
  ushort lid,
  ushort N,
  ushort M,
  ushort offset,
  const PackedCharge* buf,
  ClusterAccumulator* cluster)
{
  LOOP_UNROLL_ATTR for (ushort i = offset; i < M + offset; i++)
  {
    PackedCharge p = buf[N * lid + i];

    Delta2 d = CfConsts::OuterNeighbors[i];

    cluster->updateOuter(p, d);
  }
}

GPUd() void GPUTPCCFClusterizer::buildClusterScratchPad(
  const Array2D<PackedCharge>& chargeMap,
  ChargePos pos,
  ChargePos* posBcast,
  PackedCharge* buf,
  uchar* innerAboveThreshold,
  ClusterAccumulator* myCluster)
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
  updateClusterScratchpadInner(
    ll,
    8,
    buf,
    myCluster,
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
    updateClusterScratchpadOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      myCluster);
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
    updateClusterScratchpadOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      myCluster);
  }
#endif
}

GPUd() void GPUTPCCFClusterizer::buildClusterNaive(
  const Array2D<PackedCharge>& chargeMap,
  ClusterAccumulator* myCluster,
  const ChargePos& pos)
{
  // Add charges in top left corner:
  // O O o o o
  // O I i i o
  // o i c i o
  // o i i i o
  // o o o o o
  addCorner(chargeMap, myCluster, pos, {-1, -1});

  // Add upper charges
  // o o O o o
  // o i I i o
  // o i c i o
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, pos, {0, -1});

  // Add charges in top right corner:
  // o o o O O
  // o i i I O
  // o i c i o
  // o i i i o
  // o o o o o
  addCorner(chargeMap, myCluster, pos, {1, -1});

  // Add left charges
  // o o o o o
  // o i i i o
  // O I c i o
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, pos, {-1, 0});

  // Add right charges
  // o o o o o
  // o i i i o
  // o i c I O
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, pos, {1, 0});

  // Add charges in bottom left corner:
  // o o o o o
  // o i i i o
  // o i c i o
  // O I i i o
  // O O o o o
  addCorner(chargeMap, myCluster, pos, {-1, 1});

  // Add bottom charges
  // o o o o o
  // o i i i o
  // o i c i o
  // o i I i o
  // o o O o o
  addLine(chargeMap, myCluster, pos, {0, 1});

  // Add charges in bottom right corner:
  // o o o o o
  // o i i i o
  // o i c i o
  // o i i I O
  // o o o O O
  addCorner(chargeMap, myCluster, pos, {1, 1});
}

GPUd() void GPUTPCCFClusterizer::sortIntoBuckets(const tpc::ClusterNative& cluster, const uint bucket, const uint maxElemsPerBucket, uint* elemsInBucket, tpc::ClusterNative* buckets)
{
  uint posInBucket = CAMath::AtomicAdd(&elemsInBucket[bucket], 1);

  if (posInBucket < maxElemsPerBucket) {
    buckets[maxElemsPerBucket * bucket + posInBucket] = cluster;
  }
}
