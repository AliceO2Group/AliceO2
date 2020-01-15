// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterizer.cxx
/// \author Felix Weiglhofer

#include "Clusterizer.h"

#include "Array2D.h"
#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUd() void Clusterizer::computeClustersImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                             GPUglobalref() const PackedCharge* chargeMap,
                                             GPUglobalref() const deprecated::Digit* digits,
                                             uint clusternum,
                                             uint maxClusterPerRow,
                                             GPUglobalref() uint* clusterInRow,
                                             GPUglobalref() deprecated::ClusterNative* clusterByRow)
{
  uint idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last cluster but discard the result.
  deprecated::Digit myDigit = digits[CAMath::Min(idx, clusternum - 1)];

  GlobalPad gpad = Array2D::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  ClusterAccumulator pc;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  /* #if defined(BUILD_CLUSTER_SCRATCH_PAD) && defined(GPUCA_GPUCODE) */
  /* #if 0 */
  buildClusterScratchPad(
    chargeMap,
    (ChargePos){gpad, myDigit.time},
    smem.build.posBcast,
    smem.build.buf,
    smem.build.innerAboveThreshold,
    &pc);
#else
  buildClusterNaive(chargeMap, &pc, gpad, myDigit.time);
#endif

  if (idx >= clusternum) {
    return;
  }
  pc.finalize(myDigit);

  deprecated::ClusterNative myCluster;
  pc.toNative(myDigit, myCluster);

#if defined(CUT_QTOT)
  bool aboveQTotCutoff = (pc.Q > QTOT_CUTOFF);
#else
  bool aboveQTotCutoff = true;
#endif

  if (aboveQTotCutoff) {
    sortIntoBuckets(
      &myCluster,
      myDigit.row,
      maxClusterPerRow,
      clusterInRow,
      clusterByRow);
  }
}

GPUd() void Clusterizer::addOuterCharge(
  GPUglobalref() const PackedCharge* chargeMap,
  ClusterAccumulator* cluster,
  GlobalPad gpad,
  Timestamp time,
  Delta dp,
  Delta dt)
{
  PackedCharge p = CHARGE(chargeMap, gpad + dp, time + dt);
  cluster->updateOuter(p, dp, dt);
}

GPUd() Charge Clusterizer::addInnerCharge(
  GPUglobalref() const PackedCharge* chargeMap,
  ClusterAccumulator* cluster,
  GlobalPad gpad,
  Timestamp time,
  Delta dp,
  Delta dt)
{
  PackedCharge p = CHARGE(chargeMap, gpad + dp, time + dt);
  return cluster->updateInner(p, dp, dt);
}

GPUd() void Clusterizer::addCorner(
  GPUglobalref() const PackedCharge* chargeMap,
  ClusterAccumulator* myCluster,
  GlobalPad gpad,
  Timestamp time,
  Delta dp,
  Delta dt)
{
  Charge q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

  if (q > CHARGE_THRESHOLD) {
    addOuterCharge(chargeMap, myCluster, gpad, time, 2 * dp, dt);
    addOuterCharge(chargeMap, myCluster, gpad, time, dp, 2 * dt);
    addOuterCharge(chargeMap, myCluster, gpad, time, 2 * dp, 2 * dt);
  }
}

GPUd() void Clusterizer::addLine(
  GPUglobalref() const PackedCharge* chargeMap,
  ClusterAccumulator* myCluster,
  GlobalPad gpad,
  Timestamp time,
  Delta dp,
  Delta dt)
{
  Charge q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

  if (q > CHARGE_THRESHOLD) {
    addOuterCharge(chargeMap, myCluster, gpad, time, 2 * dp, 2 * dt);
  }
}

GPUd() void Clusterizer::updateClusterScratchpadInner(
  ushort lid,
  ushort N,
  GPUsharedref() const PackedCharge* buf,
  ClusterAccumulator* cluster,
  GPUsharedref() uchar* innerAboveThreshold)
{
  uchar aboveThreshold = 0;

  LOOP_UNROLL_ATTR for (ushort i = 0; i < N; i++)
  {
    Delta2 d = CfConsts::InnerNeighbors[i];

    Delta dp = d.x;
    Delta dt = d.y;

    PackedCharge p = buf[N * lid + i];

    Charge q = cluster->updateInner(p, dp, dt);

    aboveThreshold |= (uchar(q > CHARGE_THRESHOLD) << i);
  }

  innerAboveThreshold[lid] = aboveThreshold;

  GPUbarrier();
}

GPUd() void Clusterizer::updateClusterScratchpadOuter(
  ushort lid,
  ushort N,
  ushort M,
  ushort offset,
  GPUsharedref() const PackedCharge* buf,
  ClusterAccumulator* cluster)
{
  LOOP_UNROLL_ATTR for (ushort i = offset; i < M + offset; i++)
  {
    PackedCharge p = buf[N * lid + i];

    Delta2 d = CfConsts::OuterNeighbors[i];
    Delta dp = d.x;
    Delta dt = d.y;

    cluster->updateOuter(p, dp, dt);
  }
}

GPUd() void Clusterizer::buildClusterScratchPad(
  GPUglobalref() const PackedCharge* chargeMap,
  ChargePos pos,
  GPUsharedref() ChargePos* posBcast,
  GPUsharedref() PackedCharge* buf,
  GPUsharedref() uchar* innerAboveThreshold,
  ClusterAccumulator* myCluster)
{
  ushort ll = get_local_id(0);

  posBcast[ll] = pos;
  GPUbarrier();

  CfUtils::fillScratchPad_PackedCharge(
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

  /* ClusterAccumulator otherCluster; */
  /* reset(&otherCluster); */

  CfUtils::fillScratchPadCond_PackedCharge(
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
  CfUtils::fillScratchPadCond_PackedCharge(
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

GPUd() void Clusterizer::buildClusterNaive(
  GPUglobalref() const PackedCharge* chargeMap,
  ClusterAccumulator* myCluster,
  GlobalPad gpad,
  Timestamp time)
{
  // Add charges in top left corner:
  // O O o o o
  // O I i i o
  // o i c i o
  // o i i i o
  // o o o o o
  addCorner(chargeMap, myCluster, gpad, time, -1, -1);

  // Add upper charges
  // o o O o o
  // o i I i o
  // o i c i o
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, gpad, time, 0, -1);

  // Add charges in top right corner:
  // o o o O O
  // o i i I O
  // o i c i o
  // o i i i o
  // o o o o o
  addCorner(chargeMap, myCluster, gpad, time, 1, -1);

  // Add left charges
  // o o o o o
  // o i i i o
  // O I c i o
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, gpad, time, -1, 0);

  // Add right charges
  // o o o o o
  // o i i i o
  // o i c I O
  // o i i i o
  // o o o o o
  addLine(chargeMap, myCluster, gpad, time, 1, 0);

  // Add charges in bottom left corner:
  // o o o o o
  // o i i i o
  // o i c i o
  // O I i i o
  // O O o o o
  addCorner(chargeMap, myCluster, gpad, time, -1, 1);

  // Add bottom charges
  // o o o o o
  // o i i i o
  // o i c i o
  // o i I i o
  // o o O o o
  addLine(chargeMap, myCluster, gpad, time, 0, 1);

  // Add charges in bottom right corner:
  // o o o o o
  // o i i i o
  // o i c i o
  // o i i I O
  // o o o O O
  addCorner(chargeMap, myCluster, gpad, time, 1, 1);
}

GPUd() void Clusterizer::sortIntoBuckets(const deprecated::ClusterNative* cluster, const uint bucket, const uint maxElemsPerBucket, GPUglobalref() uint* elemsInBucket, GPUglobalref() deprecated::ClusterNative* buckets)
{
  uint posInBucket = CAMath::AtomicAdd(&elemsInBucket[bucket], 1);

  buckets[maxElemsPerBucket * bucket + posInBucket] = *cluster; // TODO: Must check for overflow over maxElemsPerBucket!
}
