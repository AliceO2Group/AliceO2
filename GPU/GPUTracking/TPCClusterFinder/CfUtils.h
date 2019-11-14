// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CfUtils.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CF_UTILS_H
#define O2_GPU_CF_UTILS_H

#include "clusterFinderDefs.h"
#include "GPUCommonAlgorithm.h"
#include "Array2D.h"
#include "CfConsts.h"
#include "GPUTPCClusterFinderKernels.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// TODO wrap these function in a CfUtils class
GPUdi() bool isAtEdge(const deprecated::Digit* d)
{
  return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW - 2);
}

GPUdi() bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
{
  return aboveThreshold & (1 << CfConsts::OuterToInner[outerIdx]);
}

GPUdi() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
{
  return aboveThreshold & (1 << CfConsts::OuterToInnerInv[outerIdx]);
}

GPUdi() ushort partition(GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem, ushort ll, bool pred, ushort partSize, ushort* newPartSize)
{
  bool participates = ll < partSize;

  ushort lpos = work_group_scan_inclusive_add((int)(!pred && participates));

  ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE - 1);

  lpos -= 1;
  ushort pos = (participates && !pred) ? lpos : part;

  *newPartSize = part;
  return pos;
}

// TODO: replace macros by templates
#define DECL_FILL_SCRATCH_PAD(type, accessFunc)                                     \
  GPUdi() void fillScratchPad_##type(                                               \
    GPUglobalref() const type* chargeMap,                                           \
    uint wgSize,                                                                    \
    uint elems,                                                                     \
    ushort ll,                                                                      \
    uint offset,                                                                    \
    uint N,                                                                         \
    GPUconstexprref() const Delta2* neighbors,                                      \
    GPUsharedref() const ChargePos* posBcast,                                       \
    GPUsharedref() type* buf)                                                       \
  {                                                                                 \
    ushort x = ll % N;                                                              \
    ushort y = ll / N;                                                              \
    Delta2 d = neighbors[x + offset];                                               \
    Delta dp = d.x;                                                                 \
    Delta dt = d.y;                                                                 \
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N))         \
    {                                                                               \
      ChargePos readFrom = posBcast[i];                                             \
      uint writeTo = N * i + x;                                                     \
      buf[writeTo] = accessFunc(chargeMap, readFrom.gpad + dp, readFrom.time + dt); \
    }                                                                               \
    GPUbarrier();                                                                   \
  }                                                                                 \
  void anonymousFunction()

#define DECL_FILL_SCRATCH_PAD_COND(type, accessFunc, expandFunc, nameAppendix, null) \
  GPUdi() void fillScratchPad##nameAppendix##_##type(                                \
    GPUglobalref() const type* chargeMap,                                            \
    uint wgSize,                                                                     \
    uint elems,                                                                      \
    ushort ll,                                                                       \
    uint offset,                                                                     \
    uint N,                                                                          \
    GPUconstexprref() const Delta2* neighbors,                                       \
    GPUsharedref() const ChargePos* posBcast,                                        \
    GPUsharedref() const uchar* aboveThreshold,                                      \
    GPUsharedref() type* buf)                                                        \
  {                                                                                  \
    ushort y = ll / N;                                                               \
    ushort x = ll % N;                                                               \
    Delta2 d = neighbors[x + offset];                                                \
    Delta dp = d.x;                                                                  \
    Delta dt = d.y;                                                                  \
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N))          \
    {                                                                                \
      ChargePos readFrom = posBcast[i];                                              \
      uchar above = aboveThreshold[i];                                               \
      uint writeTo = N * i + x;                                                      \
      type v = null;                                                                 \
      if (expandFunc(above, x + offset)) {                                           \
        v = accessFunc(chargeMap, readFrom.gpad + dp, readFrom.time + dt);           \
      }                                                                              \
      buf[writeTo] = v;                                                              \
    }                                                                                \
    GPUbarrier();                                                                    \
  }                                                                                  \
  void anonymousFunction()

DECL_FILL_SCRATCH_PAD(PackedCharge, CHARGE);
DECL_FILL_SCRATCH_PAD(uchar, IS_PEAK);
DECL_FILL_SCRATCH_PAD_COND(PackedCharge, CHARGE, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(PackedCharge, CHARGE, innerAboveThresholdInv, CondInv, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThresholdInv, CondInv, 0);

GPUdi() void fillScratchPadNaive(
  GPUglobalref() const uchar* chargeMap,
  uint wgSize,
  ushort ll,
  uint offset,
  uint N,
  GPUconstexprref() const Delta2* neighbors,
  GPUsharedref() const ChargePos* posBcast,
  GPUsharedref() uchar* buf)
{
  if (ll >= wgSize) {
    return;
  }

  ChargePos readFrom = posBcast[ll];

  for (unsigned int i = 0; i < N; i++) {
    Delta2 d = neighbors[i + offset];
    Delta dp = d.x;
    Delta dt = d.y;

    uint writeTo = N * ll + i;
    buf[writeTo] = IS_PEAK(chargeMap, readFrom.gpad + dp, readFrom.time + dt);
  }

  GPUbarrier();
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
