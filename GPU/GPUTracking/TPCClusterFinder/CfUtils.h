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

class CfUtils
{

 public:
  static GPUdi() bool isAtEdge(const deprecated::Digit* d)
  {
    return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW - 2);
  }

  static GPUdi() bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << CfConsts::OuterToInner[outerIdx]);
  }

  static GPUdi() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << CfConsts::OuterToInnerInv[outerIdx]);
  }

  static GPUdi() ushort partition(GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem, ushort ll, bool pred, ushort partSize, ushort* newPartSize)
  {
    bool participates = ll < partSize;

    ushort lpos = work_group_scan_inclusive_add((int)(!pred && participates));

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE - 1);

    lpos -= 1;
    ushort pos = (participates && !pred) ? lpos : part;

    *newPartSize = part;
    return pos;
  }

  // Maps the position of a pad given as row and index in that row to a unique
  // index between 0 and TPC_NUM_OF_PADS.
  static GPUdi() GlobalPad tpcGlobalPadIdx(Row row, Pad pad)
  {
    return TPC_PADS_PER_ROW_PADDED * row + pad + PADDING_PAD;
  }

  template <typename T>
  static GPUdi() void blockLoad(
    const Array2D<T>& map,
    uint wgSize,
    uint elems,
    ushort ll,
    uint offset,
    uint N,
    GPUconstexprref() const Delta2* neighbors,
    GPUsharedref() const ChargePos* posBcast,
    GPUsharedref() T* buf)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort x = ll % N;
    ushort y = ll / N;
    Delta2 d = neighbors[x + offset];
    Delta dp = d.x;
    Delta dt = d.y;
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N))
    {
      ChargePos readFrom = posBcast[i];
      uint writeTo = N * i + x;
      buf[writeTo] = map[{readFrom.gpad + dp, readFrom.time + dt}];
    }
    GPUbarrier();
#else
    if (ll >= wgSize) {
      return;
    }

    ChargePos readFrom = posBcast[ll];

    GPUbarrier();

    for (unsigned int i = 0; i < N; i++) {
      Delta2 d = neighbors[i + offset];
      Delta dp = d.x;
      Delta dt = d.y;

      uint writeTo = N * ll + i;
      buf[writeTo] = map[{readFrom.gpad + dp, readFrom.time + dt}];
    }

    GPUbarrier();
#endif
  }

  template <typename T, typename Pred>
  static GPUdi() void condBlockLoad(
    const Array2D<T>& map,
    ushort wgSize,
    ushort elems,
    ushort ll,
    ushort offset,
    ushort N,
    GPUconstexprref() const Delta2* neighbors,
    GPUsharedref() const ChargePos* posBcast,
    GPUsharedref() const uchar* aboveThreshold,
    GPUsharedref() T* buf,
    Pred&& pred)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort y = ll / N;
    ushort x = ll % N;
    Delta2 d = neighbors[x + offset];
    Delta dp = d.x;
    Delta dt = d.y;
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N))
    {
      ChargePos readFrom = posBcast[i];
      uchar above = aboveThreshold[i];
      uint writeTo = N * i + x;
      T v(0);
      if (pred(above, x + offset)) {
        v = map[{readFrom.gpad + dp, readFrom.time + dt}];
      }
      buf[writeTo] = v;
    }
    GPUbarrier();

#else
    if (ll >= wgSize) {
      return;
    }

    ChargePos readFrom = posBcast[ll];
    uchar above = aboveThreshold[ll];
    GPUbarrier();

    for (unsigned int i = 0; i < N; i++) {
      Delta2 d = neighbors[i + offset];
      Delta dp = d.x;
      Delta dt = d.y;

      uint writeTo = N * ll + i;
      T v(0);
      if (pred(above, i + offset)) {
        v = map[{readFrom.gpad + dp, readFrom.time + dt}];
      }
      buf[writeTo] = v;
    }

    GPUbarrier();
#endif
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
