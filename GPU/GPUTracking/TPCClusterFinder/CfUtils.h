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

namespace GPUCA_NAMESPACE::gpu
{

class CfUtils
{

 public:
  static GPUdi() bool isAtEdge(const ChargePos& pos)
  {
    return (pos.pad() < 2 || pos.pad() >= TPC_PADS_PER_ROW - 2);
  }

  static GPUdi() bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << cfconsts::OuterToInner[outerIdx]);
  }

  static GPUdi() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
  {
    return aboveThreshold & (1 << cfconsts::OuterToInnerInv[outerIdx]);
  }

  static GPUdi() bool isPeak(uchar peak) { return peak & 0x01; }

  static GPUdi() bool isAboveThreshold(uchar peak) { return peak >> 1; }

  template <size_t SCRATCH_PAD_WORK_GROUP_SIZE, typename SharedMemory>
  static GPUdi() ushort partition(SharedMemory& smem, ushort ll, bool pred, ushort partSize, ushort* newPartSize)
  {
    bool participates = ll < partSize;

    ushort lpos = work_group_scan_inclusive_add(short(!pred && participates));

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE - 1);

    lpos -= 1;
    ushort pos = (participates && !pred) ? lpos : part;

    *newPartSize = part;
    return pos;
  }

  template <typename T>
  static GPUdi() void blockLoad(
    const Array2D<T>& map,
    uint wgSize,
    uint elems,
    ushort ll,
    uint offset,
    uint N,
    GPUconstexprref() const tpccf::Delta2* neighbors,
    const ChargePos* posBcast,
    GPUgeneric() T* buf)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort x = ll % N;
    ushort y = ll / N;
    tpccf::Delta2 d = neighbors[x + offset];

    for (unsigned int i = y; i < wgSize; i += (elems / N)) {
      ChargePos readFrom = posBcast[i];
      uint writeTo = N * i + x;
      buf[writeTo] = map[readFrom.delta(d)];
    }
    GPUbarrier();
#else
    if (ll >= wgSize) {
      return;
    }

    ChargePos readFrom = posBcast[ll];

    GPUbarrier();

    for (unsigned int i = 0; i < N; i++) {
      tpccf::Delta2 d = neighbors[i + offset];

      uint writeTo = N * ll + i;
      buf[writeTo] = map[readFrom.delta(d)];
    }

    GPUbarrier();
#endif
  }

  template <typename T, bool Inv = false>
  static GPUdi() void condBlockLoad(
    const Array2D<T>& map,
    ushort wgSize,
    ushort elems,
    ushort ll,
    ushort offset,
    ushort N,
    GPUconstexprref() const tpccf::Delta2* neighbors,
    const ChargePos* posBcast,
    const uchar* aboveThreshold,
    GPUgeneric() T* buf)
  {
#if defined(GPUCA_GPUCODE)
    GPUbarrier();
    ushort y = ll / N;
    ushort x = ll % N;
    tpccf::Delta2 d = neighbors[x + offset];
    for (unsigned int i = y; i < wgSize; i += (elems / N)) {
      ChargePos readFrom = posBcast[i];
      uchar above = aboveThreshold[i];
      uint writeTo = N * i + x;
      T v(0);
      bool cond = (Inv) ? innerAboveThresholdInv(above, x + offset)
                        : innerAboveThreshold(above, x + offset);
      if (cond) {
        v = map[readFrom.delta(d)];
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
      tpccf::Delta2 d = neighbors[i + offset];

      uint writeTo = N * ll + i;
      T v(0);
      bool cond = (Inv) ? innerAboveThresholdInv(above, i + offset)
                        : innerAboveThreshold(above, i + offset);
      if (cond) {
        v = map[readFrom.delta(d)];
      }
      buf[writeTo] = v;
    }

    GPUbarrier();
#endif
  }
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
