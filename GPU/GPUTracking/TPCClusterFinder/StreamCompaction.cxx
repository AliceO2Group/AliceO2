// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file StreamCompaction.cxx
/// \author Felix Weiglhofer

#include "StreamCompaction.h"
#include "GPUCommonAlgorithm.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

using namespace deprecated;

GPUd() void StreamCompaction::nativeScanUpStartImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                                    const uchar* predicate,
                                                    int* sums,
                                                    int* incr, int nElems)
{
  int idx = get_global_id(0);
  int pred = predicate[idx];
  if (idx >= nElems) {
    pred = 0;
  }
  int scanRes = work_group_scan_inclusive_add((int)pred); // TODO: Why don't we store scanRes and read it back in compactDigit?

  /* sums[idx] = scanRes; */

  int lid = get_local_id(0);
  int lastItem = get_local_size(0) - 1;
  int gid = get_group_id(0);

  /* DBGPR_1("ScanUp: idx = %d", idx); */

  if (lid == lastItem) {
    incr[gid] = scanRes;
  }
}

GPUd() void StreamCompaction::nativeScanUpImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                               int* sums,
                                               int* incr, int nElems)
{
  int idx = get_global_id(0);
  int scanRes = work_group_scan_inclusive_add((idx < nElems) ? sums[idx] : 0);

  /* DBGPR_2("ScanUp: idx = %d, res = %d", idx, scanRes); */

  sums[idx] = scanRes;

  int lid = get_local_id(0);
  int lastItem = get_local_size(0) - 1;
  int gid = get_group_id(0);

  /* DBGPR_1("ScanUp: idx = %d", idx); */

  if (lid == lastItem) {
    incr[gid] = scanRes;
  }
}

GPUd() void StreamCompaction::nativeScanTopImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem, int* incr, int nElems)
{
  int idx = get_global_id(0);

  /* DBGPR_1("ScanTop: idx = %d", idx); */

  int scanRes = work_group_scan_inclusive_add((idx < nElems) ? incr[idx] : 0);
  incr[idx] = scanRes;
}

GPUd() void StreamCompaction::nativeScanDownImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                                 int* sums,
                                                 const int* incr,
                                                 unsigned int offset, int nElems)
{
  int gid = get_group_id(0);
  int idx = get_global_id(0) + offset;

  int shift = incr[gid];

  if (idx < nElems) {
    sums[idx] += shift;
  }
}

GPUd() void StreamCompaction::compactDigitImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                               const Digit* in,
                                               Digit* out,
                                               const uchar* predicate,
                                               int* newIdx,
                                               const int* incr,
                                               int nElems)
{
  int gid = get_group_id(0);
  int idx = get_global_id(0);

  int lastItem = get_global_size(0) - 1;

  bool iAmDummy = (idx >= nElems);

  int pred = (iAmDummy) ? 0 : predicate[idx];
  int scanRes = work_group_scan_inclusive_add(pred);

  int compIdx = scanRes;
  if (gid) {
    compIdx += incr[gid - 1];
  }

  if (pred) {
    out[compIdx - 1] = in[idx];
  }

  if (idx == lastItem) {
    newIdx[idx] = compIdx; // TODO: Eventually, we can just return the last value, no need to store to memory
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
