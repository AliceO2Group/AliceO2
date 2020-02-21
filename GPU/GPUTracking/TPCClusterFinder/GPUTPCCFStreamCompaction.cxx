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

#include "GPUTPCCFStreamCompaction.h"
#include "GPUCommonAlgorithm.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

using namespace deprecated;

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::nativeScanUpStart>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage)
{
  int nElems = compactionElems(clusterer, stage);
  nativeScanUpStartImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

GPUd() void GPUTPCCFStreamCompaction::nativeScanUpStartImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::nativeScanUp>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  nativeScanUpImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

GPUd() void GPUTPCCFStreamCompaction::nativeScanUpImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::nativeScanTop>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  nativeScanTopImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, nElems);
}

GPUd() void GPUTPCCFStreamCompaction::nativeScanTopImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, int* incr, int nElems)
{
  int idx = get_global_id(0);

  /* DBGPR_1("ScanTop: idx = %d", idx); */

  int scanRes = work_group_scan_inclusive_add((idx < nElems) ? incr[idx] : 0);
  incr[idx] = scanRes;
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::nativeScanDown>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, unsigned int offset, int nElems)
{
  nativeScanDownImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, offset, nElems);
}

GPUd() void GPUTPCCFStreamCompaction::nativeScanDownImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::compactDigit>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage, deprecated::PackedDigit* in, deprecated::PackedDigit* out)
{
  unsigned int nElems = compactionElems(clusterer, stage);
  compactDigitImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, in, out, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
  unsigned int lastId = get_global_size(0) - 1;
  if ((unsigned int)get_global_id(0) == lastId) {
    if (stage) {
      clusterer.mPmemory->counters.nClusters = clusterer.mPbuf[lastId];
    } else {
      clusterer.mPmemory->counters.nPeaks = clusterer.mPbuf[lastId];
    }
  }
}

GPUd() void GPUTPCCFStreamCompaction::compactDigitImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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

GPUd() int GPUTPCCFStreamCompaction::compactionElems(processorType& clusterer, int stage)
{
  return (stage) ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nDigits;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
