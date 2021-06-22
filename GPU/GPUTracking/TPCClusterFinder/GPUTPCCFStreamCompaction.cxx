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

#include "ChargePos.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanStart>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage)
{
  int nElems = compactionElems(clusterer, stage);
  nativeScanUpStartImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

GPUdii() void GPUTPCCFStreamCompaction::nativeScanUpStartImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                              const uchar* predicate,
                                                              int* sums,
                                                              int* incr, int nElems)
{
  int idx = get_global_id(0);
  int pred = 0;
  if (idx < nElems) {
    pred = predicate[idx];
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
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanUp>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  nativeScanUpImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems);
}

GPUdii() void GPUTPCCFStreamCompaction::nativeScanUpImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanTop>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  nativeScanTopImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, nElems);
}

GPUdii() void GPUTPCCFStreamCompaction::nativeScanTopImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, int* incr, int nElems)
{
  int idx = get_global_id(0);

  /* DBGPR_1("ScanTop: idx = %d", idx); */

  int scanRes = work_group_scan_inclusive_add((idx < nElems) ? incr[idx] : 0);
  incr[idx] = scanRes;
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanDown>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, unsigned int offset, int nElems)
{
  nativeScanDownImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, offset, nElems);
}

GPUdii() void GPUTPCCFStreamCompaction::nativeScanDownImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
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
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::compactDigits>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage, ChargePos* in, ChargePos* out)
{
  unsigned int nElems = compactionElems(clusterer, stage);
  SizeT bufferSize = (stage) ? clusterer.mNMaxClusters : clusterer.mNMaxPeaks;
  compactImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, in, out, clusterer.mPisPeak, clusterer.mPbuf + (iBuf - 1) * clusterer.mBufSize, clusterer.mPbuf + iBuf * clusterer.mBufSize, nElems, bufferSize);
  unsigned int lastId = get_global_size(0) - 1;
  if ((unsigned int)get_global_id(0) == lastId) {
    SizeT nFinal = clusterer.mPbuf[lastId];
    if (nFinal > bufferSize) {
      clusterer.raiseError(stage ? GPUErrors::ERROR_CF_CLUSTER_OVERFLOW : GPUErrors::ERROR_CF_PEAK_OVERFLOW, nFinal, bufferSize);
      nFinal = bufferSize;
    }
    if (stage) {
      clusterer.mPmemory->counters.nClusters = nFinal;
    } else {
      clusterer.mPmemory->counters.nPeaks = nFinal;
    }
  }
}

GPUdii() void GPUTPCCFStreamCompaction::compactImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                    const ChargePos* in,
                                                    ChargePos* out,
                                                    const uchar* predicate,
                                                    int* newIdx,
                                                    const int* incr,
                                                    int nElems,
                                                    SizeT bufferSize)
{
  int gid = get_group_id(0);
  int idx = get_global_id(0);

  int lastItem = get_global_size(0) - 1;

  bool iAmDummy = (idx >= nElems);

  int pred = (iAmDummy) ? 0 : predicate[idx];
  int scanRes = work_group_scan_inclusive_add(pred);

  SizeT compIdx = scanRes;
  if (gid) {
    compIdx += incr[gid - 1];
  }

  SizeT tgtIdx = compIdx - 1;
  if (pred && tgtIdx < bufferSize) {
    out[tgtIdx] = in[idx];
  }

  if (idx == lastItem) {
    newIdx[idx] = compIdx; // TODO: Eventually, we can just return the last value, no need to store to memory
  }
}

GPUdii() int GPUTPCCFStreamCompaction::compactionElems(processorType& clusterer, int stage)
{
  return (stage) ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
}
