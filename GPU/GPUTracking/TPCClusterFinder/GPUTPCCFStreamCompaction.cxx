// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file StreamCompaction.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFStreamCompaction.h"
#include "GPUCommonAlgorithm.h"

#include "ChargePos.h"
#include "CfUtils.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanStart>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage)
{
  int nElems = CompactionElems(clusterer, stage);

  const auto* predicate = clusterer.mPisPeak;
  auto* scanOffset = clusterer.GetScanBuffer(iBuf);

  int iThreadGlobal = get_global_id(0);
  int pred = 0;
  if (iThreadGlobal < nElems) {
    pred = predicate[iThreadGlobal];
  }

  int nElemsInBlock = CfUtils::blockPredicateSum<GPUCA_THREAD_COUNT_SCAN>(smem, pred);

  int lastThread = nThreads - 1;
  if (iThread == lastThread) {
    scanOffset[iBlock] = nElemsInBlock;
  }
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanUp>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  auto* scanOffset = clusterer.GetScanBuffer(iBuf - 1);
  auto* scanOffsetNext = clusterer.GetScanBuffer(iBuf);

  int iThreadGlobal = get_global_id(0);
  int offsetInBlock = work_group_scan_inclusive_add((iThreadGlobal < nElems) ? scanOffset[iThreadGlobal] : 0);

  // TODO: This write isn't needed??
  scanOffset[iThreadGlobal] = offsetInBlock;

  int lastThread = nThreads - 1;
  if (iThread == lastThread) {
    scanOffsetNext[iBlock] = offsetInBlock;
  }
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanTop>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int nElems)
{
  int iThreadGlobal = get_global_id(0);
  int* scanOffset = clusterer.GetScanBuffer(iBuf - 1);

  bool inBounds = (iThreadGlobal < nElems);

  int offsetInBlock = work_group_scan_inclusive_add(inBounds ? scanOffset[iThreadGlobal] : 0);

  if (inBounds) {
    scanOffset[iThreadGlobal] = offsetInBlock;
  }
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::scanDown>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& /*smem*/, processorType& clusterer, int iBuf, unsigned int offset, int nElems)
{
  int iThreadGlobal = get_global_id(0) + offset;

  int* scanOffsetPrev = clusterer.GetScanBuffer(iBuf - 1);
  const int* scanOffset = clusterer.GetScanBuffer(iBuf);

  int shift = scanOffset[iBlock];

  if (iThreadGlobal < nElems) {
    scanOffsetPrev[iThreadGlobal] += shift;
  }
}

template <>
GPUdii() void GPUTPCCFStreamCompaction::Thread<GPUTPCCFStreamCompaction::compactDigits>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, int iBuf, int stage, ChargePos* in, ChargePos* out)
{
  unsigned int nElems = CompactionElems(clusterer, stage);
  SizeT bufferSize = (stage) ? clusterer.mNMaxClusters : clusterer.mNMaxPeaks;

  unsigned int iThreadGlobal = get_global_id(0);

  const auto* predicate = clusterer.mPisPeak;
  const auto* scanOffset = clusterer.GetScanBuffer(iBuf);

  bool iAmDummy = (iThreadGlobal >= nElems);

  int pred = (iAmDummy) ? 0 : predicate[iThreadGlobal];
  int offsetInBlock = CfUtils::blockPredicateScan<GPUCA_THREAD_COUNT_SCAN>(smem, pred);

  SizeT globalOffsetOut = offsetInBlock;
  if (iBlock > 0) {
    globalOffsetOut += scanOffset[iBlock - 1];
  }

  if (pred && globalOffsetOut < bufferSize) {
    out[globalOffsetOut] = in[iThreadGlobal];
  }

  unsigned int lastId = get_global_size(0) - 1;
  if (iThreadGlobal == lastId) {
    SizeT nFinal = globalOffsetOut + pred;
    if (nFinal > bufferSize) {
      clusterer.raiseError(stage ? GPUErrors::ERROR_CF_CLUSTER_OVERFLOW : GPUErrors::ERROR_CF_PEAK_OVERFLOW, clusterer.mISlice, nFinal, bufferSize);
      nFinal = bufferSize;
    }
    if (stage) {
      clusterer.mPmemory->counters.nClusters = nFinal;
    } else {
      clusterer.mPmemory->counters.nPeaks = nFinal;
    }
  }
}

GPUdii() int GPUTPCCFStreamCompaction::CompactionElems(processorType& clusterer, int stage)
{
  return (stage) ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
}
