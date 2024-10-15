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

/// \file GPUChain.cxx
/// \author David Rohr

#include "GPUChain.h"
using namespace GPUCA_NAMESPACE::gpu;

constexpr GPUChain::krnlRunRange GPUChain::krnlRunRangeNone;
constexpr GPUChain::krnlEvent GPUChain::krnlEventNone;

GPUChain::krnlExec GPUChain::GetGrid(uint32_t totalItems, uint32_t nThreads, int32_t stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  const uint32_t nBlocks = (totalItems + nThreads - 1) / nThreads;
  return {nBlocks, nThreads, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGrid(uint32_t totalItems, int32_t stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  return {(uint32_t)-1, totalItems, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGridBlk(uint32_t nBlocks, int32_t stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  return {(uint32_t)-2, nBlocks, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGridBlkStep(uint32_t nBlocks, int32_t stream, GPUCA_RECO_STEP st)
{
  return {(uint32_t)-2, nBlocks, stream, GPUReconstruction::krnlDeviceType::Auto, st};
}

GPUChain::krnlExec GPUChain::GetGridAuto(int32_t stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  return {(uint32_t)-3, 0, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGridAutoStep(int32_t stream, GPUCA_RECO_STEP st)
{
  return {(uint32_t)-3, 0, stream, GPUReconstruction::krnlDeviceType::Auto, st};
}
