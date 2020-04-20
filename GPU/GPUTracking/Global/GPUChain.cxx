// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

GPUChain::krnlExec GPUChain::GetGrid(unsigned int totalItems, unsigned int nThreads, int stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  const unsigned int nBlocks = (totalItems + nThreads - 1) / nThreads;
  return {nBlocks, nThreads, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGrid(unsigned int totalItems, int stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  return {(unsigned int)-1, totalItems, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGridBlk(unsigned int nBlocks, int stream, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st)
{
  return {nBlocks, (unsigned int)-1, stream, d, st};
}

GPUChain::krnlExec GPUChain::GetGridBlkStep(unsigned int nBlocks, int stream, GPUCA_RECO_STEP st)
{
  return {nBlocks, (unsigned int)-1, stream, GPUReconstruction::krnlDeviceType::Auto, st};
}
