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

#include "GPUDef.h"
#include "clusterFinderDefs.h"
#include "CfConsts.h"

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
  return aboveThreshold & (1 << OUTER_TO_INNER[outerIdx]);
}

GPUdi() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
{
  return aboveThreshold & (1 << OUTER_TO_INNER_INV[outerIdx]);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
