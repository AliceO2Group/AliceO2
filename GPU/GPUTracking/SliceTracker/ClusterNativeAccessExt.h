// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterNativeAccessExt.h
/// \author David Rohr

#ifndef CLUSTERNATIVEACCESSEXT_H
#define CLUSTERNATIVEACCESSEXT_H

#include "GPUTPCDef.h"

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#else
namespace o2
{
namespace TPC
{
struct ClusterNative {
};
struct ClusterNativeAccessFullTPC {
  const ClusterNative* clusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
};
struct Constants {
  static constexpr int MAXSECTOR = GPUCA_NSLICES;
  static constexpr int MAXGLOBALPADROW = GPUCA_ROW_COUNT;
};
} // namespace TPC
} // namespace o2
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct ClusterNativeAccessExt : public o2::TPC::ClusterNativeAccessFullTPC {
  unsigned int clusterOffset[GPUCA_NSLICES][GPUCA_ROW_COUNT];
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
