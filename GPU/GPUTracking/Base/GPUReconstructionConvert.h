// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionConvert.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONCONVERT_H
#define GPURECONSTRUCTIONCONVERT_H

#include <memory>
#include "GPUTPCSettings.h"
class GPUTPCClusterData;

struct ClusterNativeAccessExt;
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class GPUReconstructionConvert
{
public:
	constexpr static unsigned int NSLICES = GPUCA_NSLICES;
	static void ConvertNativeToClusterData(ClusterNativeAccessExt* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin = 0);
};

#endif
