// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CosmicsClusterCollectorSpec.h
/// @brief  Collects the raw clusters of matched cosmics (attached + road around the legs) for offline refits

#ifndef O2_COSMICS_CLUSTER_COLLECTOR_SPEC_H
#define O2_COSMICS_CLUSTER_COLLECTOR_SPEC_H

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2::globaltracking
{

/// create a processor spec collecting the clusters of the cosmics found by the cosmics-matcher
/// roadDets: detectors (ITS, TOF, TRD) whose hits along the cosmic are collected besides the TPC road
framework::DataProcessorSpec getCosmicsClusterCollectorSpec(o2::dataformats::GlobalTrackID::mask_t src, bool useMC, bool itsStag, o2::detectors::DetID::mask_t roadDets);

} // namespace o2::globaltracking

#endif
