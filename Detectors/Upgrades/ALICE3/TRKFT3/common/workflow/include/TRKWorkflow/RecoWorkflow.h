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

#ifndef O2_TRK_RECOWORKFLOW_H
#define O2_TRK_RECOWORKFLOW_H

#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/WorkflowSpec.h"

namespace o2::trk
{
namespace reco_workflow
{

/// The digit readers, clusterers and cluster writers of \a detectors, TRK and/or FT3.
o2::framework::WorkflowSpec getWorkflow(bool useMC,
                                        bool upstreamDigits = false,
                                        bool upstreamClusters = false,
                                        bool disableRootOutput = false,
                                        o2::detectors::DetID::mask_t detectors = o2::detectors::DetID::getMask(o2::detectors::DetID::TRK));
}

} // namespace o2::trk

#endif
