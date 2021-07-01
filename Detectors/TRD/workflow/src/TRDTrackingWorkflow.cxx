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

/// @file  TRDTrackingWorkflow.cxx

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TRDWorkflow/TRDTrackletTransformerSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "TRDWorkflowIO/TRDTrackWriterSpec.h"
#include "TRDWorkflow/TRDTrackingWorkflow.h"
#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDWorkflowIO/TRDCalibWriterSpec.h"
#include "ITSWorkflow/IRFrameReaderSpec.h"

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

framework::WorkflowSpec getTRDTrackingWorkflow(bool disableRootInp, bool disableRootOut, GTrackID::mask_t srcTRD, bool trigRecFilterActive)
{
  framework::WorkflowSpec specs;
  bool useMC = false;
  if (!disableRootInp) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
      specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, srcTRD)) {
      specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));
    }
    specs.emplace_back(o2::trd::getTRDTrackletReaderSpec(useMC, false));
    if (trigRecFilterActive) {
      specs.emplace_back(o2::its::getIRFrameReaderSpec());
    }
  }

  specs.emplace_back(o2::trd::getTRDTrackletTransformerSpec(trigRecFilterActive));
  specs.emplace_back(o2::trd::getTRDGlobalTrackingSpec(useMC, srcTRD, trigRecFilterActive));
  specs.emplace_back(o2::trd::getTRDTrackBasedCalibSpec());

  if (!disableRootOut) {
    if (GTrackID::includesSource(GTrackID::Source::ITSTPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDGlobalTrackWriterSpec(useMC));
    }
    if (GTrackID::includesSource(GTrackID::Source::TPC, srcTRD)) {
      specs.emplace_back(o2::trd::getTRDTPCTrackWriterSpec(useMC));
    }
    specs.emplace_back(o2::trd::getTRDCalibWriterSpec());
  }
  return specs;
}

} // namespace trd
} // namespace o2
