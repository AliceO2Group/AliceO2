// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TRDTrackingWorkflow.cxx

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "TRDWorkflow/TRDTrackletReaderSpec.h"
#include "TRDWorkflow/TRDGlobalTrackingSpec.h"
#include "TRDWorkflow/TRDTrackWriterSpec.h"

namespace o2
{
namespace trd
{

framework::WorkflowSpec getTRDTrackingWorkflow(bool disableRootInp, bool disableRootOut, bool useTrkltTransf)
{
  framework::WorkflowSpec specs;
  bool useMC = false;
  if (!disableRootInp) {
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    specs.emplace_back(o2::trd::getTRDTrackletReaderSpec(useMC, useTrkltTransf));
  }

  specs.emplace_back(o2::trd::getTRDGlobalTrackingSpec(useMC, useTrkltTransf));

  if (!disableRootOut) {
    specs.emplace_back(o2::trd::getTRDTrackWriterSpec(useMC));
  }
  return specs;
}

} // namespace trd
} // namespace o2
