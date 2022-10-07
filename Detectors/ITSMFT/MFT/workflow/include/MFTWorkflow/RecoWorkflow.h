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

#ifndef O2_MFT_RECOWORKFLOW_H_
#define O2_MFT_RECOWORKFLOW_H_

/// @file   RecoWorkflow.h

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace mft
{

namespace reco_workflow
{
framework::WorkflowSpec getWorkflow(
  bool useMC,
  bool upstreamDigits,
  bool upstreamClusters,
  bool disableRootOutput,
  bool runAssessment,
  bool processGen,
  bool runTracking,
  int nThreads,
  bool runTracks2Records);
}

} // namespace mft
} // namespace o2
#endif
