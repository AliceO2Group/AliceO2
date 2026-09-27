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

#ifndef O2_MFT_CARECOWORKFLOW_H_
#define O2_MFT_CARECOWORKFLOW_H_

/// @file CARecoWorkflow.h

#include "Framework/WorkflowSpec.h"
#include "MFTWorkflow/CAWorkflowOptions.h"

namespace o2::mft::ca_reco_workflow
{

framework::WorkflowSpec getWorkflow(const ca::WorkflowOptions& options);

} // namespace o2::mft::ca_reco_workflow

#endif // O2_MFT_CARECOWORKFLOW_H_
