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

#ifndef O2_CTP_RECOWORKFLOW_H
#define O2_CTP_RECOWORKFLOW_H

#include "Framework/WorkflowSpec.h"
#include <string>
#include <vector>

namespace o2
{

namespace ctp
{

namespace reco_workflow
{

/// create the workflow for CTP reconstruction
framework::WorkflowSpec getWorkflow(bool noLostTF);

} // namespace reco_workflow

} // namespace ctp

} // namespace o2
#endif
