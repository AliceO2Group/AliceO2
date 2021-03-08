// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   write-raw-from-root-workflow.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
///

#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DispatchPolicy.h"

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  using o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("digit-hmpid-write", CompletionPolicy::CompletionOp::Consume));
}

#include "Framework/runDataProcessing.h"

#include "HMPIDWorkflow/WriteRawFromRootSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  DataProcessorSpec consumer = o2::hmpid::getWriteRawFromRootSpec();
  //  DataProcessorSpec consumer = o2::hmpid::getDecodingSpec();
  specs.push_back(consumer);
  //  specs.push_back(consumer);

  return specs;
}
