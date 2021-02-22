// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   read-raw-file-workflow.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
///

#include <iostream>
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/Task.h"

// customize dispatch policy, dispatch immediately what is ready
void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  using DispatchOp = o2::framework::DispatchPolicy::DispatchOp;
  // we customize all devices to dispatch data immediately
  auto readerMatcher = [](auto const& spec) {
    //
    //  std::cout << "customize reader = " << spec.name << std::endl;
    //    std::cout << "PingReader" << std::endl;
    return true;
    //    return std::regex_match(spec.name.begin(), spec.name.end(), std::regex(".*-reader"));
  };
  auto triggerMatcher = [](auto const& query) {
    // a bit of a hack but we want this to be configurable from the command line,
    // however DispatchPolicy is inserted before all other setup. Triggering depending
    // on the global variable set from the command line option. If scheduled messages
    // are not triggered they are sent out at the end of the computation
    //  std::cout << "customize Trigger origin = " << query.origin << " description = " << query.description << std::endl;
    // std::cout << "PingTrig" << std::endl;
    return true;
    //    return gDispatchTrigger.origin == query.origin && gDispatchTrigger.description == query.description;
  };
  policies.push_back({"pr-f-re", readerMatcher, DispatchOp::WhenReady, triggerMatcher});
}

#include "Framework/runDataProcessing.h"

#include "HMPIDWorkflow/ReadRawFileSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer = o2::hmpid::getReadRawFileSpec();
  specs.push_back(producer);

  return specs;
}
