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

///
/// \file    DigitsToRootSpec.h
/// \author  Antonio Franco
///
/// \brief Definition of a data processor to store digits in Root file
///

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DIGITSTOROOTSPEC_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DIGITSTOROOTSPEC_H_

#include "TTree.h"
#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "HMPIDBase/Common.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "CommonDataFormat/InteractionRecord.h"

#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace hmpid
{

class DigitsToRootTask : public framework::Task
{
 public:
  DigitsToRootTask() = default;
  ~DigitsToRootTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  ExecutionTimer mExTimer;
  std::vector<o2::hmpid::Trigger> mTriggers;
  std::vector<o2::hmpid::Digit> mDigits;
  TFile* mfileOut;
  TTree* mTheTree;
  std::string mOutRootFileName;
};

o2::framework::DataProcessorSpec getDigitsToRootSpec(std::string inputSpec = "HMP/DIGITS");

} // end namespace hmpid
} // end namespace o2

#endif
