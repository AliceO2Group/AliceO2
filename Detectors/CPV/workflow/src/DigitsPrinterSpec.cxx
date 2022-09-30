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

#include <vector>
#include <iostream>

#include <fairlogger/Logger.h>

#include "Framework/RootSerializationSupport.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/Digit.h"
#include "CPVWorkflow/DigitsPrinterSpec.h"

using namespace o2::cpv::reco_workflow;

void DigitsPrinterSpec::init(framework::InitContext& ctx)
{
}

void DigitsPrinterSpec::run(framework::ProcessingContext& pc)
{
  // Get the CPV block header and check whether it contains digits
  LOG(debug) << "[CPVDigitsPrinter - process] called";
  auto dataref = pc.inputs().get("digits");
  auto const* cpvheader = o2::framework::DataRefUtils::getHeader<o2::cpv::CPVBlockHeader*>(dataref);
  if (!cpvheader->mHasPayload) {
    LOG(debug) << "[CPVDigitsPrinter - process] No more digits" << std::endl;
    pc.services().get<o2::framework::ControlService>().endOfStream();
    pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
    return;
  }

  auto digits = pc.inputs().get<std::vector<o2::cpv::Digit>>("digits");
  std::cout << "[CPVDigitsPrinter - process] receiveed " << digits.size() << " digits ..." << std::endl;
  if (digits.size()) {
    for (const auto& d : digits) {
      std::cout << "[CPVDigitsPrinter - process] Channel(" << d.getAbsId() << ") energy: " << d.getAmplitude() << std::endl;
    }
  }
}

o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getPhosDigitsPrinterSpec()
{

  return o2::framework::DataProcessorSpec{"CPVDigitsPrinter",
                                          {{"digits", o2::header::gDataOriginCPV, "DIGITS", 0, o2::framework::Lifetime::Timeframe}},
                                          {},
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::DigitsPrinterSpec>()};
}
