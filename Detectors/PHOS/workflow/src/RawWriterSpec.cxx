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
#include <fairlogger/Logger.h>

/// \file   PHOS/Workflow/src/RawWriterSpec.cxx
/// \brief  Digits to raw converter spec for PHOS
/// \author Dmitri Peresunko <Dmitri.Peresunko at cern.ch>
/// \date   20 Nov 2020

#include "PHOSWorkflow/RawWriterSpec.h"
#include "Framework/RootSerializationSupport.h"
#include "DataFormatsPHOS/Digit.h"
#include "Framework/ControlService.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "PHOSSimulation/RawWriter.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::phos::reco_workflow;

void RawWriterSpec::init(framework::InitContext& ctx)
{

  auto rawdir = ctx.options().get<std::string>("rawpath");

  LOG(info) << "[PHOSRawWriter - init] Initialize raw writer ";
  if (!mRawWriter) {
    mRawWriter = new o2::phos::RawWriter();
    mRawWriter->setOutputLocation(rawdir.data());
    mRawWriter->init();
  }
}

void RawWriterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(debug) << "[PHOSRawWriter - run] called";

  auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
  auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
  LOG(info) << "[PHOSRawWriter - run]  Received " << digits.size() << " digits and " << digitsTR.size() << " TriggerRecords";

  mRawWriter->digitsToRaw(digits, digitsTR);
  LOG(info) << "[PHOSRawWriter - run]  Finished ";

  //flash and close output files
  mRawWriter->getWriter().close();
  ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getRawWriterSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  return o2::framework::DataProcessorSpec{"PHOSRawWriterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::RawWriterSpec>(),
                                          o2::framework::Options{
                                            {"rawpath", o2::framework::VariantType::String, "./", {"path to write raw"}},
                                          }};
}
