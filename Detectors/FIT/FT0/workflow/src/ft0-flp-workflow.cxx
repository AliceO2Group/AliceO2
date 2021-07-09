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

#include "CommonUtils/ConfigurableParam.h"
#include "FITWorkflow/FITDataReaderDPLSpec.h"
#include "FITWorkflow/FITDigitWriterSpec.h"
#include "FITWorkflow/RawReaderFIT.h"
#include "DataFormatsFT0/MCLabel.h"
#include "FT0Raw/RawReaderFT0Base.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(
    ConfigParamSpec{"tcm-extended-mode",
                    o2::framework::VariantType::Bool,
                    false,
                    {"in case of extended TCM mode (1 header + 1 TCMdata + 8 "
                     "TCMdataExtended)"}});

  workflowOptions.push_back(
    ConfigParamSpec{"dump-blocks-reader",
                    o2::framework::VariantType::Bool,
                    false,
                    {"enable dumping of event blocks at reader side"}});
  workflowOptions.push_back(
    ConfigParamSpec{"disable-root-output",
                    o2::framework::VariantType::Bool,
                    false,
                    {"disable root-files output writers"}});
  workflowOptions.push_back(
    ConfigParamSpec{"configKeyValues",
                    o2::framework::VariantType::String,
                    "",
                    {"Semicolon separated key=value strings"}});
  workflowOptions.push_back(
    ConfigParamSpec{"ignore-dist-stf",
                    o2::framework::VariantType::Bool,
                    false,
                    {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(INFO) << "WorkflowSpec defineDataProcessing";
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto isExtendedMode = configcontext.options().get<bool>("tcm-extended-mode");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto askSTFDist = !configcontext.options().get<bool>("ignore-dist-stf");
  LOG(INFO) << "WorkflowSpec FLPWorkflow";
  //Type aliases
  //using RawReaderFT0trgInput = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseNorm,true>;
  using RawReaderFT0 = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseNorm, false>;
  //using RawReaderFT0trgInputExt = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseExt,true>;
  using RawReaderFT0ext = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseExt, false>;
  using MCLabelCont = o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>;
  o2::header::DataOrigin dataOrigin = o2::header::gDataOriginFT0;
  //
  WorkflowSpec specs;
  if (isExtendedMode) {
    specs.emplace_back(o2::fit::getFITDataReaderDPLSpec(RawReaderFT0ext{dataOrigin, dumpReader}, askSTFDist));
    if (!disableRootOut) {
      specs.emplace_back(o2::fit::FITDigitWriterSpecHelper<RawReaderFT0ext, MCLabelCont>::getFITDigitWriterSpec(false, false, dataOrigin));
    }
  } else {
    specs.emplace_back(o2::fit::getFITDataReaderDPLSpec(RawReaderFT0{dataOrigin, dumpReader}, askSTFDist));
    if (!disableRootOut) {
      specs.emplace_back(o2::fit::FITDigitWriterSpecHelper<RawReaderFT0, MCLabelCont>::getFITDigitWriterSpec(false, false, dataOrigin));
    }
  }
  return std::move(specs);
}
