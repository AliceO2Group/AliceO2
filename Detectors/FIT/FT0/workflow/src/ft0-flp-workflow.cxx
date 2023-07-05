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
#include "Framework/CompletionPolicyHelpers.h"
#if defined(__has_include)
#if defined(__linux__) && (defined(__x86_64) || defined(__x86_64__)) && __has_include(<emmintrin.h>) && __has_include(<immintrin.h>) && defined(FT0_DECODER_AVX512)
#define FT0_NEW_DECOER_ON
#include "FT0Workflow/FT0DataDecoderDPLSpec.h"
#endif
#endif

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:FT0|ft0).*[W,w]riter.*"));
}
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
  workflowOptions.push_back(
    ConfigParamSpec{"input-sub-sampled",
                    o2::framework::VariantType::Bool,
                    false,
                    {"SUB_RAWDATA DPL channel will be used as input, in case of dispatcher usage"}});
  workflowOptions.push_back(
    ConfigParamSpec{"disable-dpl-ccdb-fetcher",
                    o2::framework::VariantType::Bool,
                    false,
                    {"Disable DPL CCDB fetcher, channel map will be uploaded during initialization by taking last entry in CCDB"}});
#if defined(FT0_NEW_DECOER_ON)
  workflowOptions.push_back(
    ConfigParamSpec{"new-decoder",
                    o2::framework::VariantType::Bool,
                    false,
                    {"New decoder which uses AVX-512 CPU instractions"}});
#endif
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(info) << "WorkflowSpec defineDataProcessing";
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto isExtendedMode = configcontext.options().get<bool>("tcm-extended-mode");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto askSTFDist = !configcontext.options().get<bool>("ignore-dist-stf");
  const auto isSubSampled = configcontext.options().get<bool>("input-sub-sampled");
  const auto disableDplCcdbFetcher = configcontext.options().get<bool>("disable-dpl-ccdb-fetcher");
  bool isNewDecoder = false;
#if defined(FT0_NEW_DECOER_ON)
  isNewDecoder = configcontext.options().get<bool>("new-decoder");
#endif
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  LOG(info) << "WorkflowSpec FLPWorkflow";
  // Type aliases
  // using RawReaderFT0trgInput = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseNorm,true>;
  using RawReaderFT0 = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseNorm, false>;
  // using RawReaderFT0trgInputExt = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseExt,true>;
  using RawReaderFT0ext = o2::fit::RawReaderFIT<o2::ft0::RawReaderFT0BaseExt, false>;
  using MCLabelCont = o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>;
  o2::header::DataOrigin dataOrigin = o2::header::gDataOriginFT0;
  //
  WorkflowSpec specs;
  if (!isNewDecoder) {
    if (isExtendedMode) {
      specs.emplace_back(o2::fit::getFITDataReaderDPLSpec(RawReaderFT0ext{dataOrigin, dumpReader}, askSTFDist, isSubSampled, disableDplCcdbFetcher));
      if (!disableRootOut) {
        specs.emplace_back(o2::fit::FITDigitWriterSpecHelper<RawReaderFT0ext, MCLabelCont>::getFITDigitWriterSpec(false, false, dataOrigin));
      }
    } else {
      specs.emplace_back(o2::fit::getFITDataReaderDPLSpec(RawReaderFT0{dataOrigin, dumpReader}, askSTFDist, isSubSampled, disableDplCcdbFetcher));
      if (!disableRootOut) {
        specs.emplace_back(o2::fit::FITDigitWriterSpecHelper<RawReaderFT0, MCLabelCont>::getFITDigitWriterSpec(false, false, dataOrigin));
      }
    }
  } else {
#if defined(FT0_NEW_DECOER_ON)
    specs.emplace_back(o2::ft0::getFT0DataDecoderDPLSpec(askSTFDist));
#endif
  }
  return std::move(specs);
}
