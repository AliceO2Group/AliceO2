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
#include <string>
#include "Algorithm/RangeTokenizer.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCFourierTransformAggregatorSpec.h"
#include "TPCCalibration/IDCFactorization.h"
#include "TPCCalibration/IDCFourierTransform.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-idc-aggregator-ft.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"timeframes", VariantType::Int, 2000, {"Number of TFs which will be aggregated per aggregation interval."}},
    {"rangeIDC", VariantType::Int, 200, {"Number of 1D-IDCs which will be used for the calculation of the fourier coefficients. TODO ALREADY SET IN ABERAGEGROUP"}},
    {"nFourierCoeff", VariantType::Int, 60, {"Number of fourier coefficients (real+imag) which will be stored in the CCDB. The maximum can be 'rangeIDC + 2'."}},
    {"nthreads", VariantType::Int, 1, {"Number of threads which will be used during the calculation of the fourier coefficients."}},
    {"debug", VariantType::Bool, false, {"create debug files"}},
    {"sendOutput", VariantType::Bool, false, {"send IDC0, IDC1, IDCDelta, fourier coefficients (for debugging)"}},
    {"use-naive-fft", VariantType::Bool, false, {"using naive fourier transform (true) or FFTW (false)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcaggregate1didc_configuration.ini");

  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto nCRUs = tpcCRUs.size();
  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto debug = config.options().get<bool>("debug");
  const auto sendOutput = config.options().get<bool>("sendOutput");
  const bool fft = config.options().get<bool>("use-naive-fft");
  const auto rangeIDC = static_cast<unsigned int>(config.options().get<int>("rangeIDC"));
  const auto nFourierCoeff = std::clamp(static_cast<unsigned int>(config.options().get<int>("nFourierCoeff")), static_cast<unsigned int>(0), rangeIDC + 2);
  const auto nthreadsFourier = static_cast<unsigned long>(config.options().get<int>("nthreads"));
  TPCFourierTransformAggregatorSpec::IDCFType::setNThreads(nthreadsFourier);
  TPCFourierTransformAggregatorSpec::IDCFType::setFFT(!fft);

  const auto first = tpcCRUs.begin();
  const auto last = std::min(tpcCRUs.end(), first + nCRUs);
  const std::vector<uint32_t> rangeCRUs(first, last);
  WorkflowSpec workflow{getTPCFourierTransformAggregatorSpec(rangeCRUs, timeframes, rangeIDC, nFourierCoeff, debug, sendOutput)};
  return workflow;
}
