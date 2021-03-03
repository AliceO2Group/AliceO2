// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------
// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    {"use-process", VariantType::Bool, false, {"enable processor for data taking/dumping"}},
    {"dump-blocks-process", VariantType::Bool, false, {"enable dumping of event blocks at processor side"}},
    {"dump-blocks-reader", VariantType::Bool, false, {"enable dumping of event blocks at reader side"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"ccdb-url", VariantType::String, "http://ccdb-test.cern.ch:8080", {"url of CCDB"}},
    {"not-check-trigger", VariantType::Bool, true, {"avoid to check trigger condition during conversion"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"
#include "ZDCWorkflow/ZDCDataReaderDPLSpec.h"
#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"
#include "ZDCRaw/RawReaderZDC.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto useProcessor = configcontext.options().get<bool>("use-process");
  auto dumpProcessor = configcontext.options().get<bool>("dump-blocks-process");
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto ccdbURL = configcontext.options().get<std::string>("ccdb-url");
  auto checkTrigger = true;
  auto notCheckTrigger = configcontext.options().get<bool>("not-check-trigger");
  if (notCheckTrigger) {
    LOG(INFO) << "Not checking trigger condition during conversion";
    checkTrigger = false;
  }

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  specs.emplace_back(o2::zdc::getZDCDataReaderDPLSpec(o2::zdc::RawReaderZDC{dumpReader}, ccdbURL, checkTrigger));
  //  if (useProcess) {
  //    specs.emplace_back(o2::zdc::getZDCDataProcessDPLSpec(dumpProcessor));
  //  }
  if (!disableRootOut) {
    specs.emplace_back(o2::zdc::getZDCDigitWriterDPLSpec());
  }
  return std::move(specs);
}
