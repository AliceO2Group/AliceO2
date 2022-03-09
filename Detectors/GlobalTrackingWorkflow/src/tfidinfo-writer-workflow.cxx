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
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"dataspec", VariantType::String, "tfidinfo:FLP/DISTSUBTIMEFRAME/52443", {"spec from which the TFIDInfo will be extracted"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/Task.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "CommonUtils/NameConf.h"
#include <vector>
#include <TFile.h>

class TFIDInfoWriter : public o2::framework::Task
{
 public:
  ~TFIDInfoWriter() override = default;

  void init(o2::framework::InitContext& ic) final
  {
    mOutFileName = ic.options().get<std::string>("tfidinfo-file-name");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true));
    mData.emplace_back(o2::dataformats::TFIDInfo{dh->firstTForbit, dh->tfCounter, dh->runNumber});
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    TFile fl(mOutFileName.c_str(), "recreate");
    fl.WriteObjectAny(&mData, "std::vector<o2::dataformats::TFIDInfo>", "tfidinfo");
    LOGP(info, "Wrote TFIDInfo vector with {} entries to {}", mData.size(), fl.GetName());
  }

 private:
  std::string mOutFileName{};
  std::vector<o2::dataformats::TFIDInfo> mData;
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(DataProcessorSpec{"tfid-info-writer", o2::framework::select(cfgc.options().get<std::string>("dataspec").c_str()),
                                    std::vector<OutputSpec>{}, AlgorithmSpec{adaptFromTask<TFIDInfoWriter>()},
                                    Options{{"tfidinfo-file-name", VariantType::String, o2::base::NameConf::getTFIDInfoFileName(), {"output file for TFIDInfo"}}}});
  return wf;
}
