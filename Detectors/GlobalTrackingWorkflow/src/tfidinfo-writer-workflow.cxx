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
#include "DetectorsBase/TFIDInfoHelper.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*tfid-info-writer.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/Task.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonUtils/NameConf.h"
#include "CommonConstants/LHCConstants.h"
#include "Framework/CCDBParamSpec.h"
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
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (tinfo.globalRunNumberChanged) { // new run is starting
      auto v = pc.inputs().get<std::vector<Long64_t>*>("orbitReset");
      mOrbitReset = (*v)[0];
    }
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mData.emplace_back());
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    o2::utils::TreeStreamRedirector pcstream;
    TFile fl(mOutFileName.c_str(), "recreate");
    fl.WriteObjectAny(&mData, "std::vector<o2::dataformats::TFIDInfo>", "tfidinfo");
    pcstream.SetFile(&fl);
    for (const auto& info : mData) {
      long ts = (mOrbitReset + long(info.firstTForbit * o2::constants::lhc::LHCOrbitMUS)) / 1000;
      pcstream << "tfidTree"
               << "tfidinfo=" << info << "ts=" << ts << "\n";
    }
    pcstream.Close();
    LOGP(info, "Wrote tfidinfo vector and tfidTree with {} entries to {}", mData.size(), fl.GetName());
  }

 private:
  long mOrbitReset = 0;
  std::string mOutFileName{};
  std::vector<o2::dataformats::TFIDInfo> mData;
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(DataProcessorSpec{"tfid-info-writer",
                                    {{"orbitReset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset")}},
                                    std::vector<OutputSpec>{},
                                    AlgorithmSpec{adaptFromTask<TFIDInfoWriter>()},
                                    Options{{"tfidinfo-file-name", VariantType::String, o2::base::NameConf::getTFIDInfoFileName(), {"output file for TFIDInfo"}}}});
  return wf;
}
