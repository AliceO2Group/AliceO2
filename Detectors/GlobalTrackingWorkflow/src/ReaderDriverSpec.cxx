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

/// @file   ReaderDriverSpec.cxx

#include <vector>
#include <cassert>
#include <fairmq/Device.h>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/RateLimiter.h"
#include "Framework/RawDeviceService.h"
#include "GlobalTrackingWorkflow/ReaderDriverSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "CommonUtils/StringUtils.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class ReadeDriverSpec : public o2::framework::Task
{
 public:
  ReadeDriverSpec(size_t minSHM) : mMinSHM(minSHM) {}
  ~ReadeDriverSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  int mTFRateLimit = -999;
  int mNTF = 0;
  size_t mMinSHM = 0;
};

void ReadeDriverSpec::init(InitContext& ic)
{
  mNTF = ic.options().get<int>("max-tf");
}

void ReadeDriverSpec::run(ProcessingContext& pc)
{
  if (mTFRateLimit == -999) {
    mTFRateLimit = std::stoi(pc.services().get<RawDeviceService>().device()->fConfig->GetValue<std::string>("timeframes-rate-limit"));
  }
  static RateLimiter limiter;
  static int count = 0;
  if (!count) {
    if (o2::raw::HBFUtilsInitializer::NTFs < 0) {
      LOGP(fatal, "Number of TFs to process was not initizalized in the HBFUtilsInitializer");
    }
    mNTF = (mNTF > 0 && mNTF < o2::raw::HBFUtilsInitializer::NTFs) ? mNTF : o2::raw::HBFUtilsInitializer::NTFs;
  } else { // check only for count > 0
    limiter.check(pc, mTFRateLimit, mMinSHM);
  }
  std::vector<char> v{};
  pc.outputs().snapshot(Output{"GLO", "READER_DRIVER", 0}, v);
  if (++count >= mNTF) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getReaderDriverSpec(const std::string& metricChannel, size_t minSHM)
{
  std::vector<OutputSpec> outputSpec;
  std::vector<ConfigParamSpec> options;
  options.emplace_back(ConfigParamSpec{"max-tf", o2::framework::VariantType::Int, -1, {"max TFs to process (<1 : no limits)"}});
  if (!metricChannel.empty()) {
    options.emplace_back(ConfigParamSpec{"channel-config", VariantType::String, metricChannel, {"Out-of-band channel config for TF throttling"}});
  }
  return DataProcessorSpec{
    o2::raw::HBFUtilsInitializer::ReaderDriverDevice,
    Inputs{},
    Outputs{{"GLO", "READER_DRIVER", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ReadeDriverSpec>(minSHM)},
    options};
}

} // namespace globaltracking
} // namespace o2
