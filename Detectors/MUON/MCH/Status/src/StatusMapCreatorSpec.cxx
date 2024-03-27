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

#include "MCHStatus/StatusMapCreatorSpec.h"

#include "DataFormatsMCH/DsChannelId.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Task.h"
#include "Framework/TimingInfo.h"
#include "Framework/WorkflowSpec.h"
#include "MCHStatus/HVStatusCreator.h"
#include "MCHStatus/StatusMap.h"
#include "MCHStatus/StatusMapCreatorParam.h"
#include <fmt/format.h>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace o2::framework;

namespace o2::mch
{

size_t size(const StatusMap& sm)
{
  size_t n{0};

  for (const auto& p : sm) {
    ++n;
  }
  return n;
}

class StatusMapCreatorTask
{
 public:
  StatusMapCreatorTask() = default;

  void updateStatusMap()
  {
    mStatusMap.clear();
    mStatusMap.add(mBadChannels, StatusMap::kBadPedestal);
    mStatusMap.add(mRejectList, StatusMap::kRejectList);
    mHVStatusCreator.updateStatusMap(mStatusMap);
    mUpdateStatusMap = false;
  }

  void init(InitContext& ic)
  {
    mUseBadChannels = StatusMapCreatorParam::Instance().useBadChannels;
    mUseRejectList = StatusMapCreatorParam::Instance().useRejectList;
    mUseHV = StatusMapCreatorParam::Instance().useHV;
    mBadChannels.clear();
    mRejectList.clear();
    mHVStatusCreator.clear();
    mStatusMap.clear();
    mUpdateStatusMap = false;
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher("MCH", "BADCHANNELS", 0)) {
      auto bad = static_cast<std::vector<o2::mch::DsChannelId>*>(obj);
      mBadChannels = *bad;
      mUpdateStatusMap = true;
    } else if (matcher == ConcreteDataMatcher("MCH", "REJECTLIST", 0)) {
      auto rl = static_cast<std::vector<o2::mch::DsChannelId>*>(obj);
      mRejectList = *rl;
      mUpdateStatusMap = true;
    } else if (matcher == ConcreteDataMatcher("MCH", "HV", 0)) {
      auto hv = static_cast<o2::mch::HVStatusCreator::DPMAP*>(obj);
      mHVStatusCreator.findBadHVs(*hv);
    }
  }

  void run(ProcessingContext& pc)
  {
    if (mUseBadChannels) {
      // to trigger call to finaliseCCDB
      pc.inputs().get<std::vector<o2::mch::DsChannelId>*>("badchannels");
    }

    if (mUseRejectList) {
      // to trigger call to finaliseCCDB
      pc.inputs().get<std::vector<o2::mch::DsChannelId>*>("rejectlist");
    }

    if (mUseHV) {
      // to trigger call to finaliseCCDB
      pc.inputs().get<o2::mch::HVStatusCreator::DPMAP*>("hv");

      // check for update of bad HV channels
      auto timestamp = pc.services().get<o2::framework::TimingInfo>().creation;
      if (mHVStatusCreator.findCurrentBadHVs(timestamp)) {
        mUpdateStatusMap = true;
      }
    }

    // update the status map if needed
    if (mUpdateStatusMap) {
      updateStatusMap();
      LOGP(info, "Sending updated StatusMap of size {}", size(mStatusMap));
    } else {
      LOGP(info, "Sending unchanged StatusMap of size {}", size(mStatusMap));
    }

    // create the output message
    pc.outputs().snapshot(OutputRef{"statusmap"}, mStatusMap);
  }

 private:
  bool mUseBadChannels{false};
  bool mUseRejectList{false};
  bool mUseHV{false};
  std::vector<o2::mch::DsChannelId> mBadChannels{};
  std::vector<o2::mch::DsChannelId> mRejectList{};
  HVStatusCreator mHVStatusCreator{};
  StatusMap mStatusMap{};
  bool mUpdateStatusMap{false};
};

framework::DataProcessorSpec getStatusMapCreatorSpec(std::string_view specName)
{
  std::vector<InputSpec> inputs{};
  if (StatusMapCreatorParam::Instance().useBadChannels) {
    inputs.emplace_back(InputSpec{"badchannels", "MCH", "BADCHANNELS", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/BadChannel")});
  }
  if (StatusMapCreatorParam::Instance().useRejectList) {
    inputs.emplace_back(InputSpec{"rejectlist", "MCH", "REJECTLIST", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/RejectList")});
  }
  if (StatusMapCreatorParam::Instance().useHV) {
    inputs.emplace_back(InputSpec{"hv", "MCH", "HV", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/HV", {}, 1)}); // query every TF
  }

  std::vector<OutputSpec> outputs{};
  outputs.emplace_back(OutputSpec{{"statusmap"}, "MCH", "STATUSMAP", 0, Lifetime::Timeframe});

  if (inputs.empty()) {
    return DataProcessorSpec{
      specName.data(),
      {},
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
          ctx.services().get<ControlService>().endOfStream();
        }},
      Options{}};
  }

  return DataProcessorSpec{
    specName.data(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<StatusMapCreatorTask>()},
    Options{}};
}
} // namespace o2::mch
