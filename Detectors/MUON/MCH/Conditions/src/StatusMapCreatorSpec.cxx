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

#include "MCHConditions/StatusMapCreatorSpec.h"

#include "DataFormatsMCH/DsChannelId.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHConditions/StatusMap.h"
#include "MCHConditions/StatusMapCreatorParam.h"
#include <fmt/format.h>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace o2::framework;

namespace o2::mch
{

class StatusMapCreatorTask
{
 public:
  StatusMapCreatorTask() = default;

  void updateStatusMap()
  {
    mStatusMap.clear();
    mStatusMap.add(mBadChannels, StatusMap::kBadPedestal);
    mStatusMap.add(mRejectList, StatusMap::kRejectList);
  }

  void
    init(InitContext& ic)
  {
    mUseBadChannels = StatusMapCreatorParam::Instance().useBadChannels;
    mUseRejectList = StatusMapCreatorParam::Instance().useRejectList;
    mBadChannels.clear();
    mRejectList.clear();
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    std::vector<o2::mch::DsChannelId>* bad{nullptr};
    std::vector<o2::mch::DsChannelId>* rl{nullptr};

    if (matcher == ConcreteDataMatcher("MCH", "BADCHANNELS", 0)) {
      auto bad = static_cast<std::vector<o2::mch::DsChannelId>*>(obj);
      mBadChannels = *bad;
      updateStatusMap();
    }
    if (matcher == ConcreteDataMatcher("MCH", "REJECTLIST", 0)) {
      auto rl = static_cast<std::vector<o2::mch::DsChannelId>*>(obj);
      mRejectList = *rl;
      updateStatusMap();
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

    // create the output message
    pc.outputs().snapshot(OutputRef{"statusmap"}, mStatusMap);
  }

 private:
  bool mUseBadChannels{false};
  bool mUseRejectList{false};
  std::vector<o2::mch::DsChannelId> mBadChannels;
  std::vector<o2::mch::DsChannelId> mRejectList;
  StatusMap mStatusMap;
};

framework::DataProcessorSpec getStatusMapCreatorSpec(std::string_view specName)
{
  std::string input;

  if (StatusMapCreatorParam::Instance().useBadChannels) {
    input = "badchannels:MCH/BADCHANNELS/0?lifetime=condition&ccdb-path=MCH/Calib/BadChannel";
  }
  if (StatusMapCreatorParam::Instance().useRejectList) {
    if (!input.empty()) {
      input += ";";
    }
    input += "rejectlist:MCH/REJECTLIST/0?lifetime=condition&ccdb-path=MCH/Calib/RejectList";
  }

  if (input.empty()) {
    return {"mch-dummy-statusmap-creator",
            {},
            {},
            AlgorithmSpec::dummyAlgorithm()};
  }

  std::string output = "statusmap:MCH/STATUSMAP/0";

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    specName.data(),
    Inputs{select(input.c_str())},
    outputs,
    AlgorithmSpec{adaptFromTask<StatusMapCreatorTask>()},
    Options{}};
}
} // namespace o2::mch
