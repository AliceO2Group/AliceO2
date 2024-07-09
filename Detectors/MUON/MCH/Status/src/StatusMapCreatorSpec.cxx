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

#include "CommonConstants/LHCConstants.h"
#include "DataFormatsMCH/DsChannelId.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/CallbackService.h"
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
#include <chrono>
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
  StatusMapCreatorTask(bool useBadChannels, bool useRejectList, bool useHV,
                       std::shared_ptr<base::GRPGeomRequest> ggRequest)
    : mUseBadChannels(useBadChannels),
      mUseRejectList(useRejectList),
      mUseHV(useHV),
      mGGRequest(ggRequest) {}

  void init(InitContext& ic)
  {
    if (mGGRequest) {
      base::GRPGeomHelper::instance().setRequest(mGGRequest);
    }
    mBadChannels.clear();
    mRejectList.clear();
    mHVStatusCreator.clear();
    mStatusMap.clear();
    mUpdateStatusMap = false;

    auto stop = [this]() {
      auto fullTime = mFindBadHVsTime + mFindCurrentBadHVsTime + mUpdateStatusTime;
      LOGP(info, "duration: {:g} ms (findBadHVs: {:g} ms, findCurrentBadHVs: {:g} ms, updateStatusMap: {:g} ms)",
           fullTime.count(), mFindBadHVsTime.count(), mFindCurrentBadHVsTime.count(), mUpdateStatusTime.count());
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
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
      auto tStart = std::chrono::high_resolution_clock::now();
      auto hv = static_cast<o2::mch::HVStatusCreator::DPMAP*>(obj);
      mHVStatusCreator.findBadHVs(*hv);
      mFindBadHVsTime += std::chrono::high_resolution_clock::now() - tStart;
    } else if (mGGRequest) {
      o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    }
  }

  void run(ProcessingContext& pc)
  {
    if (mGGRequest) {
      o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    }

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
      auto tStart = std::chrono::high_resolution_clock::now();
      auto orbitReset = base::GRPGeomHelper::instance().getOrbitResetTimeMS();
      auto firstTForbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
      auto timestamp = orbitReset + static_cast<uint64_t>(firstTForbit * constants::lhc::LHCOrbitMUS * 1.e-3);
      if (mHVStatusCreator.findCurrentBadHVs(timestamp)) {
        LOGP(info, "HV status updated at timestamp {}", timestamp);
        mUpdateStatusMap = true;
      }
      mFindCurrentBadHVsTime += std::chrono::high_resolution_clock::now() - tStart;
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
  std::shared_ptr<base::GRPGeomRequest> mGGRequest{};
  std::vector<o2::mch::DsChannelId> mBadChannels{};
  std::vector<o2::mch::DsChannelId> mRejectList{};
  HVStatusCreator mHVStatusCreator{};
  StatusMap mStatusMap{};
  bool mUpdateStatusMap{false};
  std::chrono::duration<double, std::milli> mFindBadHVsTime{};
  std::chrono::duration<double, std::milli> mFindCurrentBadHVsTime{};
  std::chrono::duration<double, std::milli> mUpdateStatusTime{};

  void updateStatusMap()
  {
    auto tStart = std::chrono::high_resolution_clock::now();
    mStatusMap.clear();
    mStatusMap.add(mBadChannels, StatusMap::kBadPedestal);
    mStatusMap.add(mRejectList, StatusMap::kRejectList);
    mHVStatusCreator.updateStatusMap(mStatusMap);
    mUpdateStatusMap = false;
    mUpdateStatusTime += std::chrono::high_resolution_clock::now() - tStart;
  }
};

framework::DataProcessorSpec getStatusMapCreatorSpec(std::string_view specName)
{
  auto useBadChannels = StatusMapCreatorParam::Instance().useBadChannels;
  auto useRejectList = StatusMapCreatorParam::Instance().useRejectList;
  auto useHV = StatusMapCreatorParam::Instance().useHV;
  std::shared_ptr<base::GRPGeomRequest> ggRequest{};

  std::vector<InputSpec> inputs{};
  if (useBadChannels) {
    inputs.emplace_back(InputSpec{"badchannels", "MCH", "BADCHANNELS", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/BadChannel")});
  }
  if (useRejectList) {
    inputs.emplace_back(InputSpec{"rejectlist", "MCH", "REJECTLIST", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/RejectList")});
  }
  if (useHV) {
    inputs.emplace_back(InputSpec{"hv", "MCH", "HV", 0, Lifetime::Condition, ccdbParamSpec("MCH/Calib/HV", {}, 1)}); // query every TF

    ggRequest = std::make_shared<base::GRPGeomRequest>(true,                       // orbitResetTime
                                                       false,                      // GRPECS=true
                                                       false,                      // GRPLHCIF
                                                       false,                      // GRPMagField
                                                       false,                      // askMatLUT
                                                       base::GRPGeomRequest::None, // geometry
                                                       inputs);
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
    AlgorithmSpec{adaptFromTask<StatusMapCreatorTask>(useBadChannels, useRejectList, useHV, ggRequest)},
    Options{}};
}
} // namespace o2::mch
