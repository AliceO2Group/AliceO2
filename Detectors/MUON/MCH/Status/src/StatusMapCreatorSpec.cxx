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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHStatus/StatusMap.h"
#include "MCHStatus/StatusMapCreatorParam.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHConstants/DetectionElements.h"
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
        //create filtered vectors to reject unphysical Solar Id, Elink Id, Channel Id stored in the Bad Channel CCDB
        std::vector<o2::mch::DsChannelId> filteredBadChannels;
        std::vector<o2::mch::DsChannelId> filteredRejectList;
        
        for (const auto& channel : mBadChannels) {
            // Check if the channel has a (SolarId, LinkId, channelId)  that corresponds to a valid Detection Element and valid Pad
            auto group =  o2::mch::raw::groupFromElinkId(channel.getElinkId());
            auto index =  o2::mch::raw::indexFromElinkId(channel.getElinkId());
            o2::mch::raw::DsElecId dsElecId(channel.getSolarId(), group.value(), index.value());
            auto mElec2DetMapper = o2::mch::raw::createElec2DetMapper<o2::mch::raw::ElectronicMapperGenerated>();
            std::optional<o2::mch::raw::DsDetId> dsDetId = mElec2DetMapper(dsElecId);
            if(!dsDetId){
                LOGP(info, "No detection element associated to this channel from Bad Channels CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     channel.getSolarId(), channel.getElinkId(), channel.getChannel());
                continue;}
            auto deId = dsDetId->deId();
            if(!deId){
                LOGP(info, "No detection element id associated to this channel from Bad Channels CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     channel.getSolarId(), channel.getElinkId(), channel.getChannel());
                continue;}
            const auto& seg = o2::mch::mapping::segmentation(deId);
            auto dsId = dsDetId->dsId();
            if(!dsId){
                LOGP(info, "No ds element associated to this channel from Bad Channels CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     channel.getSolarId(), channel.getElinkId(), channel.getChannel());
                continue;}
            int dePadIndex = seg.findPadByFEE(dsId, channel.getChannel());
            if (!seg.isValid(dePadIndex)){
                LOGP(info, "No Valid Pad index associated to this channel from Bad Channels CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     channel.getSolarId(), channel.getElinkId(), channel.getChannel());
                continue;
            }
            
            
            LOGP(info, "List of Valid Channels from Bad Channels CCDB - SolarId: {} ElinkId: {} Channel: {}. This channel is kept",
                 channel.getSolarId(), channel.getElinkId(), channel.getChannel());
            filteredBadChannels.push_back(channel);
        }
        
        
        for (const auto& rlchannel : mRejectList) {
            // Check if the channel has a (SolarId, LinkId, channelId)  that corresponds to a valid Detection Element and valid pad
            auto rlgroup =  o2::mch::raw::groupFromElinkId(rlchannel.getElinkId());
            auto rlindex =  o2::mch::raw::indexFromElinkId(rlchannel.getElinkId());
            o2::mch::raw::DsElecId rldsElecId(rlchannel.getSolarId(), rlgroup.value(), rlindex.value());
            auto rlmElec2DetMapper = o2::mch::raw::createElec2DetMapper<o2::mch::raw::ElectronicMapperGenerated>();
            std::optional<o2::mch::raw::DsDetId> rldsDetId = rlmElec2DetMapper(rldsElecId);
            if(!rldsDetId){
                LOGP(info, "No detection element associated to this channel from Reject List CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     rlchannel.getSolarId(), rlchannel.getElinkId(), rlchannel.getChannel());
                continue;}
            auto rldeId = rldsDetId->deId();
            if(!rldeId){
                LOGP(info, "No detection element id associated to this channel from Reject List CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     rlchannel.getSolarId(), rlchannel.getElinkId(), rlchannel.getChannel());
                continue;}
            const auto& rlseg = o2::mch::mapping::segmentation(rldeId);
            auto rldsId = rldsDetId->dsId();
            if(!rldsId){
                LOGP(info, "No ds element associated to this channel from Reject List CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     rlchannel.getSolarId(), rlchannel.getElinkId(), rlchannel.getChannel());
                continue;}
            int rldePadIndex = rlseg.findPadByFEE(rldsId, rlchannel.getChannel());
            if (!rlseg.isValid(rldePadIndex)){
                LOGP(info, "No Valid Pad index associated to this channel from Reject List CCDB - SolarId: {} ElinkId: {} Channel: {}. Skip this channel",
                     rlchannel.getSolarId(), rlchannel.getElinkId(), rlchannel.getChannel());
                continue;
            }
            
            
            LOGP(info, "List of Valid Channels from Reject List CCDB - SolarId: {} ElinkId: {} Channel: {}. This channel is kept",
                 rlchannel.getSolarId(), rlchannel.getElinkId(), rlchannel.getChannel());
            filteredRejectList.push_back(rlchannel);
        }
        
        
        // Update the StatusMap with the filtered vector
        mStatusMap.add(filteredBadChannels, StatusMap::kBadPedestal);
        mStatusMap.add(filteredRejectList, StatusMap::kRejectList);
        
        mStatusMapUpdated = true;
    }

  void
    init(InitContext& ic)
  {
    mUseBadChannels = StatusMapCreatorParam::Instance().useBadChannels;
    mUseRejectList = StatusMapCreatorParam::Instance().useRejectList;
    mBadChannels.clear();
    mRejectList.clear();
    mStatusMapUpdated = true;
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
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

    if (mStatusMapUpdated) {
      // create the output message
      LOGP(info, "Sending updated StatusMap of size {}", size(mStatusMap));
      pc.outputs().snapshot(OutputRef{"statusmap"}, mStatusMap);
      mStatusMapUpdated = false;
    } else {
      LOGP(info, "Sending unchanged StatusMap of size {}", size(mStatusMap));
      pc.outputs().snapshot(OutputRef{"statusmap"}, mStatusMap);
    }
  }

 private:
  bool mUseBadChannels{false};
  bool mUseRejectList{false};
  std::vector<o2::mch::DsChannelId> mBadChannels;
  std::vector<o2::mch::DsChannelId> mRejectList;
  StatusMap mStatusMap;
  bool mStatusMapUpdated{false};
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

  std::string output = "statusmap:MCH/STATUSMAP/0";

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  if (input.empty()) {
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
    Inputs{select(input.c_str())},
    outputs,
    AlgorithmSpec{adaptFromTask<StatusMapCreatorTask>()},
    Options{}};
}
} // namespace o2::mch
