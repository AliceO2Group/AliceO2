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

#include "PrimaryServerState.h"
#include <SimConfig/SimConfig.h>
#include <fairmq/Channel.h>
#include <fairmq/Message.h>
#include <fairlogger/Logger.h>
#include <FairLogger.h>
#include <memory>

namespace o2
{

bool querySimConfig(fair::mq::Channel& channel)
{
  std::unique_ptr<fair::mq::Message> request(channel.NewSimpleMessage((int)O2PrimaryServerInfoRequest::Config));
  std::unique_ptr<fair::mq::Message> reply(channel.NewMessage());

  int timeoutinMS = 60000; // wait for 60s max --> should be fast reply
  if (channel.Send(request, timeoutinMS) > 0) {
    LOG(info) << "Waiting for configuration answer ";
    if (channel.Receive(reply, timeoutinMS) > 0) {
      LOG(info) << "Configuration answer received, containing " << reply->GetSize() << " bytes ";

      // the answer is a TMessage containing the simulation Configuration
      auto message = std::make_unique<TMessageWrapper>(reply->GetData(), reply->GetSize());
      auto config = static_cast<o2::conf::SimConfigData*>(message.get()->ReadObjectAny(message.get()->GetClass()));
      if (!config) {
        return false;
      }

      LOG(info) << "COMMUNICATED ENGINE " << config->mMCEngine;

      auto& conf = o2::conf::SimConfig::Instance();
      conf.resetFromConfigData(*config);
      FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());
      delete config;
    } else {
      LOG(error) << "No configuration received within " << timeoutinMS << "ms\n";
      return false;
    }
  } else {
    LOG(error) << "Could not send configuration request within " << timeoutinMS << "ms\n";
    return false;
  }
  return true;
}

} // namespace o2
