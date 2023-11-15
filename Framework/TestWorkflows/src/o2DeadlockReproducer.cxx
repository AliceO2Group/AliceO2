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

#include <fairmq/Device.h>
#include <fairmq/runDevice.h>

#include <string>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
}

struct Deadlocked : fair::mq::Device {
  Deadlocked()
  {
    auto stateWatcher = [this](fair::mq::State newState) {
      static bool first = true;
      LOG(info) << "State changed to " << newState;
      if (newState != fair::mq::State::Ready) {
        LOG(info) << "Not ready state, ignoring";
        return;
      }
      if (first) {
        first = false;
        return;
      }
      fair::mq::Parts parts;
      LOG(info) << "Draining messages" << std::endl;
      auto& channels = this->GetChannels();
      LOG(info) << "Number of channels: " << channels.size();
      if (channels.size() == 0) {
        LOG(info) << "No messages to drain";
        return;
      }
      auto& channel = channels.at("data")[0];
      while (this->NewStatePending() == false) {
        channel.Receive(parts, 10);
        if (parts.Size() != 0) {
          LOG(info) << "Draining" << parts.Size() << "messages";
        }
      }
      LOG(info) << "Done draining";
    };
    this->SubscribeToStateChange("99-drain", stateWatcher);
  }

  void Run() override
  {
    LOG(info) << "Simply exit";
  }
};

std::unique_ptr<fair::mq::Device> getDevice(fair::mq::ProgOptions& /*config*/)
{
  return std::make_unique<Deadlocked>();
}
