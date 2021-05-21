// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DevicesManager.h"
#include "Framework/RuntimeError.h"
#include "Framework/Logger.h"
#include "Framework/DeviceController.h"

namespace o2::framework
{

void DevicesManager::queueMessage(char const* target, char const* message)
{
  for (int di = 0; di < specs.size(); ++di) {
    if (specs[di].id == target) {
      messages.push_back({di, message});
    }
  }
}

void DevicesManager::flush()
{
  for (auto& handle : messages) {
    auto controller = controls[handle.ref.index].controller;
    // Device might not be started yet, by the time we write to it.
    if (!controller) {
      LOGP(INFO, "Controller for {} not yet available.", specs[handle.ref.index].name);
      continue;
    }
    controller->write(handle.message.c_str(), handle.message.size());
  }

  auto checkIfController = [this](DeviceMessageHandle const& handle) {
    return this->controls[handle.ref.index].controller != nullptr;
  };
  auto it = std::remove_if(messages.begin(), messages.end(), checkIfController);
  auto r = std::distance(it, messages.end());
  messages.erase(it, messages.end());
}

} // namespace o2::framework
