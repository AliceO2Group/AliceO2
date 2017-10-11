// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SubTimeFrameTransporter/SubTimeFrameTransporterDevice.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameVisitors.h"

#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>

namespace o2 {
namespace DataDistribution {

StfHandlerDevice::StfHandlerDevice() : O2Device{}
{
}

StfHandlerDevice::~StfHandlerDevice()
{
}

void StfHandlerDevice::InitTask()
{
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mFreeShmChannelName = GetConfig()->GetValue<std::string>(OptionKeyFreeShmChannelName);
}

bool StfHandlerDevice::ConditionalRun()
{
  static auto sStartTime = std::chrono::high_resolution_clock::now();
  O2SubTimeFrame lStf;

#if STF_SERIALIZATION == 1
  InterleavedHdrDataDeserializer lStfReceiver(*this, mInputChannelName, 0);
  if (!lStfReceiver.deserialize(lStf)) {
    LOG(WARN) << "Error while receiving of a STF. Exiting...";
    return false;
  }
#elif STF_SERIALIZATION == 2
  HdrDataDeserializer lStfReceiver(*this, mInputChannelName, 0);
  if (!lStfReceiver.deserialize(lStf)) {
    LOG(WARN) << "Error while receiving of a STF. Exiting...";
    return false;
  }
#else
  #error "Unknown STF_SERIALIZATION type"
#endif

  // free region chunks for reuse
  std::map<unsigned, std::vector<FairMQMessagePtr>> lMessagesToReturn;
  lStf.getShmRegionMessages(lMessagesToReturn);

  for (auto& lCruIdMessages : lMessagesToReturn) {
    const unsigned lCruChanId = lCruIdMessages.first;

#if defined(SHM_MULTIPART)
    FairMQParts lMpart;
    lMpart.fParts = std::move(lCruIdMessages.second);
    if (Send(lMpart, mFreeShmChannelName, lCruChanId) < 0) {
      LOG(WARN) << "Returning of region chunks failed. Exiting...";
      return false;
    }
#else
    // TODO: these should be sent interleaved to avoid overloading one channel
    for (auto& lMsg : lCruIdMessages.second) {
      if (Send(lMsg, mFreeShmChannelName, lCruChanId) < 0) {
        LOG(WARN) << "Returning of region chunks failed. Exiting...";
        return false;
      }
    }
#endif
  }

  return true;
}
}
} /* namespace o2::DataDistribution */
