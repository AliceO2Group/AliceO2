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

  // Do something with the STF

  // Stf Readout messages will be returned when lStf goes of the scope!
  // TODO: remove getsize when fix for region mapping lands
  lStf.getDataSize();

  return true;
}
}
} /* namespace o2::DataDistribution */
