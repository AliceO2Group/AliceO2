// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @file   HeartbeatSampler.h
// @author Matthias Richter
// @since  2017-02-03
// @brief  Heartbeat sampler device

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "DataFlow/HeartbeatSampler.h"
#include "Headers/HeartbeatFrame.h"
#include <options/FairMQProgOptions.h>

void o2::data_flow::HeartbeatSampler::InitTask()
{
  mPeriod = GetConfig()->GetValue<uint32_t>(OptionKeyPeriod);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
}

bool o2::data_flow::HeartbeatSampler::ConditionalRun()
{
  std::this_thread::sleep_for(std::chrono::nanoseconds(mPeriod));

  o2::header::HeartbeatStatistics hbfPayload;

  o2::header::DataHeader dh;
  dh.dataDescription = o2::header::gDataDescriptionHeartbeatFrame;
  dh.dataOrigin = o2::header::DataOrigin("SMPL");
  dh.subSpecification = 0;
  dh.payloadSize = sizeof(hbfPayload);

  // Note: the block type of both header an trailer members of the envelope
  // structure are autmatically initialized to the appropriate block type
  // and size '1' (i.e. only one 64bit word)
  o2::header::HeartbeatFrameEnvelope specificHeader;
  specificHeader.header.orbit = mCount;
  specificHeader.trailer.hbAccept = 1;

  O2Message outgoing;

  // build multipart message from header and payload
  o2::base::addDataBlock(outgoing, {dh, specificHeader}, NewSimpleMessage(hbfPayload));

  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  mCount++;
  return true;
}
