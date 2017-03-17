// @file   HeartbeatSampler.h
// @author Matthias Richter
// @since  2017-02-03
// @brief  Heartbeat sampler device

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "DataFlow/HeartbeatSampler.h"
#include "Headers/HeartbeatFrame.h"
#include "FairMQProgOptions.h"

void AliceO2::DataFlow::HeartbeatSampler::InitTask()
{
  mPeriod = GetConfig()->GetValue<uint32_t>(OptionKeyPeriod);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
}

bool AliceO2::DataFlow::HeartbeatSampler::ConditionalRun()
{
  std::this_thread::sleep_for(std::chrono::nanoseconds(mPeriod));

  AliceO2::Header::HeartbeatStatistics hbfPayload;

  AliceO2::Header::DataHeader dh;
  dh.dataDescription = AliceO2::Header::gDataDescriptionHeartbeatFrame;
  dh.dataOrigin = AliceO2::Header::DataOrigin("SMPL");
  dh.subSpecification = 0;
  dh.payloadSize = sizeof(hbfPayload);

  // Note: the block type of both header an trailer members of the envelope
  // structure are autmatically initialized to the appropriate block type
  // and size '1' (i.e. only one 64bit word)
  AliceO2::Header::HeartbeatFrameEnvelope specificHeader;
  specificHeader.header.orbit = mCount;
  specificHeader.trailer.hbAccept = 1;

  O2Message outgoing;

  // build multipart message from header and payload
  AddMessage(outgoing, {dh, specificHeader}, NewSimpleMessage(hbfPayload));

  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  mCount++;
  return true;
}
