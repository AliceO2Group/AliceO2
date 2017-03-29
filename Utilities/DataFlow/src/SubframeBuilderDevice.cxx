/// @file   SubframeBuilderDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Demonstrator device for a subframe builder

#include <thread> // this_thread::sleep_for
#include <chrono>
#include <functional>

#include "DataFlow/SubframeBuilderDevice.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/HeartbeatFrame.h"
#include "Headers/DataHeader.h"
#include "FairMQProgOptions.h"

// temporary hack to solve issue 279; linking problem for this constexpr
// variable
constexpr uint32_t AliceO2::DataFlow::SubframeBuilderDevice::mDuration;

using HeartbeatHeader = AliceO2::Header::HeartbeatHeader;
using HeartbeatTrailer = AliceO2::Header::HeartbeatTrailer;
using DataHeader = AliceO2::Header::DataHeader;

AliceO2::DataFlow::SubframeBuilderDevice::SubframeBuilderDevice()
  : O2Device()
{
  LOG(INFO) << "AliceO2::DataFlow::SubframeBuilderDevice::SubframeBuilderDevice " << mDuration << "\n";
}

AliceO2::DataFlow::SubframeBuilderDevice::~SubframeBuilderDevice()
= default;

void AliceO2::DataFlow::SubframeBuilderDevice::InitTask()
{
//  mDuration = GetConfig()->GetValue<uint32_t>(OptionKeyDuration);
  mIsSelfTriggered = GetConfig()->GetValue<bool>(OptionKeySelfTriggered);
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);

  if (!mIsSelfTriggered) {
    // depending on whether the device is self-triggered or expects input,
    // the handler function needs to be registered or not.
    // ConditionalRun is not called anymore from the base class if the
    // callback is registered
    LOG(INFO) << "Obtaining data from DataPublisher\n";
    OnData(mInputChannelName.c_str(), &AliceO2::DataFlow::SubframeBuilderDevice::HandleData);
  } else {
    LOG(INFO) << "Self triggered mode. Doing nothing for now.\n";
  }
  LOG(INFO) << "AliceO2::DataFlow::SubframeBuilderDevice::InitTask " << mDuration << "\n";
}

// FIXME: how do we actually find out the payload size???
int64_t extractDetectorPayload(char **payload, char *buffer, size_t bufferSize) {
  *payload = buffer + sizeof(HeartbeatHeader);
  return bufferSize - sizeof(HeartbeatHeader) - sizeof(HeartbeatTrailer);
}

bool AliceO2::DataFlow::SubframeBuilderDevice::BuildAndSendFrame(FairMQParts &inParts)
{
  LOG(INFO) << "AliceO2::DataFlow::SubframeBuilderDevice::BuildAndSendFrame" << mDuration << "\n";
  char *incomingBuffer = (char *)inParts.At(1)->GetData();
  HeartbeatHeader *hbh = reinterpret_cast<HeartbeatHeader*>(incomingBuffer);

  // top level subframe header, the DataHeader is going to be used with
  // description "SUBTIMEFRAMEMD"
  // this should be defined in a common place, and also the origin
  // the origin can probably name a detector identifier, but not sure if
  // all CRUs of a FLP in all cases serve a single detector
  AliceO2::Header::DataHeader dh;
  dh.dataDescription = AliceO2::Header::DataDescription("SUBTIMEFRAMEMD");
  dh.dataOrigin = AliceO2::Header::DataOrigin("TEST");
  dh.subSpecification = 0;
  dh.payloadSize = sizeof(SubframeMetadata);

  // subframe meta information as payload
  SubframeMetadata md;
  // md.startTime = (hbh->orbit / mOrbitsPerTimeframe) * mDuration;
  md.startTime = static_cast<uint64_t>(hbh->orbit) * static_cast<uint64_t>(mOrbitDuration);
  md.duration = mDuration;
  LOG(INFO) << "Start time for subframe (" << hbh->orbit << ", "
                                           << mDuration
            << ") " << timeframeIdFromTimestamp(md.startTime, mDuration) << " " << md.startTime<< "\n";

  // send an empty subframe (no detector payload), only the data header
  // and the subframe meta data are added to the sub timeframe
  // TODO: this is going to be changed as soon as the device implements
  // handling of the input data
  O2Message outgoing;

  // build multipart message from header and payload
  AddMessage(outgoing, dh, NewSimpleMessage(md));

  char *sourcePayload = nullptr;
  auto payloadSize = extractDetectorPayload(&sourcePayload,
                                            incomingBuffer,
                                            inParts.At(1)->GetSize());
  LOG(INFO) << "Got "  << inParts.Size() << " parts\n";
  for (auto pi = 0; pi < inParts.Size(); ++pi)
  {
    LOG(INFO) << " Part " << pi << ": " << inParts.At(pi)->GetSize() << " bytes \n";
  }
  if (payloadSize <= 0)
  {
    LOG(ERROR) << "Payload is too small: " << payloadSize << "\n";
    return true;
  }
  else
  {
    LOG(INFO) << "Payload of size " << payloadSize << "received\n";
  }

  char *payload = new char[payloadSize]();
  memcpy(payload, sourcePayload, payloadSize);
  DataHeader payloadheader(*AliceO2::Header::get<DataHeader>((byte*)inParts.At(0)->GetData()));

  payloadheader.subSpecification = 0;
  payloadheader.payloadSize = payloadSize;

  // FIXME: take care of multiple HBF per SubtimeFrame
  AddMessage(outgoing, payloadheader,
             NewMessage(payload, payloadSize,
                        [](void* data, void* hint) { delete[] reinterpret_cast<char *>(hint); }, payload));
  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  return true;
}

bool AliceO2::DataFlow::SubframeBuilderDevice::HandleData(FairMQParts& msgParts, int /*index*/)
{
  // loop over header payload pairs in the incoming multimessage
  // for each pair
  // - check timestamp
  // - create new subtimeframe if none existing where the timestamp of the data fits
  // - add pair to the corresponding subtimeframe

  // check for completed subtimeframes and send all completed frames
  // the builder does not implement the routing to the EPN, this is done in the
  // specific FLP-EPN setup
  // to fit into the simple emulation of event/frame ids in the flpSender the order of
  // subtimeframes needs to be preserved
  BuildAndSendFrame(msgParts);
  return true;
}
