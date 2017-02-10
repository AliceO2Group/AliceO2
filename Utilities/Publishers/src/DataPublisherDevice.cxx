/// @file   DataPublisherDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-10
/// @brief  Utility device for data publishing

#include "Publishers/DataPublisherDevice.h"
#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "FairMQProgOptions.h"

AliceO2::Utilities::DataPublisherDevice::DataPublisherDevice()
  : O2Device()
  , mInputChannelName("input")
  , mOutputChannelName("output")
  , mLastIndex(-1)
  , mDataDescription()
  , mDataOrigin()
  , mSubSpecification(~(SubSpecificationT)0)
{
}

AliceO2::Utilities::DataPublisherDevice::~DataPublisherDevice()
{
}

void AliceO2::Utilities::DataPublisherDevice::InitTask()
{
  mInputChannelName = fConfig->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = fConfig->GetValue<std::string>(OptionKeyOutputChannelName);
  mDataDescription = AliceO2::Header::DataDescription(fConfig->GetValue<std::string>(OptionKeyDataDescription).c_str());
  mDataOrigin = AliceO2::Header::DataOrigin(fConfig->GetValue<std::string>(OptionKeyDataOrigin).c_str());
  mSubSpecification = fConfig->GetValue<SubSpecificationT>(OptionKeySubspecification);

  OnData(mInputChannelName.c_str(), &AliceO2::Utilities::DataPublisherDevice::HandleData);
}

bool AliceO2::Utilities::DataPublisherDevice::HandleData(FairMQParts& msgParts, int index)
{
  // top level subframe header, the DataHeader is going to be used with
  // description "SUBTIMEFRAMEMETA"
  // this should be defined in a common place, and also the origin
  // the origin can probably name a detector identifier, but not sure if
  // all CRUs of a FLP in all cases serve a single detector
  AliceO2::Header::DataHeader dh;
  dh.dataDescription = mDataDescription;
  dh.dataOrigin = mDataOrigin;
  dh.subSpecification = mSubSpecification;
  dh.payloadSize = 0;

  O2Message outgoing;

  // build multipart message from header and payload
  AddMessage(outgoing, dh, NULL);

  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  return true;
}
