// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DataPublisherDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-10
/// @brief  Utility device for data publishing

#include "Publishers/DataPublisherDevice.h"
#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "Headers/SubframeMetadata.h"
#include <options/FairMQProgOptions.h>

using HeartbeatHeader = o2::header::HeartbeatHeader;
using HeartbeatTrailer = o2::header::HeartbeatTrailer;
using TPCTestCluster = o2::DataFlow::TPCTestCluster;
using ITSRawData = o2::DataFlow::ITSRawData;

using DataDescription = o2::header::DataDescription;
using DataOrigin = o2::header::DataOrigin;

template <typename T>
void fakePayload(std::vector<byte> &buffer, std::function<void(T&,int)> filler, int numOfElements) {
  auto payloadSize = sizeof(T)*numOfElements;
  LOG(INFO) << "Payload size " << payloadSize << "\n";
  buffer.resize(buffer.size() + payloadSize);

  T *payload = reinterpret_cast<T*>(buffer.data() + sizeof(HeartbeatHeader));
  for (int i = 0; i < numOfElements; ++i) {
    new (payload + i) T();
    // put some random toy time stamp to each cluster
    filler(payload[i], i);
  }
}

namespace o2 {
namespace utilities {

DataPublisherDevice::DataPublisherDevice()
  : O2Device()
  , mInputChannelName("input")
  , mOutputChannelName("output")
  , mLastIndex(-1)
  , mDataDescription()
  , mDataOrigin()
  , mSubSpecification(~(SubSpecificationT)0)
  , mFileName()
{
}

DataPublisherDevice::~DataPublisherDevice()
= default;

void DataPublisherDevice::InitTask()
{
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  // TODO: this turned out to be a missing feature in the description fields of
  // the different headers. The idea is to use a field of n bytes as an integer
  // but create the integers from a char sequence. This was thought to be done
  // at compile time, but here we want to have it configurable at runtime.
  //
  // GetConfig()->GetValue<std::string>(OptionKeyDataDescription).c_str()
  // GetConfig()->GetValue<std::string>(OptionKeyDataOrigin).c_str()
  //
  // Still needed:
  // * a registry for descriptors (probably for different descriptor types of
  //   even the same size) to ensure uniqueness
  // * create the unsigned integer value once from the configurable string and
  //   check in the registry
  // * constructors and assignment operators taking the integer type as argument
  if (GetConfig()->GetValue<std::string>(OptionKeyDataDescription) == "TPCCLUSTER")
  {
    mDataDescription = DataDescription("CLUSTERS");
    mDataOrigin = DataOrigin("TPC");
  }
  else if (GetConfig()->GetValue<std::string>(OptionKeyDataDescription) == "ITSRAW")
  {
    mDataDescription = DataDescription("CLUSTERS");
    mDataOrigin = DataOrigin("ITS");
  }
  mSubSpecification = GetConfig()->GetValue<SubSpecificationT>(OptionKeySubspecification);
  mFileName = GetConfig()->GetValue<std::string>(OptionKeyFileName);

  OnData(mInputChannelName.c_str(), &DataPublisherDevice::HandleData);

  // reserve space for the HBH at the beginning
  mFileBuffer.resize(sizeof(o2::header::HeartbeatHeader));

  if (!mFileName.empty()) {
    AppendFile(mFileName.c_str(), mFileBuffer);
  }
  else if (mDataDescription ==  DataDescription("CLUSTERS") 
           && mDataOrigin == DataOrigin("TPC"))
  {
    auto f = [](TPCTestCluster &cluster, int idx) {cluster.timeStamp = idx;};
    fakePayload<TPCTestCluster>(mFileBuffer, f, 1000);
    LOG(INFO) << "Payload size (after) " << mFileBuffer.size() << "\n";
    // For the moment, add the data as another part to this message
  } 
  else if (mDataDescription ==  DataDescription("CLUSTERS") 
           && mDataOrigin == DataOrigin("ITS"))
  {
    auto f = [](ITSRawData &cluster, int idx) {cluster.timeStamp = idx;};
    fakePayload<ITSRawData>(mFileBuffer, f, 500);
  }

  mFileBuffer.resize(mFileBuffer.size() + sizeof(o2::header::HeartbeatTrailer));
  auto* hbhOut = reinterpret_cast<o2::header::HeartbeatHeader*>(&mFileBuffer[0]);
  auto* hbtOut = reinterpret_cast<o2::header::HeartbeatTrailer*>(&mFileBuffer[mFileBuffer.size() - sizeof(o2::header::HeartbeatTrailer)]);
  *hbhOut = o2::header::HeartbeatHeader();
  *hbtOut = o2::header::HeartbeatTrailer();
}

bool DataPublisherDevice::HandleData(FairMQParts& msgParts, int index)
{
  ForEach(msgParts, &DataPublisherDevice::HandleO2LogicalBlock);

  return true;
}

bool DataPublisherDevice::HandleO2LogicalBlock(const byte* headerBuffer,
                                               size_t headerBufferSize,
                                               const byte* dataBuffer,
                                               size_t dataBufferSize)
{
  //  AliceO2::header::hexDump("data buffer", dataBuffer, dataBufferSize);
  const auto* dataHeader = o2::header::get<o2::header::DataHeader>(headerBuffer);
  const auto* hbfEnvelope = o2::header::get<o2::header::HeartbeatFrameEnvelope>(headerBuffer);

  // TODO: not sure what the return value is supposed to indicate, it's
  // not handled in O2Device::ForEach at the moment
  // indicate that the block has not been processed by a 'false'
  if (!dataHeader ||
      (dataHeader->dataDescription) != o2::header::gDataDescriptionHeartbeatFrame) return false;

  if (!hbfEnvelope) {
    LOG(ERROR) << "no heartbeat frame envelope header found";
    return false;
  }

  // TODO: consistency checks
  //  hbfEnvelope->header;
  //  hbfEnvelope->trailer;
  // - block type in both HBH and HBT
  // - HBH size + payload size (specified in HBT) + HBT size == dataBufferSize
  // - dynamically adjust start of the trailer (if this contains more than one
  //   64 bit word

  // TODO: make tool for reading and manipulation of the HeartbeatFrame/Envelop


  // assume everything valid
  // write the HBH and HBT as envelop to the buffer of the file data
  auto* hbhOut = reinterpret_cast<o2::header::HeartbeatHeader*>(&mFileBuffer[0]);
  auto* hbtOut = reinterpret_cast<o2::header::HeartbeatTrailer*>(&mFileBuffer[mFileBuffer.size() - sizeof(o2::header::HeartbeatTrailer)]);

  // copy HBH and HBT, but set the length explicitely to 1
  // TODO: handle all kinds of corner cases, or add an assert
  *hbhOut = hbfEnvelope->header;
  hbhOut->headerLength = 1;
  *hbtOut = hbfEnvelope->trailer;
  hbtOut->dataLength = mFileBuffer.size() - sizeof(o2::header::HeartbeatFrameEnvelope);

  // top level subframe header, the DataHeader is going to be used with
  // configured description, origin and sub specification
  o2::header::DataHeader dh;
  dh.dataDescription = mDataDescription;
  dh.dataOrigin = mDataOrigin;
  dh.subSpecification = mSubSpecification;
  dh.payloadSize = mFileBuffer.size();

  O2Message outgoing;

  LOG(DEBUG) << "Sending buffer of size " << mFileBuffer.size() << "\n";
  LOG(DEBUG) << "Orbit number " << hbhOut->orbit << "\n";
  // build multipart message from header and payload
  // TODO: obviously there is a lot to do here, avoid copying etc, this
  // is just a proof of principle
  // NewSimpleMessage(mFileBuffer) does not work with the vector


  // TODO: fix payload size in dh
  auto *buffer = new char[mFileBuffer.size()];
  memcpy(buffer, mFileBuffer.data(), mFileBuffer.size());
  AddMessage(outgoing, dh, NewMessage(buffer, mFileBuffer.size(),
                        [](void* data, void* hint) { delete[] reinterpret_cast<char *>(data); }, nullptr));

  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  return true;
}

bool DataPublisherDevice::AppendFile(const char* name, std::vector<byte>& buffer)
{
  bool result = true;
  std::ifstream ifile(name, std::ifstream::binary);
  if (ifile.bad()) return false;

  // get length of file:
  ifile.seekg (0, ifile.end);
  int length = ifile.tellg();
  ifile.seekg (0, ifile.beg);

  // allocate memory:
  int position = buffer.size();
  buffer.resize(buffer.size() + length);

  // read data as a block:
  ifile.read(reinterpret_cast<char*>(&buffer[position]),length);
  if (!(result = ifile.good())) {
    LOG(ERROR) << "failed to read " << length << " byte(s) from file " << name << std::endl;
  }

  ifile.close();

  return result;
}

} // namespace utilities
} // namespace o2
