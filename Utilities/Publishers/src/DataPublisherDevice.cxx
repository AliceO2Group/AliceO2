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
  , mFileName()
{
}

AliceO2::Utilities::DataPublisherDevice::~DataPublisherDevice()
{
}

void AliceO2::Utilities::DataPublisherDevice::InitTask()
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
  mDataDescription = AliceO2::Header::DataDescription("FILEDATA");
  mDataOrigin = AliceO2::Header::DataOrigin("TEST");
  mSubSpecification = GetConfig()->GetValue<SubSpecificationT>(OptionKeySubspecification);
  mFileName = GetConfig()->GetValue<std::string>(OptionKeyFileName);

  OnData(mInputChannelName.c_str(), &AliceO2::Utilities::DataPublisherDevice::HandleData);

  // reserve space for the HBH at the beginning
  mFileBuffer.resize(sizeof(AliceO2::Header::HeartbeatHeader));

  if (!mFileName.empty()) {
    AppendFile(mFileName.c_str(), mFileBuffer);
  }
  mFileBuffer.resize(mFileBuffer.size() + sizeof(AliceO2::Header::HeartbeatTrailer));
  auto* hbhOut = reinterpret_cast<AliceO2::Header::HeartbeatHeader*>(&mFileBuffer[0]);
  auto* hbtOut = reinterpret_cast<AliceO2::Header::HeartbeatTrailer*>(&mFileBuffer[mFileBuffer.size() - sizeof(AliceO2::Header::HeartbeatTrailer)]);
  *hbhOut = AliceO2::Header::HeartbeatHeader();
  *hbtOut = AliceO2::Header::HeartbeatTrailer();
}

bool AliceO2::Utilities::DataPublisherDevice::HandleData(FairMQParts& msgParts, int index)
{
  ForEach(msgParts, &DataPublisherDevice::HandleO2LogicalBlock);

  return true;
}

bool AliceO2::Utilities::DataPublisherDevice::HandleO2LogicalBlock(const byte* headerBuffer,
								   size_t headerBufferSize,
								   const byte* dataBuffer,
								   size_t dataBufferSize)
{
  AliceO2::Header::hexDump("data buffer", dataBuffer, dataBufferSize);
  const auto* dataHeader = AliceO2::Header::get<AliceO2::Header::DataHeader>(headerBuffer);
  const auto* hbfEnvelope = AliceO2::Header::get<AliceO2::Header::HeartbeatFrameEnvelope>(headerBuffer);

  // TODO: not sure what the return value is supposed to indicate, it's
  // not handled in O2Device::ForEach at the moment
  // indicate that the block has not been processed by a 'false'
  if (!dataHeader ||
      (dataHeader->dataDescription) != AliceO2::Header::gDataDescriptionHeartbeatFrame) return false;

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
  auto* hbhOut = reinterpret_cast<AliceO2::Header::HeartbeatHeader*>(&mFileBuffer[0]);
  auto* hbtOut = reinterpret_cast<AliceO2::Header::HeartbeatTrailer*>(&mFileBuffer[mFileBuffer.size() - sizeof(AliceO2::Header::HeartbeatTrailer)]);

  // copy HBH and HBT, but set the length explicitely to 1
  // TODO: handle all kinds of corner cases, or add an assert
  *hbhOut = hbfEnvelope->header;
  hbhOut->headerLength = 1;
  *hbtOut = hbfEnvelope->trailer;
  hbtOut->trailerLength;
  hbtOut->dataLength = mFileBuffer.size() - sizeof(AliceO2::Header::HeartbeatFrameEnvelope);

  // top level subframe header, the DataHeader is going to be used with
  // configured description, origin and sub specification
  AliceO2::Header::DataHeader dh;
  dh.dataDescription = mDataDescription;
  dh.dataOrigin = mDataOrigin;
  dh.subSpecification = mSubSpecification;
  dh.payloadSize = mFileBuffer.size();

  O2Message outgoing;

  AliceO2::Header::hexDump("send buffer", &mFileBuffer, mFileBuffer.size());
  // build multipart message from header and payload
  // TODO: obviously there is a lot to do here, avoid copying etc, this
  // is just a proof of principle
  // NewSimpleMessage(mFileBuffer) does not work with the vector
  std::unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
  msg->Rebuild(mFileBuffer.size());
  memcpy(msg->GetData(), &mFileBuffer[0], mFileBuffer.size());
  AddMessage(outgoing, dh, move(msg));

  // send message
  Send(outgoing, mOutputChannelName.c_str());
  outgoing.fParts.clear();

  return true;
}

bool AliceO2::Utilities::DataPublisherDevice::AppendFile(const char* name, std::vector<byte>& buffer)
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
