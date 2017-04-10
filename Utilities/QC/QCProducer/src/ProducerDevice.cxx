#include <chrono>
#include <ctime>
#include <thread>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <FairMQLogger.h>
#include <TMessage.h>

#include "QCProducer/ProducerDevice.h"

using namespace std;
using namespace dds;
using namespace dds::intercom_api;

namespace o2
{
namespace qc
{
ProducerDevice::ProducerDevice(const char* producerId, const int numIoThreads, shared_ptr<Producer>& producer)
  : ddsCustomCmd(new CCustomCmd(mService))
{
  this->SetTransport("zeromq");
  this->SetProperty(Id, producerId);
  this->SetProperty(NumIoThreads, numIoThreads);
  mProducer = producer;
  lastCheckedSecond = getCurrentSecond();
}

void ProducerDevice::deleteTMessage(void* data, void* hint) { delete static_cast<TMessage*>(hint); }
void ProducerDevice::Run()
{
  while (CheckCurrentState(RUNNING)) {
    TObject* newDataObject = mProducer->produceData();
    auto* message = new TMessage(kMESS_OBJECT);
    message->WriteObject(newDataObject);

    unique_ptr<FairMQMessage> request(NewMessage(message->Buffer(), message->BufferSize(), deleteTMessage, message));

    if (outputLimitReached()) {
      waitForLimitUnlock();
    }

    ++sentObjectsInCurrentSecond;
    sendDataToMerger(move(request));

    delete newDataObject;
  }
}

bool ProducerDevice::outputLimitReached()
{
  bool output;
  int now = getCurrentSecond();

  if (now == lastCheckedSecond && sentObjectsInCurrentSecond > OUTPUT_LIMIT_PER_SECOND - 1) {
    output = true;
  } else {
    output = false;
  }

  if (now != lastCheckedSecond) {
    sentObjectsInCurrentSecond = 0;
  }
  lastCheckedSecond = now;

  return output;
}

void ProducerDevice::waitForLimitUnlock()
{
  while (getCurrentSecond() == lastCheckedSecond) {
    this_thread::sleep_for(chrono::milliseconds(10));
  }

  lastCheckedSecond = getCurrentSecond();
  sentObjectsInCurrentSecond = 0;
}

int ProducerDevice::getCurrentSecond() const
{
  using namespace std::chrono;

  auto now = system_clock::now().time_since_epoch();

  return duration_cast<std::chrono::seconds>(now).count();
}

void ProducerDevice::sendDataToMerger(unique_ptr<FairMQMessage> request)
{
  if (fChannels.at("data-out").at(0).SendAsync(request) == -2) {
    mLastBufferOverloadTime = clock();
    mBufferOverloaded = true;
    LOG(DEBUG) << "Buffer of data-out channel is full. Waiting for free buffer...";

    while (fChannels.at("data-out").at(0).SendAsync(request) == -2) {
      this_thread::sleep_for(chrono::milliseconds(10));
    }

    mBufferOverloaded = false;
    LOG(DEBUG) << "Buffer was released after " << double(clock() - mLastBufferOverloadTime) / CLOCKS_PER_SEC
               << " seconds.";
  }
}

void ProducerDevice::subscribeDdsCommands()
{
  using namespace dds::intercom_api;

  mService.subscribeOnError([](const EErrorCode _errorCode, const string& _errorMsg) {
    LOG(ERROR) << "Error received with code: " << _errorCode << ", message: " << _errorMsg;
  });

  ddsCustomCmd->subscribe([&](const string& command, const string& condition, uint64_t senderId) {
    using namespace boost::property_tree;
    using namespace boost::posix_time;

    stringstream requestStream;
    requestStream << command;
    ptree request;
    read_json(requestStream, request);

    if (request.get<string>("command") == "check-state") {
      ptree response;
      response.put("command", "check-state");
      response.put("node_id", GetProperty(Id, "error"));
      response.put("node_state", GetCurrentStateName());
      response.put("internal_message_id", to_string(mInternalStateMessageId));
      response.put("request_timestamp", request.get<string>("requestTimestamp"));
      response.put("response_timestamp", to_iso_extended_string(second_clock::local_time()).substr(0, 19));
      response.put("receive_buffer_size", to_string(0));
      response.put("receive_buffer_overloaded", "false");
      response.put("send_buffer_size", to_string(fChannels.at("data-out").at(0).GetSndBufSize()));
      response.put("send_buffer_overloaded", mBufferOverloaded ? "true" : "false");

      mInternalStateMessageId++;

      stringstream responseStream;
      write_json(responseStream, response);

      ddsCustomCmd->send(responseStream.str(), to_string(senderId));
    }
  });

  mService.start();
}

void ProducerDevice::establishChannel(std::string type, std::string method, std::string address,
                                      std::string channelName, const int bufferSize)
{
  FairMQChannel requestChannel(type, method, address);
  requestChannel.UpdateSndBufSize(bufferSize);
  requestChannel.UpdateRcvBufSize(bufferSize);
  requestChannel.UpdateRateLogging(1);
  fChannels[channelName].push_back(requestChannel);
}

void ProducerDevice::executeRunLoop()
{
  ChangeState("INIT_DEVICE");
  WaitForInitialValidation();
  WaitForEndOfState("INIT_DEVICE");

  subscribeDdsCommands();

  ChangeState("INIT_TASK");
  WaitForEndOfState("INIT_TASK");

  ChangeState("RUN");
  WaitForEndOfState("RUN");

  ChangeState("RESET_TASK");
  WaitForEndOfState("RESET_TASK");

  ChangeState("RESET_DEVICE");
  WaitForEndOfState("RESET_DEVICE");

  ChangeState("END");
}
}
}