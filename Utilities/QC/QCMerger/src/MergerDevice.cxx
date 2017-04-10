#include <ratio>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <FairMQLogger.h>

#include <dds_intercom.h>

#include "QCCommon/TMessageWrapper.h"
#include "QCMerger/MergerDevice.h"

using namespace std;
using namespace std::chrono;
using namespace boost::property_tree;
using namespace dds;
using namespace dds::intercom_api;

namespace o2
{
namespace qc
{
MergerDevice::MergerDevice(unique_ptr<Merger> merger, string mergerId, int numIoThreads)
  : mMerger(move(merger)), ddsCustomCmd(new CCustomCmd(mService))
{
  this->SetTransport("zeromq");
  this->SetProperty(Id, mergerId);
  this->SetProperty(NumIoThreads, numIoThreads);

  procSelfStatus.open("/proc/self/status");

  calculateCpuUsage();
  calculateNumberOfMergedObjectsPerSecond();
}

MergerDevice::~MergerDevice() { procSelfStatus.close(); }
void MergerDevice::deleteTMessage(void* data, void* hint) { delete static_cast<TMessage*>(hint); }
void MergerDevice::establishChannel(string type, string method, string address, string channelName, int receiveBuffer,
                                    int sendBuffer)
{
  FairMQChannel channel(type, method, address);
  channel.UpdateSndBufSize(sendBuffer);
  channel.UpdateRcvBufSize(receiveBuffer);
  channel.UpdateRateLogging(1);
  fChannels[channelName].push_back(channel);
}

void MergerDevice::executeRunLoop()
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

void MergerDevice::subscribeDdsCommands()
{
  subscribeOnError();
  subscribeCustomCommands();
  mService.start();
}

void MergerDevice::subscribeOnError()
{
  mService.subscribeOnError([](const dds::intercom_api::EErrorCode _errorCode, const string& _errorMsg) {
    LOG(ERROR) << "Error received: error code: " << _errorCode << ", error message: " << _errorMsg;
  });
}

void MergerDevice::subscribeCustomCommands()
{
  ddsCustomCmd->subscribe([&](const string& command, const string& condition, uint64_t senderId) {
    stringstream ss;
    ptree request;

    ss << command;
    read_json(ss, request);

    if (request.get<string>("command") == "check-state") {
      const ptree response = createCheckStateResponse(request);
      sendControlResponse(response, to_string(senderId));
    } else if (request.get<string>("command") == "get-metrics") {
      const ptree response = createGetMetricsResponse(request);
      sendControlResponse(response, to_string(senderId));
    }
  });
}

boost::property_tree::ptree MergerDevice::createCheckStateResponse(const ptree& request)
{
  using namespace boost::posix_time;

  ptree response;
  response.put("command", "state");
  response.put("node_id", GetProperty(Id, "error"));
  response.put("node_state", GetCurrentStateName());
  response.put("internal_message_id", to_string(mInternalStateMessageId));
  response.put("request_timestamp", request.get<string>("requestTimestamp"));
  response.put("response_timestamp", to_iso_extended_string(second_clock::local_time()).substr(0, 19));
  response.put("receive_buffer_size", to_string(fChannels.at("data-in").at(0).GetRcvBufSize()));
  response.put("receive_buffer_overloaded", mReceiveBufferOverloaded ? "true" : "false");
  response.put("send_buffer_size", to_string(fChannels.at("data-out").at(0).GetSndBufSize()));
  response.put("send_buffer_overloaded", mReceiveBufferOverloaded ? "true" : "false");

  mInternalStateMessageId++;

  return response;
}

boost::property_tree::ptree MergerDevice::createGetMetricsResponse(const ptree& request)
{
  using namespace boost::posix_time;

  ptree response;
  response.put("command", "metrics");
  response.put("node_id", GetProperty(Id, "error"));
  response.put("PID", getpid());
  response.put("internal_message_id", to_string(mInternalMetricMessageId));
  response.put("request_timestamp", request.get<string>("requestTimestamp"));
  response.put("response_timestamp", to_iso_extended_string(second_clock::local_time()).substr(0, 19));
  response.put("average_merge_time", calculateAvgMegreTime());
  response.put("VmRSS", getVmRSSUsage());
  response.put("cpu_clock", calculateCpuUsage());
  response.put("merged_objects_per_second", calculateNumberOfMergedObjectsPerSecond());

  mInternalMetricMessageId++;

  return response;
}

void MergerDevice::sendControlResponse(const ptree& response, string senderId)
{
  stringstream responseStream;
  write_json(responseStream, response);

  ddsCustomCmd->send(responseStream.str(), senderId);
}

void MergerDevice::Run()
{
  while (CheckCurrentState(RUNNING)) {
    handleReceivedDataObject();
  }
}

void MergerDevice::handleReceivedDataObject()
{
  TObject* receivedObject = receiveDataObjectFromProducer();

  if (isObjectNotEmpty(receivedObject)) {
    TObject* mergedObject = mMerger->mergeObject(receivedObject);

    if (isObjectNotEmpty(mergedObject)) {
      updateMetrics();
      size_t messageSize = sendMergedObjectToViewer(mergedObject);
      delete mergedObject;
    }
  }
}

bool MergerDevice::isObjectNotEmpty(const TObject* object) const { return object == nullptr ? false : true; }
void MergerDevice::updateMetrics()
{
  mNumberOfMergedObjects++;

  if (mMergeTimes.size() >= LOGGED_MESSAGES) {
    mMergeTimes.pop_back();
  }

  mMergeTimes.push_front(mMerger->getMergeTime());
}

TMessage* MergerDevice::createTMessageForViewer(const TObject* objectToSend) const
{
  auto* viewerMessage = new TMessage(kMESS_OBJECT);
  viewerMessage->WriteObject(objectToSend);
  return viewerMessage;
}

TObject* MergerDevice::receiveDataObjectFromProducer()
{
  int respondeCode;
  TObject* receivedDataObject;
  unique_ptr<FairMQMessage> input(NewMessage());

  if ((respondeCode = fChannels.at("data-in").at(0).ReceiveAsync(input)) == -2) {
    if ((respondeCode = fChannels.at("data-in").at(0).ReceiveAsync(input)) == -2) {
      mLastReceiveBufferOverloadTime = clock();
      mReceiveBufferOverloaded = true;
      LOG(DEBUG) << "Buffer of data-in channel is full. Waiting for free buffer...";

      while ((respondeCode = fChannels.at("data-in").at(0).ReceiveAsync(input)) == -2) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }

      mReceiveBufferOverloaded = false;
      LOG(DEBUG) << "Buffer was released after " << double(clock() - mLastReceiveBufferOverloadTime) / CLOCKS_PER_SEC
                 << " seconds.";
    }
  }

  if (respondeCode >= 0) {
    TMessage* message = new TMessageWrapper(input->GetData(), input->GetSize());
    receivedDataObject = reinterpret_cast<TObject*>(message->ReadObject(message->GetClass()));
    delete message;
  } else {
    LOG(ERROR) << "Received empty message from producer, nothing to merge";
    receivedDataObject = nullptr;
  }

  return receivedDataObject;
}

size_t MergerDevice::sendMergedObjectToViewer(TObject* dataObject)
{
  int respondeCode;
  TMessage* viewerMessage = createTMessageForViewer(dataObject);
  unique_ptr<FairMQMessage> viewerRequest(fTransportFactory->CreateMessage(
    viewerMessage->Buffer(), viewerMessage->BufferSize(), deleteTMessage, viewerMessage));
  size_t messageSize = viewerRequest->GetSize();
  if ((respondeCode = fChannels.at("data-out").at(0).SendAsync(viewerRequest)) == -2) {
    if ((respondeCode = fChannels.at("data-out").at(0).SendAsync(viewerRequest)) == -2) {
      mLastSendBufferOverloadTime = clock();
      mSendBufferOverloaded = true;
      LOG(DEBUG) << "Buffer of data-out channel is full. Waiting for free buffer...";

      while ((respondeCode = fChannels.at("data-out").at(0).SendAsync(viewerRequest)) == -2) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }

      mSendBufferOverloaded = false;
      LOG(DEBUG) << "Buffer was released after " << double(clock() - mLastSendBufferOverloadTime) / CLOCKS_PER_SEC
                 << " seconds.";
    }
  }

  return messageSize;
}

double MergerDevice::calculateAvgMegreTime()
{
  double sum = 0.;

  for (double entry : mMergeTimes) {
    sum += entry;
  }

  return sum / mMergeTimes.size();
}

string MergerDevice::getVmRSSUsage()
{
  long memory = 0;

  if (procSelfStatus.is_open()) {
    string line;

    while (getline(procSelfStatus, line)) {
      if (line.substr(0, 5) == "VmRSS") {
        istringstream ss(line);
        string dump;
        ss >> dump >> memory;
        procSelfStatus.seekg(0, ios::beg);
        break;
      }
    }
  } else {
    LOG(ERROR) << "Error during opening file /proc/self/status";
  }

  return to_string(memory);
}

string MergerDevice::calculateCpuUsage()
{
  auto measureTime = high_resolution_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(measureTime - lastCpuMeasuredTime).count();
  auto currentCpuClockTime = clock();
  auto elapsedCpuClockTics = currentCpuClockTime - lastCpuMeasuredValue;

  lastCpuMeasuredTime = measureTime;
  lastCpuMeasuredValue = currentCpuClockTime;

  double normalizeFactor = 1.0 / (elapsedTime / 1000000.0);

  return to_string(static_cast<int>(round(elapsedCpuClockTics * normalizeFactor)));
}

double MergerDevice::calculateNumberOfMergedObjectsPerSecond()
{
  auto measureTime = high_resolution_clock::now();
  auto elapsedTime =
    std::chrono::duration_cast<std::chrono::microseconds>(measureTime - mLastNumberOfMergeObjectsTime).count();

  mLastNumberOfMergeObjectsTime = measureTime;
  double normalizeFactor = 1.0 / (elapsedTime / 1000000.0);
  double mergedObjectsPerSecond = mNumberOfMergedObjects * normalizeFactor;
  mNumberOfMergedObjects = 0;

  return mergedObjectsPerSecond;
}
}
}