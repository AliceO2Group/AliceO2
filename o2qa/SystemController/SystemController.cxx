#include <FairMQTransportFactoryZMQ.h>
#include <chrono>
#include <thread>

#include "SystemController.h"

using namespace std;

SystemController::SystemController(std::string controllerId, std::string logFileName, int numIoThreads)
{
  this->SetTransport(new FairMQTransportFactoryZMQ);
  this->SetProperty(SystemController::Id, controllerId);
  this->SetProperty(SystemController::NumIoThreads, numIoThreads);
  mLogFile.open(logFileName);
}

SystemController::~SystemController()
{
  mLogFile.close();
}

void SystemController::establishChannel(std::string type, std::string method, std::string address, std::string channelName)
{
  FairMQChannel requestChannel(type, method, address);
  requestChannel.UpdateSndBufSize(5000);
  requestChannel.UpdateRcvBufSize(1000);
  requestChannel.UpdateRateLogging(1);
  fChannels[channelName].push_back(requestChannel);
}

void SystemController::executeRunLoop()
{
  ChangeState("INIT_DEVICE");
  WaitForEndOfState("INIT_DEVICE");

  ChangeState("INIT_TASK");
  WaitForEndOfState("INIT_TASK");

  ChangeState("RUN");
  InteractiveStateLoop();
}

void SystemController::CustomCleanup(void *data, void *hint)
{
  delete (string*)hint;
}

void SystemController::Run()
{
  FairMQPoller* poller = fTransportFactory->CreatePoller(fChannels["data"]);
  poller->Poll(0);

  while (GetCurrentState() == RUNNING) {
     mLogFile << getCurrentTime() << "\tSending status request at 5001 tcp port "  << endl;

    chrono::time_point<std::chrono::system_clock> startTime;
    startTime = chrono::system_clock::now();

    FairMQMessage* request = sendMessageToNodes();

    while (chrono::system_clock::now() < (startTime + chrono::seconds(5))) {
      if (poller->CheckInput(0)) {
        getStatusFromSystemNodes();
      }
    }
  }
}

FairMQMessage* SystemController::sendMessageToNodes()
{
  string* text = new string("report_status");

  FairMQMessage* request = fTransportFactory->CreateMessage(const_cast<char*>(text->c_str()),
                                                            text->length(),
                                                            CustomCleanup,
                                                            text);
  LOG(INFO) << "Sending status request";
  fChannels["data"].at(0).Send(request, "no-block");
}

void SystemController::getStatusFromSystemNodes()
{
  FairMQMessage* reply = fTransportFactory->CreateMessage();
  fChannels["data"].at(0).Receive(reply);

  if (reply->GetSize() != 0) {
    auto replyContent = string(static_cast<char*>(reply->GetData()), reply->GetSize());
    LOG(INFO) << "Received reply from node: " << replyContent.c_str();

    if (reply->GetSize() != 0) {
      mLogFile << getCurrentTime() << "\tReceived:\t" << replyContent.c_str() << endl;
    }
  }
  delete reply;
}

string SystemController::getCurrentTime()
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];

  time (&rawtime);
  timeinfo = localtime (&rawtime);
  strftime (buffer, 80, "%H:%M:%S %F", timeinfo);

  return string(buffer);
}
