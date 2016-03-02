#include <TSystem.h>
#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <chrono>
#include <thread>

#include "ViewerDevice.h"
#include "HistogramTMessage.h"

using namespace std;

ViewerDevice::ViewerDevice(std::string viewerId, int numIoThreads, string drawingOptions)
{
  this->SetTransport(new FairMQTransportFactoryZMQ);
  this->SetProperty(ViewerDevice::Id, viewerId);
  this->SetProperty(ViewerDevice::NumIoThreads, numIoThreads);
  mDrawingOptions = drawingOptions;
}

void ViewerDevice::CustomCleanup(void *data, void *hint)
{
  delete (string*)hint;
}

void ViewerDevice::Run()
{
  mObjectCanvas = new TCanvas("mObjectCanvas", "Object canvas", 100, 10, 1200, 800);

  while (GetCurrentState() == RUNNING) {
    this_thread::sleep_for(chrono::milliseconds(1000));

    TObject* receivedObject = receiveDataObjectFromMerger();

    if (receivedObject != nullptr) {
      updateCanvas(receivedObject);
      updateObjectCanvas(receivedObject);
      sendReplyToMerger(new string("VIEWER_OK"));
    }
  }
  delete mObjectCanvas;
}

void ViewerDevice::updateCanvas(TObject* receivedObject)
{
  if (mNamesOfObjectsToDraw.find(receivedObject->GetTitle()) == mNamesOfObjectsToDraw.end()) {

    mNamesOfObjectsToDraw.insert(receivedObject->GetTitle());

    mObjectCanvas->Clear();
    mObjectCanvas->Divide(mNamesOfObjectsToDraw.size(), 1);
    mObjectCanvas->cd(mNamesOfObjectsToDraw.size());
    mObjectCanvas->Update();
  }
  else {
    unsigned padId = mNamesOfObjectsToDraw.size();
    for (auto const & name : mNamesOfObjectsToDraw) {
      if (receivedObject->GetTitle() == name) {
        break;
      }
      padId--;
    }
    mObjectCanvas->cd(padId);
  }
}

void ViewerDevice::sendReplyToMerger(string* message)
{
  LOG(INFO) << "Sending reply to merger";
  FairMQMessage* reply = fTransportFactory->CreateMessage(const_cast<char*>(message->c_str()),
                                                          message->length(),
                                                          CustomCleanup,
                                                          message);
  fChannels["data"].at(0).Send(reply);
}

TObject* ViewerDevice::receiveDataObjectFromMerger()
{
  TObject* receivedObject;
  unique_ptr<FairMQMessage> request(move(receiveMessageFromMerger()));

  if (request->GetSize() != 0) {
    HistogramTMessage tm(request->GetData(), request->GetSize());
    receivedObject = static_cast<TObject*>(tm.ReadObject(tm.GetClass()));
  }
  else {
    LOG(ERROR) << "Received empty request from merger";
    receivedObject = nullptr;
  }

  return receivedObject;
}

unique_ptr<FairMQMessage> ViewerDevice::receiveMessageFromMerger()
{
  unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
  fChannels["data"].at(0).Receive(&(*request));
  LOG(INFO) << "Received data object from merger";
  return request;
}

void ViewerDevice::updateObjectCanvas(TObject* receivedObject)
{
  mObjectCanvas->Update();
  receivedObject->Draw(mDrawingOptions.c_str());
  mObjectCanvas->Modified();
  mObjectCanvas->Update();
  gSystem->ProcessEvents();
}

void ViewerDevice::executeRunLoop()
{
  ChangeState("INIT_DEVICE");
  WaitForEndOfState("INIT_DEVICE");

  ChangeState("INIT_TASK");
  WaitForEndOfState("INIT_TASK");

  ChangeState("RUN");
  InteractiveStateLoop();
}

void ViewerDevice::establishChannel(string type, string method, string address, string channelName)
{
  FairMQChannel requestChannel(type, method, address);
  requestChannel.UpdateSndBufSize(1000);
  requestChannel.UpdateRcvBufSize(1000);
  requestChannel.UpdateRateLogging(1);
  fChannels[channelName].push_back(requestChannel);
}
