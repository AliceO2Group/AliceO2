#include <chrono>
#include <limits>
#include <thread>

#include <FairMQLogger.h>
#include <TSystem.h>

#include "QCCommon/TMessageWrapper.h"
#include "QCViewer/ViewerDevice.h"

using namespace std;

namespace o2
{
namespace qc
{
ViewerDevice::ViewerDevice(std::string viewerId, int numIoThreads, string drawingOptions)
{
  this->SetTransport("zeromq");
  this->SetProperty(ViewerDevice::Id, viewerId);
  this->SetProperty(ViewerDevice::NumIoThreads, numIoThreads);
  mDrawingOptions = drawingOptions;
}

void ViewerDevice::Run()
{
  while (CheckCurrentState(RUNNING)) {
    TObject* receivedObject = receiveDataObjectFromMerger();

    if (receivedObject != nullptr) {
      LOG(INFO) << "Received QC objects from Merger device";
      // updateCanvas(receivedObject); // Visualization is disabled because there was no support of X11 protocol on the
      // previous testing environment
    }

    delete receivedObject;
  }
}

void ViewerDevice::updateCanvas(TObject* receivedObject)
{
  auto objectIterator = objectsToDraw.find(receivedObject->GetTitle());
  shared_ptr<TCanvas> activeCanvas;

  if (objectIterator == objectsToDraw.end()) {
    auto newObjectIterator = objectsToDraw.insert(
      { receivedObject->GetTitle(), make_shared<TCanvas>(receivedObject->GetTitle(), "QC object", 100, 10, 500, 400) });
    activeCanvas = newObjectIterator.first->second;
  } else {
    activeCanvas = objectIterator->second;
  }

  activeCanvas->cd();
  receivedObject->Draw(mDrawingOptions.c_str());
  activeCanvas->Update();
  gSystem->ProcessEvents();
}

TObject* ViewerDevice::receiveDataObjectFromMerger()
{
  TObject* receivedObject;
  unique_ptr<FairMQMessage> request(NewMessage());

  if (fChannels.at("data-in").at(0).ReceiveAsync(request) >= 0) {
    TMessageWrapper tm(request->GetData(), request->GetSize());
    receivedObject = static_cast<TObject*>(tm.ReadObject(tm.GetClass()));
  } else {
    receivedObject = nullptr;
  }

  return receivedObject;
}

void ViewerDevice::executeRunLoop()
{
  ChangeState("INIT_DEVICE");
  WaitForInitialValidation();
  WaitForEndOfState("INIT_DEVICE");

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

void ViewerDevice::establishChannel(string type, string method, string address, string channelName)
{
  FairMQChannel requestChannel(type, method, address);
  requestChannel.UpdateSndBufSize(numeric_limits<int>::max());
  requestChannel.UpdateRcvBufSize(numeric_limits<int>::max());
  requestChannel.UpdateRateLogging(1);
  fChannels[channelName].push_back(requestChannel);
}
}
}