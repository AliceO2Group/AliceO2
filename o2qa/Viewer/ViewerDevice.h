#pragma once

#include <TApplication.h>
#include <memory>
#include <TList.h>
#include <TCanvas.h>
#include <FairMQDevice.h>
#include <string>
#include <unordered_set>

class ViewerDevice : public FairMQDevice
{
public:
  ViewerDevice(std::string viewerId, int numIoThreads, std::string drawingOptions = "");
  virtual ~ViewerDevice() = default;

  static void CustomCleanup(void *data, void* hint);
  void executeRunLoop();
  void establishChannel(std::string type, std::string method, std::string address, std::string channelName);
protected:
  ViewerDevice() = default;
  virtual void Run();

private:
  TCanvas* mObjectCanvas;
  std::unordered_set<std::string> mNamesOfObjectsToDraw;
  std::string mDrawingOptions;

  std::unique_ptr<FairMQMessage> receiveMessageFromMerger();
  void sendReplyToMerger(std::string* message);
  TObject* receiveDataObjectFromMerger();
  void updateObjectCanvas(TObject* receivedObject);
  void updateCanvas(TObject* receivedObject);
};
