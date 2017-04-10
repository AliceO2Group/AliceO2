#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <FairMQDevice.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TList.h>

namespace o2
{
namespace qc
{
class ViewerDevice : public FairMQDevice
{
 public:
  ViewerDevice(std::string viewerId, int numIoThreads, std::string drawingOptions = "");
  ~ViewerDevice() override = default;

  void executeRunLoop();
  void establishChannel(std::string type, std::string method, std::string address, std::string channelName);

 protected:
  ViewerDevice() = default;
  void Run() override;

 private:
  std::unordered_map<std::string, std::shared_ptr<TCanvas>> objectsToDraw;
  std::string mDrawingOptions;

  std::unique_ptr<FairMQMessage> receiveMessageFromMerger();
  TObject* receiveDataObjectFromMerger();
  void updateCanvas(TObject* receivedObject);

  std::string getVmRSSUsage();
};
}
}