// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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