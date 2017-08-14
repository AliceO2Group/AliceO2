// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSING_DEVICE_H
#define FRAMEWORK_DATAPROCESSING_DEVICE_H

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataRelayer.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/MessageContext.h"

namespace o2 {
namespace framework {

class DataProcessingDevice : public FairMQDevice {
public:
  DataProcessingDevice(const DeviceSpec &spec, ServiceRegistry &);
  void InitTask() final;
protected:
  bool HandleData(FairMQParts &parts, int index);
  void error(const char *msg);
private:
  DataProcessorSpec::InitCallback mInit;
  DataProcessorSpec::ProcessCallback mProcess;
  DataProcessorSpec::ErrorCallback mError;
  ConfigParamRegistry mConfigRegistry;
  ServiceRegistry mServiceRegistry;
  MessageContext mContext;
  DataAllocator mAllocator;
  DataRelayer mRelayer;

  std::vector<ChannelSpec> mChannels;
  std::map<std::string, InputSpec> mInputs;
  std::map<std::string, InputSpec> mForwards;
  int mErrorCount;
  int mProcessingCount;
};

}
}
#endif
