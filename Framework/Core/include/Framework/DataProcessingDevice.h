// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSING_DEVICE_H
#define FRAMEWORK_DATAPROCESSING_DEVICE_H

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataRelayer.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/InputRoute.h"
#include "Framework/ForwardRoute.h"

#include <memory>

namespace o2 {
namespace framework {

class DataProcessingDevice : public FairMQDevice {
public:
  DataProcessingDevice(const DeviceSpec &spec, ServiceRegistry &);
  void Init() final;
protected:
  bool HandleData(FairMQParts &parts, int index);
  void error(const char *msg);
private:
  AlgorithmSpec::InitCallback mInit;
  AlgorithmSpec::ProcessCallback mStatefulProcess;
  AlgorithmSpec::ProcessCallback mStatelessProcess;
  AlgorithmSpec::ErrorCallback mError;
  std::unique_ptr<ConfigParamRegistry> mConfigRegistry;
  ServiceRegistry mServiceRegistry;
  MessageContext mContext;
  RootObjectContext mRootContext;
  DataAllocator mAllocator;
  DataRelayer mRelayer;

  std::vector<InputChannelSpec> mInputChannels;
  std::vector<OutputChannelSpec> mOutputChannels;

  std::vector<InputRoute> mInputs;
  std::vector<ForwardRoute> mForwards;
  int mErrorCount;
  int mProcessingCount;
};

}
}
#endif
