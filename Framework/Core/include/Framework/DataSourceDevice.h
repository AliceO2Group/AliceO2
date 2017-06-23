// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATASOURCE_DEVICE_H
#define FRAMEWORK_DATASOURCE_DEVICE_H

#include <fairmq/FairMQDevice.h>

#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"

namespace o2 {
namespace framework {
/// Implements the boilerplate for a generic
/// framework device which only produces data
class DataSourceDevice : public FairMQDevice {
public:
  DataSourceDevice(const DeviceSpec &spec);
  void InitTask() override final;
protected:
  bool ConditionalRun() override final;
private:
  DataProcessorSpec::InitCallback mInit;
  DataProcessorSpec::ProcessCallback mProcess;
  DataProcessorSpec::ErrorCallback mError;

  ConfigParamRegistry mConfigRegistry;
  ServiceRegistry mServiceRegistry;
  MessageContext mContext;
  DataAllocator mAllocator;
};

}
}
#endif
