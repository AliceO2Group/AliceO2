#ifndef O2_DATAINSPECTOR_H
#define O2_DATAINSPECTOR_H

#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataInspectorService.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include <cstring>

namespace o2::framework::DataInspector
{
/* Checks if a command line argument relates to the data inspector. */
inline bool isInspectorArgument(const char* argument)
{
  return std::strcmp(argument, "--inspector") == 0;
}

/* Checks if device is used by the data inspector */
inline bool isInspectorDevice(const DataProcessorSpec& spec)
{
  return spec.name == "DataInspector";
}

inline bool isInspectorDevice(const DeviceSpec& spec)
{
  return spec.name == "DataInspector";
}

inline bool isNonInternalDevice(const DeviceSpec& spec)
{
  return spec.name.find("internal") == std::string::npos;
}

/* Injects onProcess interceptor to check for messages from Proxy. */
void injectInterceptors(WorkflowSpec& workflow);

void sendToProxy(DataInspectorProxyService& diProxyService, const std::vector<DataRef>& refs, const std::string& deviceName);
}

#endif //O2_DATAINSPECTOR_H
