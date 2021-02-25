#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include <cstring>

constexpr char INSPECTOR_ACTIVATION_PROPERTY[]{"inspector123"};

namespace o2
{
namespace framework
{

/* Checks if a command line argument relates to the data inspector. */
inline bool isInspectorArgument(const char* argument)
{
  return std::strstr(argument, "--inspector") != nullptr;
}

/* Checks if device is used by the data inspector */
inline bool isInspectorDevice(const DataProcessorSpec& spec)
{
  return spec.name == "DataInspector";
}

/* Checks if device appears in the list of devices inspected from the start. */
inline bool shouldBeInspected(const char* inspected, const DeviceSpec& spec)
{
  return std::strstr(inspected, spec.id.c_str()) != nullptr;
}

/* Checks if the data should be sent to the DataInspector. */
bool isDataInspectorActive(FairMQDevice& device);

/* Copies `parts` and sends it to the `device`. The copy is necessary because of
   memory management. */
void sendCopyToDataInspector(FairMQDevice& device, FairMQParts& parts, unsigned index);

/* Creates an O2 Device for the DataInspector and adds it to `workflow`. */
void addDataInspector(WorkflowSpec& workflow);

} // namespace framework
} // namespace o2
