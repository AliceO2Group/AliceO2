#include "runFairMQDevice.h"

#include "CCDB/ConditionsMQClient.h"

using namespace o2::CDB;
using namespace std;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()("parameter-name", bpo::value<string>()->default_value("DET/Calib/Histo"), "Parameter Name")(
    "operation-type", bpo::value<string>()->default_value("GET"), "Operation Type")(
    "data-source", bpo::value<string>()->default_value("OCDB"), "Data Source")(
    "object-path", bpo::value<string>()->default_value("OCDB"), "Object Path");
}

FairMQDevice* getDevice(const FairMQProgOptions& config) { return new ConditionsMQClient(); }
