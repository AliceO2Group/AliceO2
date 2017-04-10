#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <FairMQLogger.h>
#include <TApplication.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <dds_intercom.h>

#include "QCCommon/TMessageWrapper.h"
#include "QCMerger/Merger.h"
#include "QCMerger/MergerDevice.h"

using namespace dds;
using namespace dds::intercom_api;
using namespace std;
using namespace o2::qc;

namespace bpo = boost::program_options;

namespace
{
const int NUMBER_OF_IO_THREADS = 1;
const int NUMBER_OF_REQUIRED_PROGRAM_PARAMETERS = 6;
ostringstream localAddress;

string exec(const char* cmd)
{
  char buffer[128];
  string result;
  shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

  if (!pipe) {
    throw runtime_error("Runtime error for popen() function.");
  }

  while (!feof(pipe.get())) {
    if (fgets(buffer, 128, pipe.get()) != nullptr) {
      result += buffer;
    }
  }
  return result;
}
}

int main(int argc, char** argv)
{
  CIntercomService service;
  CKeyValue keyValue(service);

  const char* inputAddress = argv[1];

  service.subscribeOnError([](EErrorCode _errorCode, const string& _msg) {
    LOG(ERROR) << "DDS key-value error code: " << _errorCode << ", message: " << _msg;
  });

  service.start();

  localAddress << "tcp://" << exec("hostname -i") << ":" << argv[4];
  string stringLocalAddress = localAddress.str();
  boost::erase_all(stringLocalAddress, "\n");

  keyValue.putValue(inputAddress, stringLocalAddress.c_str());

  if (argc != NUMBER_OF_REQUIRED_PROGRAM_PARAMETERS + 1) {
    LOG(ERROR) << "Not sufficient arguments value: " << NUMBER_OF_REQUIRED_PROGRAM_PARAMETERS;
    exit(-1);
  }

  const char* MERGER_DEVICE_ID = argv[2];
  const int NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA = atoi(argv[3]);
  const int INPUT_BUFFER_SIZE = atoi(argv[5]);
  const char* OUTPUT_HOST = argv[6];

  bpo::options_description options("task-custom-cmd options");
  options.add_options()("help,h", "Produce help message");

  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(options).run(), vm);
  bpo::notify(vm);

  MergerDevice mergerDevice(unique_ptr<Merger>(new Merger(NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA)), MERGER_DEVICE_ID,
                            NUMBER_OF_IO_THREADS);
  mergerDevice.CatchSignals();

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Merger id: " << mergerDevice.GetProperty(MergerDevice::Id, "default_id");

  mergerDevice.establishChannel("pull", "bind", stringLocalAddress.c_str(), "data-in", INPUT_BUFFER_SIZE,
                                INPUT_BUFFER_SIZE);
  mergerDevice.establishChannel("push", "connect", OUTPUT_HOST, "data-out", numeric_limits<int>::max(),
                                numeric_limits<int>::max());

  mergerDevice.executeRunLoop();
}
