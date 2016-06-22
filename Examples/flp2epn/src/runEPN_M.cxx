/**
 * runEPN.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "flp2epn/O2EpnMerger.h"

#ifdef NANOMSG
#include "FairMQTransportFactoryNN.h"
#else

#include "FairMQTransportFactoryZMQ.h"

#endif

using namespace std;

struct DeviceOptions
{
    DeviceOptions() :
      id(), ioThreads(1),
      inputSocketType(), inputBufSize(1000), inputMethod(), inputAddress()
    { }

    string id;
    int ioThreads;
    string inputSocketType;
    int inputBufSize;
    string inputMethod;
    string inputAddress;
};

inline bool parse_cmd_line(int _argc, char *_argv[], DeviceOptions *_options)
{
  if (_options == NULL) {
    throw std::runtime_error("Internal error: options' container is empty.");
  }

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("input-socket-type", bpo::value<string>()->required(), "Input socket type: sub/pull")
    ("input-buff-size", bpo::value<int>()->required(),
     "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("input-method", bpo::value<string>()->required(), "Input method: bind/connect")
    ("input-address", bpo::value<string>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "EPN Merger" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id")) { _options->id = vm["id"].as<string>(); }
  if (vm.count("io-threads")) { _options->ioThreads = vm["io-threads"].as<int>(); }
  if (vm.count("input-socket-type")) { _options->inputSocketType = vm["input-socket-type"].as<string>(); }
  if (vm.count("input-buff-size")) { _options->inputBufSize = vm["input-buff-size"].as<int>(); }
  if (vm.count("input-method")) { _options->inputMethod = vm["input-method"].as<string>(); }
  if (vm.count("input-address")) { _options->inputAddress = vm["input-address"].as<string>(); }

  return true;
}

int main(int argc, char **argv)
{
  O2EpnMerger epn;
  epn.CatchSignals();

  DeviceOptions options;
  try {
    if (!parse_cmd_line(argc, argv, &options)) {
      return 0;
    }
  }
  catch (exception &e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory *transportFactory = new FairMQTransportFactoryZMQ();
#endif

  epn.SetTransport(transportFactory);

  epn.SetProperty(O2EpnMerger::Id, options.id);
  epn.SetProperty(O2EpnMerger::NumIoThreads, options.ioThreads);

  FairMQChannel inputChannel(options.inputSocketType, options.inputMethod, options.inputAddress);
  inputChannel.UpdateSndBufSize(options.inputBufSize);
  inputChannel.UpdateRcvBufSize(options.inputBufSize);

  epn.fChannels["data-in"].push_back(inputChannel);

  epn.ChangeState("INIT_DEVICE");
  epn.WaitForEndOfState("INIT_DEVICE");

  epn.ChangeState("INIT_TASK");
  epn.WaitForEndOfState("INIT_TASK");

  epn.ChangeState("RUN");
  epn.InteractiveStateLoop();

  return 0;
}

