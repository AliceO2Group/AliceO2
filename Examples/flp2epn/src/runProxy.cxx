/**
 * runProxy.cxx
 *
 * @since 2013-10-07
 * @author A. Rybalchenko
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "O2Proxy.h"

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
      inputSocketType(), inputBufSize(1000), inputMethod(), inputAddress(),
      outputSocketType(), outputBufSize(1000), outputMethod(), outputAddress()
    { }

    string id;
    int ioThreads;
    string inputSocketType;
    int inputBufSize;
    string inputMethod;
    string inputAddress;
    string outputSocketType;
    int outputBufSize;
    string outputMethod;
    string outputAddress;
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
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(),
     "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "Proxy" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id")) { _options->id = vm["id"].as<string>(); }
  if (vm.count("io-threads")) { _options->ioThreads = vm["io-threads"].as<int>(); }
  if (vm.count("input-socket-type")) { _options->inputSocketType = vm["input-socket-type"].as<string>(); }
  if (vm.count("input-buff-size")) { _options->inputBufSize = vm["input-buff-size"].as<int>(); }
  if (vm.count("input-method")) { _options->inputMethod = vm["input-method"].as<string>(); }
  if (vm.count("input-address")) { _options->inputAddress = vm["input-address"].as<string>(); }
  if (vm.count("output-socket-type")) { _options->outputSocketType = vm["output-socket-type"].as<string>(); }
  if (vm.count("output-buff-size")) { _options->outputBufSize = vm["output-buff-size"].as<int>(); }
  if (vm.count("output-method")) { _options->outputMethod = vm["output-method"].as<string>(); }
  if (vm.count("output-address")) { _options->outputAddress = vm["output-address"].as<string>(); }

  return true;
}

int main(int argc, char **argv)
{
  O2Proxy proxy;
  proxy.CatchSignals();

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

  proxy.SetTransport(transportFactory);

  proxy.SetProperty(O2Proxy::Id, options.id);
  proxy.SetProperty(O2Proxy::NumIoThreads, options.ioThreads);

  FairMQChannel inputChannel(options.inputSocketType, options.inputMethod, options.inputAddress);
  inputChannel.UpdateSndBufSize(options.inputBufSize);
  inputChannel.UpdateRcvBufSize(options.inputBufSize);
  inputChannel.UpdateRateLogging(1);

  proxy.fChannels["data-in"].push_back(inputChannel);

  FairMQChannel outputChannel(options.outputSocketType, options.outputMethod, options.outputAddress);
  outputChannel.UpdateSndBufSize(options.outputBufSize);
  outputChannel.UpdateRcvBufSize(options.outputBufSize);
  outputChannel.UpdateRateLogging(1);

  proxy.fChannels["data-out"].push_back(outputChannel);

  proxy.ChangeState("INIT_DEVICE");
  proxy.WaitForEndOfState("INIT_DEVICE");

  proxy.ChangeState("INIT_TASK");
  proxy.WaitForEndOfState("INIT_TASK");

  proxy.ChangeState("RUN");
  proxy.InteractiveStateLoop();

  return 0;
}

