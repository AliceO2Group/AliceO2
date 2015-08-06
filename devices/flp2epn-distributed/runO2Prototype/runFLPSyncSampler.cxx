/**
 * runFLPSyncSampler.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <map>
#include <string>

#include <boost/asio.hpp> // for DDS
#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQTransportFactoryZMQ.h"
#include "FairMQTools.h"

#include "FLPSyncSampler.h"

#include "KeyValue.h" // DDS

using namespace std;
using namespace AliceO2::Devices;

FLPSyncSampler sampler;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  sampler.ChangeState(FLPSyncSampler::END);

  cout << "Shutdown complete. Bye!" << endl;
  exit(1);
}

static void s_catch_signals (void)
{
  struct sigaction action;
  action.sa_handler = s_signal_handler;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
}

typedef struct DeviceOptions
{
  string id;
  int eventRate;
  int ioThreads;

  string inputSocketType;
  int inputBufSize;
  string inputMethod;
  // string inputAddress;
  int inputRateLogging;

  string outputSocketType;
  int outputBufSize;
  string outputMethod;
  // string outputAddress;
  int outputRateLogging;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw runtime_error("Internal error: options' container is empty.");

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("event-rate", bpo::value<int>()->default_value(0), "Event rate limit in maximum number of events per second")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("input-socket-type", bpo::value<string>()->required(), "Input socket type: sub/pull")
    ("input-buff-size", bpo::value<int>()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("input-method", bpo::value<string>()->required(), "Input method: bind/connect")
    // ("input-address", bpo::value<string>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("input-rate-logging", bpo::value<int>()->default_value(1), "Log input rate on socket, 1/0")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("output-rate-logging", bpo::value<int>()->default_value(1), "Log output rate on socket, 1/0")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "FLP Sync Sampler" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                  { _options->id                = vm["id"].as<string>(); }
  if (vm.count("event-rate"))          { _options->eventRate         = vm["event-rate"].as<int>(); }
  if (vm.count("io-threads"))          { _options->ioThreads         = vm["io-threads"].as<int>(); }

  if (vm.count("input-socket-type"))   { _options->inputSocketType   = vm["input-socket-type"].as<string>(); }
  if (vm.count("input-buff-size"))     { _options->inputBufSize      = vm["input-buff-size"].as<int>(); }
  if (vm.count("input-method"))        { _options->inputMethod       = vm["input-method"].as<string>(); }
  // if (vm.count("input-address"))      { _options->inputAddress     = vm["input-address"].as<string>(); }
  if (vm.count("input-rate-logging"))      { _options->inputRateLogging  = vm["input-rate-logging"].as<int>(); }

  if (vm.count("output-socket-type"))  { _options->outputSocketType  = vm["output-socket-type"].as<string>(); }
  if (vm.count("output-buff-size"))    { _options->outputBufSize     = vm["output-buff-size"].as<int>(); }
  if (vm.count("output-method"))       { _options->outputMethod      = vm["output-method"].as<string>(); }
  // if (vm.count("output-address"))     { _options->outputAddress    = vm["output-address"].as<string>(); }
  if (vm.count("output-rate-logging")) { _options->outputRateLogging = vm["output-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char** argv)
{
  s_catch_signals();

  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  } catch (exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  map<string,string> IPs;
  FairMQ::tools::getHostIPs(IPs);

  stringstream ss;

  if (IPs.count("ib0")) {
    ss << "tcp://" << IPs["ib0"] << ":5655";
  } else {
    ss << "tcp://" << IPs["eth0"] << ":5655";
  }

  string initialInputAddress  = ss.str();
  string initialOutputAddress = ss.str();

  LOG(INFO) << "FLP Sync Sampler";
  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  sampler.SetTransport(transportFactory);

  sampler.SetProperty(FLPSyncSampler::Id, options.id);
  sampler.SetProperty(FLPSyncSampler::NumIoThreads, options.ioThreads);
  sampler.SetProperty(FLPSyncSampler::EventRate, options.eventRate);

  FairMQChannel inputChannel(options.inputSocketType, options.inputMethod, initialInputAddress);
  inputChannel.UpdateSndBufSize(options.inputBufSize);
  inputChannel.UpdateRcvBufSize(options.inputBufSize);
  inputChannel.UpdateRateLogging(options.inputRateLogging);
  sampler.fChannels["data-in"].push_back(inputChannel);

  FairMQChannel outputChannel(options.outputSocketType, options.outputMethod, initialOutputAddress);
  outputChannel.UpdateSndBufSize(options.outputBufSize);
  outputChannel.UpdateRcvBufSize(options.outputBufSize);
  outputChannel.UpdateRateLogging(options.outputRateLogging);
  sampler.fChannels["data-out"].push_back(outputChannel);

  sampler.ChangeState("INIT_DEVICE");
  sampler.WaitForInitialValidation();

  // Advertise the bound addresses via DDS properties
  dds::key_value::CKeyValue ddsKeyValue;
  ddsKeyValue.putValue("FLPSyncSamplerOutputAddress", sampler.fChannels["data-out"].at(0).GetAddress());
  ddsKeyValue.putValue("FLPSyncSamplerInputAddress", sampler.fChannels["data-in"].at(0).GetAddress());

  sampler.WaitForEndOfState("INIT_DEVICE");

  sampler.ChangeState("INIT_TASK");
  sampler.WaitForEndOfState("INIT_TASK");

  sampler.ChangeState("RUN");
  sampler.WaitForEndOfState("RUN");

  sampler.ChangeState("STOP");

  sampler.ChangeState("RESET_TASK");
  sampler.WaitForEndOfState("RESET_TASK");

  sampler.ChangeState("RESET_DEVICE");
  sampler.WaitForEndOfState("RESET_DEVICE");

  sampler.ChangeState("END");

  return 0;
}
