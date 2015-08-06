/**
 * runEPNReceiver.cxx
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

#include "EPNReceiver.h"

#include "KeyValue.h" // DDS

using namespace std;
using namespace AliceO2::Devices;

EPNReceiver epn;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  epn.ChangeState(EPNReceiver::END);

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
  int ioThreads;
  int numOutputs;
  int heartbeatIntervalInMs;
  int bufferTimeoutInMs;
  int numFLPs;
  int testMode;

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

  string nextStepSocketType;
  int nextStepBufSize;
  string nextStepMethod;
  // string nextStepAddress;
  int nextStepRateLogging;

  string rttackSocketType;
  int rttackBufSize;
  string rttackMethod;
  // string rttackAddress;
  int rttackRateLogging;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw runtime_error("Internal error: options' container is empty.");

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("num-outputs", bpo::value<int>()->required(), "Number of EPN output sockets")
    ("heartbeat-interval", bpo::value<int>()->default_value(5000), "Heartbeat interval in milliseconds")
    ("buffer-timeout", bpo::value<int>()->default_value(5000), "Buffer timeout in milliseconds")
    ("num-flps", bpo::value<int>()->required(), "Number of FLPs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("input-socket-type", bpo::value<string>()->required(), "Input socket type: sub/pull")
    ("input-buff-size", bpo::value<int>()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("input-method", bpo::value<string>()->required(), "Input method: bind/connect")
    // ("input-address", bpo::value<string>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("input-rate-logging", bpo::value<int>()->required(), "Log input rate on socket, 1/0")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("output-rate-logging", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("nextstep-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("nextstep-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("nextstep-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("nextstep-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("nextstep-rate-logging", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("rttack-socket-type", bpo::value<string>(), "Output socket type: pub/push")
    ("rttack-buff-size", bpo::value<int>(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("rttack-method", bpo::value<string>(), "Output method: bind/connect")
    // ("rttack-address", bpo::value<string>(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("rttack-rate-logging", bpo::value<int>(), "Log output rate on socket, 1/0")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "EPN Receiver" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                    { _options->id                    = vm["id"].as<string>(); }
  if (vm.count("io-threads"))            { _options->ioThreads             = vm["io-threads"].as<int>(); }
  if (vm.count("num-outputs"))           { _options->numOutputs            = vm["num-outputs"].as<int>(); }
  if (vm.count("heartbeat-interval"))    { _options->heartbeatIntervalInMs = vm["heartbeat-interval"].as<int>(); }
  if (vm.count("buffer-timeout"))        { _options->bufferTimeoutInMs     = vm["buffer-timeout"].as<int>(); }
  if (vm.count("num-flps"))              { _options->numFLPs               = vm["num-flps"].as<int>(); }
  if (vm.count("test-mode"))             { _options->testMode              = vm["test-mode"].as<int>(); }

  if (vm.count("input-socket-type"))     { _options->inputSocketType       = vm["input-socket-type"].as<string>(); }
  if (vm.count("input-buff-size"))       { _options->inputBufSize          = vm["input-buff-size"].as<int>(); }
  if (vm.count("input-method"))          { _options->inputMethod           = vm["input-method"].as<string>(); }
  // if (vm.count("input-address"))         { _options->inputAddress          = vm["input-address"].as<string>(); }
  if (vm.count("input-rate-logging"))    { _options->inputRateLogging      = vm["input-rate-logging"].as<int>(); }

  if (vm.count("output-socket-type"))    { _options->outputSocketType      = vm["output-socket-type"].as<string>(); }
  if (vm.count("output-buff-size"))      { _options->outputBufSize         = vm["output-buff-size"].as<int>(); }
  if (vm.count("output-method"))         { _options->outputMethod          = vm["output-method"].as<string>(); }
  // if (vm.count("output-address"))        { _options->outputAddress         = vm["output-address"].as<string>(); }
  if (vm.count("output-rate-logging"))   { _options->outputRateLogging     = vm["output-rate-logging"].as<int>(); }

  if (vm.count("nextstep-socket-type"))  { _options->nextStepSocketType    = vm["nextstep-socket-type"].as<string>(); }
  if (vm.count("nextstep-buff-size"))    { _options->nextStepBufSize       = vm["nextstep-buff-size"].as<int>(); }
  if (vm.count("nextstep-method"))       { _options->nextStepMethod        = vm["nextstep-method"].as<string>(); }
  // if (vm.count("nextstep-address"))      { _options->nextStepAddress       = vm["nextstep-address"].as<string>(); }
  if (vm.count("nextstep-rate-logging")) { _options->nextStepRateLogging   = vm["nextstep-rate-logging"].as<int>(); }

  if (vm.count("rttack-socket-type"))    { _options->rttackSocketType      = vm["rttack-socket-type"].as<string>(); }
  if (vm.count("rttack-buff-size"))      { _options->rttackBufSize         = vm["rttack-buff-size"].as<int>(); }
  if (vm.count("rttack-method"))         { _options->rttackMethod          = vm["rttack-method"].as<string>(); }
  // if (vm.count("rttack-address"))        { _options->rttackAddress         = vm["rttack-address"].as<string>(); }
  if (vm.count("rttack-rate-logging"))   { _options->rttackRateLogging     = vm["rttack-rate-logging"].as<int>(); }

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

  LOG(INFO) << "EPN Receiver";
  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  epn.SetTransport(transportFactory);

  // configure device
  epn.SetProperty(EPNReceiver::Id, options.id);
  epn.SetProperty(EPNReceiver::NumIoThreads, options.ioThreads);
  epn.SetProperty(EPNReceiver::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);
  epn.SetProperty(EPNReceiver::BufferTimeoutInMs, options.bufferTimeoutInMs);
  epn.SetProperty(EPNReceiver::NumFLPs, options.numFLPs);
  epn.SetProperty(EPNReceiver::TestMode, options.testMode);

  // configure inputs
  FairMQChannel inputChannel(options.inputSocketType, options.inputMethod, initialInputAddress);
  inputChannel.UpdateSndBufSize(options.inputBufSize);
  inputChannel.UpdateRcvBufSize(options.inputBufSize);
  inputChannel.UpdateRateLogging(options.inputRateLogging);
  epn.fChannels["data-in"].push_back(inputChannel);

  // configure outputs
  for (int i = 0; i < options.numFLPs; ++i) {
    FairMQChannel outputChannel(options.outputSocketType, options.outputMethod, "");
    outputChannel.UpdateSndBufSize(options.outputBufSize);
    outputChannel.UpdateRcvBufSize(options.outputBufSize);
    outputChannel.UpdateRateLogging(options.outputRateLogging);
    epn.fChannels["data-out"].push_back(outputChannel);
  }

  FairMQChannel nextStepChannel(options.nextStepSocketType, options.nextStepMethod, initialOutputAddress);
  nextStepChannel.UpdateSndBufSize(options.nextStepBufSize);
  nextStepChannel.UpdateRcvBufSize(options.nextStepBufSize);
  nextStepChannel.UpdateRateLogging(options.nextStepRateLogging);
  epn.fChannels["data-out"].push_back(nextStepChannel);

  if (options.testMode == 1) {
    // In test mode, initialize the feedback socket to the FLPSyncSampler
    FairMQChannel rttackChannel(options.rttackSocketType, options.rttackMethod, "");
    rttackChannel.UpdateSndBufSize(options.rttackBufSize);
    rttackChannel.UpdateRcvBufSize(options.rttackBufSize);
    rttackChannel.UpdateRateLogging(options.rttackRateLogging);
    epn.fChannels["data-out"].push_back(rttackChannel);
  }

  epn.ChangeState("INIT_DEVICE");
  epn.WaitForInitialValidation();

  dds::key_value::CKeyValue ddsKeyValue;

  ddsKeyValue.putValue("EPNReceiverInputAddress", epn.fChannels["data-in"].at(0).GetAddress());
  if (options.testMode == 0) {
    // In regular mode, advertise the bound data output address via DDS.
    ddsKeyValue.putValue("EPNReceiverOutputAddress", epn.fChannels["data-out"].at(options.numFLPs).GetAddress());
  }

  dds::key_value::CKeyValue::valuesMap_t values;
  {
  mutex keyMutex;
  condition_variable keyCondition;

  ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

  ddsKeyValue.getValues("FLPSenderHeartbeatInputAddress", &values);
  while (values.size() != options.numFLPs) {
    unique_lock<mutex> lock(keyMutex);
    keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("FLPSenderHeartbeatInputAddress", &values);
  }
  }

  dds::key_value::CKeyValue::valuesMap_t::const_iterator it_values = values.begin();
  for (int i = 0; i < options.numFLPs; ++i) {
    epn.fChannels["data-out"].at(i).UpdateAddress(it_values->second);
    it_values++;
  }

  if (options.testMode == 1) {
    // In test mode, get the value of the FLPSyncSampler input address for the feedback socket.
    values.clear();
    {
    mutex keyMutex;
    condition_variable keyCondition;
    
    ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

    ddsKeyValue.getValues("FLPSyncSamplerInputAddress", &values);
    while (values.empty()) {
      unique_lock<mutex> lock(keyMutex);
      keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
      ddsKeyValue.getValues("FLPSyncSamplerInputAddress", &values);
    }
    }

    epn.fChannels["data-out"].at(options.numFLPs + 1).UpdateAddress(values.begin()->second);
  }

  epn.WaitForEndOfState("INIT_DEVICE");

  epn.ChangeState("INIT_TASK");
  epn.WaitForEndOfState("INIT_TASK");

  epn.ChangeState("RUN");
  epn.WaitForEndOfState("RUN");

  epn.ChangeState("STOP");

  epn.ChangeState("RESET_TASK");
  epn.WaitForEndOfState("RESET_TASK");

  epn.ChangeState("RESET_DEVICE");
  epn.WaitForEndOfState("RESET_DEVICE");

  epn.ChangeState("END");

  return 0;
}
