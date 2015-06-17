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
    LOG(INFO) << "EPN" << endl << desc;
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

  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  epn.SetTransport(transportFactory);

  epn.SetProperty(EPNReceiver::Id, options.id);
  epn.SetProperty(EPNReceiver::NumIoThreads, options.ioThreads);

  epn.SetProperty(EPNReceiver::NumInputs, 1);
  epn.SetProperty(EPNReceiver::NumOutputs, options.numOutputs);
  epn.SetProperty(EPNReceiver::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);
  epn.SetProperty(EPNReceiver::BufferTimeoutInMs, options.bufferTimeoutInMs);
  epn.SetProperty(EPNReceiver::NumFLPs, options.numFLPs);
  epn.SetProperty(EPNReceiver::TestMode, options.testMode);

  epn.ChangeState(EPNReceiver::INIT);

  epn.SetProperty(EPNReceiver::InputSocketType, options.inputSocketType);
  epn.SetProperty(EPNReceiver::InputRcvBufSize, options.inputBufSize);
  epn.SetProperty(EPNReceiver::InputMethod, options.inputMethod);
  epn.SetProperty(EPNReceiver::InputAddress, initialInputAddress);
  epn.SetProperty(EPNReceiver::LogInputRate, options.inputRateLogging);

  for (int i = 0; i < (options.numFLPs); ++i) {
    epn.SetProperty(EPNReceiver::OutputSocketType, options.outputSocketType, i);
    epn.SetProperty(EPNReceiver::OutputSndBufSize, options.outputBufSize, i);
    epn.SetProperty(EPNReceiver::OutputMethod, options.outputMethod, i);
    epn.SetProperty(EPNReceiver::OutputAddress, "tcp://127.0.0.1:1234", i);
    epn.SetProperty(EPNReceiver::LogOutputRate, options.outputRateLogging, i);
  }

  epn.SetProperty(EPNReceiver::OutputSocketType, options.nextStepSocketType, options.numFLPs);
  epn.SetProperty(EPNReceiver::OutputSndBufSize, options.nextStepBufSize, options.numFLPs);
  epn.SetProperty(EPNReceiver::OutputMethod, options.nextStepMethod, options.numFLPs);
  epn.SetProperty(EPNReceiver::OutputAddress, initialOutputAddress, options.numFLPs);
  epn.SetProperty(EPNReceiver::LogOutputRate, options.nextStepRateLogging, options.numFLPs);

  if (options.testMode == 1) {
    // In test mode, initialize the feedback socket to the FLPSyncSampler
    epn.SetProperty(EPNReceiver::OutputSocketType, options.rttackSocketType, options.numFLPs + 1);
    epn.SetProperty(EPNReceiver::OutputSndBufSize, options.rttackBufSize, options.numFLPs + 1);
    epn.SetProperty(EPNReceiver::OutputMethod, options.rttackMethod, options.numFLPs + 1);
    epn.SetProperty(EPNReceiver::OutputAddress, "tcp://127.0.0.1:1234", options.numFLPs + 1);
    epn.SetProperty(EPNReceiver::LogOutputRate, options.rttackRateLogging, options.numFLPs + 1);
  }

  epn.ChangeState(EPNReceiver::SETOUTPUT);
  epn.ChangeState(EPNReceiver::SETINPUT);
  epn.ChangeState(EPNReceiver::BIND);

  dds::CKeyValue ddsKeyValue;

  ddsKeyValue.putValue("EPNReceiverInputAddress", epn.GetProperty(EPNReceiver::InputAddress, "", 0));
  if (options.testMode == 0) {
    // In regular mode, advertise the bound data output address via DDS.
    ddsKeyValue.putValue("EPNOutputAddress", epn.GetProperty(EPNReceiver::OutputAddress, "", options.numFLPs));
  }

  dds::CKeyValue::valuesMap_t values;
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

  dds::CKeyValue::valuesMap_t::const_iterator it_values = values.begin();
  for (int i = 0; i < options.numFLPs; ++i) {
    epn.SetProperty(EPNReceiver::OutputAddress, it_values->second, i);
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

    epn.SetProperty(EPNReceiver::OutputAddress, values.begin()->second, options.numFLPs + 1);
  }

  epn.ChangeState(EPNReceiver::CONNECT);
  epn.ChangeState(EPNReceiver::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(epn.fRunningMutex);
  while (!epn.fRunningFinished)
  {
    epn.fRunningCondition.wait(lock);
  }

  epn.ChangeState(EPNReceiver::STOP);
  epn.ChangeState(EPNReceiver::END);

  return 0;
}
