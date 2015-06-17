/**
 * runFLPSender.cxx
 *
 * @since 2013-04-23
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

#include "FLPSender.h"

#include "KeyValue.h" // DDS

using namespace std;
using namespace AliceO2::Devices;

FLPSender flp;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  flp.ChangeState(FLPSender::END);

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
  int eventSize;
  int ioThreads;
  int numInputs;
  int numOutputs;
  int heartbeatTimeoutInMs;
  int testMode;
  int sendOffset;
  vector<string> inputSocketType;
  vector<int> inputBufSize;
  vector<string> inputMethod;
  // vector<string> inputAddress;
  vector<int> inputRateLogging;
  string outputSocketType;
  int outputBufSize;
  string outputMethod;
  // vector<string> outputAddress;
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
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("num-inputs", bpo::value<int>()->required(), "Number of FLP input sockets")
    ("num-outputs", bpo::value<int>()->required(), "Number of FLP output sockets")
    ("heartbeat-timeout", bpo::value<int>()->default_value(20000), "Heartbeat timeout in milliseconds")
    ("test-mode", bpo::value<int>()->default_value(0),"Run in test mode")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("input-socket-type", bpo::value<vector<string>>()->required(), "Input socket type: sub/pull")
    ("input-buff-size", bpo::value<vector<int>>()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("input-method", bpo::value<vector<string>>()->required(), "Input method: bind/connect")
    // ("input-address", bpo::value<vector<string>>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("input-rate-logging", bpo::value<vector<int>>()->required(), "Log input rate on socket, 1/0")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("output-address", bpo::value<vector<string>>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("output-rate-logging", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "FLP" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                  { _options->id                   = vm["id"].as<string>(); }
  if (vm.count("event-size"))          { _options->eventSize            = vm["event-size"].as<int>(); }
  if (vm.count("io-threads"))          { _options->ioThreads            = vm["io-threads"].as<int>(); }
  if (vm.count("num-inputs"))          { _options->numInputs            = vm["num-inputs"].as<int>(); }
  if (vm.count("num-outputs"))         { _options->numOutputs           = vm["num-outputs"].as<int>(); }
  if (vm.count("heartbeat-timeout"))   { _options->heartbeatTimeoutInMs = vm["heartbeat-timeout"].as<int>(); }
  if (vm.count("test-mode"))           { _options->testMode             = vm["test-mode"].as<int>(); }
  if (vm.count("send-offset"))         { _options->sendOffset           = vm["send-offset"].as<int>(); }

  if (vm.count("input-socket-type"))   { _options->inputSocketType      = vm["input-socket-type"].as<vector<string>>(); }
  if (vm.count("input-buff-size"))     { _options->inputBufSize         = vm["input-buff-size"].as<vector<int>>(); }
  if (vm.count("input-method"))        { _options->inputMethod          = vm["input-method"].as<vector<string>>(); }
  // if (vm.count("input-address"))      { _options->inputAddress         = vm["input-address"].as<vector<string>>(); }
  if (vm.count("input-rate-logging"))      { _options->inputRateLogging     = vm["input-rate-logging"].as<vector<int>>(); }

  if (vm.count("output-socket-type"))  { _options->outputSocketType     = vm["output-socket-type"].as<string>(); }
  if (vm.count("output-buff-size"))    { _options->outputBufSize        = vm["output-buff-size"].as<int>(); }
  if (vm.count("output-method"))       { _options->outputMethod         = vm["output-method"].as<string>(); }
  // if (vm.count("output-address"))     { _options->outputAddress        = vm["output-address"].as<vector<string>>(); }
  if (vm.count("output-rate-logging")) { _options->outputRateLogging    = vm["output-rate-logging"].as<int>(); }

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

  string initialInputAddress = ss.str();

  // DDS
  // Waiting for properties
  dds::CKeyValue ddsKeyValue;
  dds::CKeyValue::valuesMap_t values;

  // In test mode, retreive the output address of FLPSyncSampler to connect to
  if (options.testMode == 1) {
    mutex keyMutex;
    condition_variable keyCondition;

    ddsKeyValue.subscribe([&keyCondition](const string& /*_key*/, const string& /*_value*/) { keyCondition.notify_all(); });
    ddsKeyValue.getValues("FLPSyncSamplerOutputAddress", &values);
    while (values.empty()) {
      unique_lock<mutex> lock(keyMutex);
      keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
      ddsKeyValue.getValues("FLPSyncSamplerOutputAddress", &values);
    }
  }

  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  flp.SetTransport(transportFactory);

  flp.SetProperty(FLPSender::Id, options.id);
  flp.SetProperty(FLPSender::NumIoThreads, options.ioThreads);
  flp.SetProperty(FLPSender::EventSize, options.eventSize);

  flp.SetProperty(FLPSender::NumInputs, options.numInputs);
  flp.SetProperty(FLPSender::NumOutputs, options.numOutputs);
  flp.SetProperty(FLPSender::HeartbeatTimeoutInMs, options.heartbeatTimeoutInMs);
  flp.SetProperty(FLPSender::TestMode, options.testMode);
  flp.SetProperty(FLPSender::SendOffset, options.sendOffset);

  flp.ChangeState(FLPSender::INIT);

  for (int i = 0; i < options.numInputs; ++i) {
    flp.SetProperty(FLPSender::InputSocketType, options.inputSocketType.at(i), i);
    flp.SetProperty(FLPSender::InputRcvBufSize, options.inputBufSize.at(i), i);
    flp.SetProperty(FLPSender::InputMethod, options.inputMethod.at(i), i);
    // flp.SetProperty(FLPSender::InputAddress, initialInputAddress, i);
    flp.SetProperty(FLPSender::LogInputRate, options.inputRateLogging.at(i), i);
  }

  flp.SetProperty(FLPSender::InputAddress, initialInputAddress, 0); // commands
  flp.SetProperty(FLPSender::InputAddress, initialInputAddress, 1); // heartbeats
  if (options.testMode == 1) {
    // In test mode, assign address that was received from the FLPSyncSampler via DDS.
    flp.SetProperty(FLPSender::InputAddress, values.begin()->second, 2); // FLPSyncSampler signal
  } else {
    // In regular mode, assign placeholder address, that will be set when binding.
    flp.SetProperty(FLPSender::InputAddress, initialInputAddress, 2); // data
  }

  for (int i = 0; i < options.numOutputs; ++i) {
    flp.SetProperty(FLPSender::OutputSocketType, options.outputSocketType, i);
    flp.SetProperty(FLPSender::OutputSndBufSize, options.outputBufSize, i);
    flp.SetProperty(FLPSender::OutputMethod, options.outputMethod, i);
    flp.SetProperty(FLPSender::OutputAddress, "tcp://127.0.0.1:123");
    flp.SetProperty(FLPSender::LogOutputRate, options.outputRateLogging, i);
  }

  flp.ChangeState(FLPSender::SETOUTPUT);
  flp.ChangeState(FLPSender::SETINPUT);
  flp.ChangeState(FLPSender::BIND);

  if (options.testMode == 0) {
    // In regular mode, advertise the bound data input address to the DDS.
    ddsKeyValue.putValue("FLPInputAddress", flp.GetProperty(FLPSender::InputAddress, "", 2));
  }

  ddsKeyValue.putValue("FLPSenderHeartbeatInputAddress", flp.GetProperty(FLPSender::InputAddress, "", 1));

  dds::CKeyValue::valuesMap_t values2;

  // Receive the EPNReceiver input addresses from DDS.
  {
  mutex keyMutex;
  condition_variable keyCondition;

  ddsKeyValue.subscribe([&keyCondition](const string& /*_key*/, const string& /*_value*/) {keyCondition.notify_all();});
  ddsKeyValue.getValues("EPNReceiverInputAddress", &values2);
  while (values2.size() != options.numOutputs) {
    unique_lock<mutex> lock(keyMutex);
    keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("EPNReceiverInputAddress", &values2);
  }
  }

  // Assign the received EPNReceiver input addresses to the device.
  dds::CKeyValue::valuesMap_t::const_iterator it_values2 = values2.begin();
  for (int i = 0; i < options.numOutputs; ++i) {
    flp.SetProperty(FLPSender::OutputAddress, it_values2->second, i);
    it_values2++;
  }

  flp.ChangeState(FLPSender::CONNECT);
  flp.ChangeState(FLPSender::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(flp.fRunningMutex);
  while (!flp.fRunningFinished) {
    flp.fRunningCondition.wait(lock);
  }

  flp.ChangeState(FLPSender::STOP);
  flp.ChangeState(FLPSender::END);

  return 0;
}
