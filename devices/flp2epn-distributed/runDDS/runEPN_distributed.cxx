/**
 * runEPN_distributed.cxx
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

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQTransportFactoryZMQ.h"
#include "FairMQTools.h"

#include "EPNex.h"

//DDS
#include "KeyValue.h"
#include <boost/asio.hpp>

using namespace std;

using namespace AliceO2::Devices;

EPNex epn;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  epn.ChangeState(EPNex::STOP);
  epn.ChangeState(EPNex::END);

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
  int logInputRate;
  string outputSocketType;
  int outputBufSize;
  string outputMethod;
  // string outputAddress;
  int logOutputRate;
  string nextStepSocketType;
  int nextStepBufSize;
  string nextStepMethod;
  // string nextStepAddress;
  int logNextStepRate;
  string rttackSocketType;
  int rttackBufSize;
  string rttackMethod;
  // string rttackAddress;
  int logRttackRate;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw std::runtime_error("Internal error: options' container is empty.");

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
    ("log-input-rate", bpo::value<int>()->required(), "Log input rate on socket, 1/0")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("log-output-rate", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("nextstep-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("nextstep-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("nextstep-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("nextstep-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("log-nextstep-rate", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("rttack-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("rttack-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("rttack-method", bpo::value<string>()->required(), "Output method: bind/connect")
    // ("rttack-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("log-rttack-rate", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "EPN" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))
    _options->id = vm["id"].as<string>();

  if (vm.count("io-threads"))
    _options->ioThreads = vm["io-threads"].as<int>();

  if (vm.count("num-outputs"))
    _options->numOutputs = vm["num-outputs"].as<int>();

  if (vm.count("heartbeat-interval"))
    _options->heartbeatIntervalInMs = vm["heartbeat-interval"].as<int>();

  if (vm.count("buffer-timeout"))
    _options->bufferTimeoutInMs = vm["buffer-timeout"].as<int>();

  if (vm.count("num-flps"))
    _options->numFLPs = vm["num-flps"].as<int>();

  if (vm.count("test-mode"))
  _options->testMode = vm["test-mode"].as<int>();

  if (vm.count("input-socket-type"))
    _options->inputSocketType = vm["input-socket-type"].as<string>();

  if (vm.count("input-buff-size"))
    _options->inputBufSize = vm["input-buff-size"].as<int>();

  if (vm.count("input-method"))
    _options->inputMethod = vm["input-method"].as<string>();

//  if (vm.count("input-address"))
//    _options->inputAddress = vm["input-address"].as<string>();

  if (vm.count("log-input-rate"))
    _options->logInputRate = vm["log-input-rate"].as<int>();

  if (vm.count("output-socket-type"))
    _options->outputSocketType = vm["output-socket-type"].as<string>();

  if (vm.count("output-buff-size"))
    _options->outputBufSize = vm["output-buff-size"].as<int>();

  if (vm.count("output-method"))
    _options->outputMethod = vm["output-method"].as<string>();

  // if (vm.count("output-address"))
  //   _options->outputAddress = vm["output-address"].as<string>();

  if (vm.count("log-output-rate"))
    _options->logOutputRate = vm["log-output-rate"].as<int>();


  if (vm.count("nextstep-socket-type"))
    _options->nextStepSocketType = vm["nextstep-socket-type"].as<string>();

  if (vm.count("nextstep-buff-size"))
    _options->nextStepBufSize = vm["nextstep-buff-size"].as<int>();

  if (vm.count("nextstep-method"))
    _options->nextStepMethod = vm["nextstep-method"].as<string>();

  // if (vm.count("nextstep-address"))
  //   _options->nextStepAddress = vm["nextstep-address"].as<string>();

  if (vm.count("log-nextstep-rate"))
    _options->logNextStepRate = vm["log-nextstep-rate"].as<int>();


  if (vm.count("rttack-socket-type"))
    _options->rttackSocketType = vm["rttack-socket-type"].as<string>();

  if (vm.count("rttack-buff-size"))
    _options->rttackBufSize = vm["rttack-buff-size"].as<int>();

  if (vm.count("rttack-method"))
    _options->rttackMethod = vm["rttack-method"].as<string>();

  // if (vm.count("rttack-address"))
  //   _options->rttackAddress = vm["rttack-address"].as<string>();

  if (vm.count("log-rttack-rate"))
    _options->logRttackRate = vm["log-rttack-rate"].as<int>();

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

  std::stringstream ss;
  if(IPs.count("ib0")) {
    ss << "tcp://" << IPs["ib0"] << ":5655";
  } else {
    ss << "tcp://" << IPs["eth0"] << ":5655";
  }
  std::string initialInputAddress = ss.str();

  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  epn.SetTransport(transportFactory);

  epn.SetProperty(EPNex::Id, options.id);
  epn.SetProperty(EPNex::NumIoThreads, options.ioThreads);

  epn.SetProperty(EPNex::NumInputs, 1);
  epn.SetProperty(EPNex::NumOutputs, options.numOutputs);
  epn.SetProperty(EPNex::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);
  epn.SetProperty(EPNex::BufferTimeoutInMs, options.bufferTimeoutInMs);
  epn.SetProperty(EPNex::NumFLPs, options.numFLPs);
  epn.SetProperty(EPNex::TestMode, options.testMode);

  epn.ChangeState(EPNex::INIT);

  epn.SetProperty(EPNex::InputSocketType, options.inputSocketType);
  epn.SetProperty(EPNex::InputRcvBufSize, options.inputBufSize);
  epn.SetProperty(EPNex::InputMethod, options.inputMethod);
  epn.SetProperty(EPNex::InputAddress, initialInputAddress);
  epn.SetProperty(EPNex::LogInputRate, options.logInputRate);

  //dds::CKeyValue::valuesMap_t::const_iterator it_values = values.begin();
  for (int i = 0; i < (options.numFLPs); ++i) {
    epn.SetProperty(EPNex::OutputSocketType, options.outputSocketType, i);
    epn.SetProperty(EPNex::OutputSndBufSize, options.outputBufSize, i);
    epn.SetProperty(EPNex::OutputMethod, options.outputMethod, i);
    epn.SetProperty(EPNex::OutputAddress, "tcp://127.0.0.1:1234", i);
    epn.SetProperty(EPNex::LogOutputRate, options.logOutputRate, i);
    // it_values++;
  }

  epn.SetProperty(EPNex::OutputSocketType, options.nextStepSocketType, options.numFLPs);
  epn.SetProperty(EPNex::OutputSndBufSize, options.nextStepBufSize, options.numFLPs);
  epn.SetProperty(EPNex::OutputMethod, options.nextStepMethod, options.numFLPs);
  epn.SetProperty(EPNex::OutputAddress, "tcp://127.0.0.1:1234", options.numFLPs);
  epn.SetProperty(EPNex::LogOutputRate, options.logNextStepRate, options.numFLPs);

  epn.SetProperty(EPNex::OutputSocketType, options.rttackSocketType, options.numFLPs + 1);
  epn.SetProperty(EPNex::OutputSndBufSize, options.rttackBufSize, options.numFLPs + 1);
  epn.SetProperty(EPNex::OutputMethod, options.rttackMethod, options.numFLPs + 1);
  epn.SetProperty(EPNex::OutputAddress, "tcp://127.0.0.1:1234", options.numFLPs + 1);
  epn.SetProperty(EPNex::LogOutputRate, options.logRttackRate, options.numFLPs + 1);

  epn.ChangeState(EPNex::SETOUTPUT);
  epn.ChangeState(EPNex::SETINPUT);
  epn.ChangeState(EPNex::BIND);

  LOG(INFO) << "I am here";

  dds::CKeyValue ddsKeyValue;

  ddsKeyValue.putValue("testEPNdistributedInputAddress", epn.GetProperty(EPNex::InputAddress, "", 0));

  // DDS
  // Waiting for properties
  dds::CKeyValue::valuesMap_t values;
  
  // Subscribe on key update events
  {
  std::mutex keyMutex;
  std::condition_variable keyCondition;
  
  ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

  ddsKeyValue.getValues("heartbeatInputAddress", &values);
  while (values.size() != options.numFLPs) {
    std::unique_lock<std::mutex> lock(keyMutex);
    keyCondition.wait_until(lock, std::chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("heartbeatInputAddress", &values);
  }
  }

  dds::CKeyValue::valuesMap_t::const_iterator it_values = values.begin();
  for (int i = 0; i < options.numFLPs; ++i) {
    epn.SetProperty(EPNex::OutputAddress, it_values->second, i);
    it_values++;
  }

  // output to send to next step. TODO: rename sockets to something more meaningfull
  epn.SetProperty(EPNex::OutputAddress, "tcp://127.0.0.1:5590", options.numFLPs);

  values.clear();

  // Subscribe on key update events
  {
  std::mutex keyMutex;
  std::condition_variable keyCondition;
  
  ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

  ddsKeyValue.getValues("testFLPSamplerInputAddress", &values);
  while (values.empty()) {
    std::unique_lock<std::mutex> lock(keyMutex);
    keyCondition.wait_until(lock, std::chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("testFLPSamplerInputAddress", &values);
  }
  }

  epn.SetProperty(EPNex::OutputAddress, values.begin()->second, options.numFLPs + 1);









  epn.ChangeState(EPNex::CONNECT);
  epn.ChangeState(EPNex::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(epn.fRunningMutex);
  while (!epn.fRunningFinished)
  {
    epn.fRunningCondition.wait(lock);
  }

  epn.ChangeState(EPNex::STOP);
  epn.ChangeState(EPNex::END);

  return 0;
}
