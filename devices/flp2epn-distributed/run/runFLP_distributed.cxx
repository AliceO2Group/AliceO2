/**
 * runFLP_distributed.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>
#include <csignal>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "O2FLPex.h"

#ifdef NANOMSG
#include "FairMQTransportFactoryNN.h"
#else
#include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;

O2FLPex flp;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  flp.ChangeState(O2FLPex::STOP);
  flp.ChangeState(O2FLPex::END);

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
  int numInputs;
  int numOutputs;
  int heartbeatTimeoutInMs;
  int sendOffset;
  vector<string> inputSocketType;
  vector<int> inputBufSize;
  vector<string> inputMethod;
  vector<string> inputAddress;
  vector<string> outputSocketType;
  vector<int> outputBufSize;
  vector<string> outputMethod;
  vector<string> outputAddress;
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
    ("num-inputs", bpo::value<int>()->required(), "Number of FLP input sockets")
    ("num-outputs", bpo::value<int>()->required(), "Number of FLP output sockets")
    ("heartbeat-timeout", bpo::value<int>()->default_value(20000), "Heartbeat timeout in milliseconds")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("input-socket-type", bpo::value< vector<string> >()->required(), "Input socket type: sub/pull")
    ("input-buff-size", bpo::value< vector<int> >()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("input-method", bpo::value< vector<string> >()->required(), "Input method: bind/connect")
    ("input-address", bpo::value< vector<string> >()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("output-socket-type", bpo::value< vector<string> >()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value< vector<int> >()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value< vector<string> >()->required(), "Output method: bind/connect")
    ("output-address", bpo::value< vector<string> >()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if ( vm.count("help") ) {
    LOG(INFO) << "FLP" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if ( vm.count("id") ) {
    _options->id = vm["id"].as<string>();
  }

  if ( vm.count("io-threads") ) {
    _options->ioThreads = vm["io-threads"].as<int>();
  }

  if ( vm.count("num-inputs") ) {
    _options->numInputs = vm["num-inputs"].as<int>();
  }

  if ( vm.count("num-outputs") ) {
    _options->numOutputs = vm["num-outputs"].as<int>();
  }

  if ( vm.count("heartbeat-timeout") ) {
    _options->heartbeatTimeoutInMs = vm["heartbeat-timeout"].as<int>();
  }

  if ( vm.count("send-offset") ) {
    _options->sendOffset = vm["send-offset"].as<int>();
  }

  if ( vm.count("input-socket-type") ) {
    _options->inputSocketType = vm["input-socket-type"].as<vector <string> >();
  }

  if ( vm.count("input-buff-size") ) {
    _options->inputBufSize = vm["input-buff-size"].as<vector <int> >();
  }

  if ( vm.count("input-method") ) {
    _options->inputMethod = vm["input-method"].as<vector <string> >();
  }

  if ( vm.count("input-address") ) {
    _options->inputAddress = vm["input-address"].as<vector <string> >();
  }

  if ( vm.count("output-socket-type") ) {
    _options->outputSocketType = vm["output-socket-type"].as< vector<string> >();
  }

  if ( vm.count("output-buff-size") ) {
    _options->outputBufSize = vm["output-buff-size"].as< vector<int> >();
  }

  if ( vm.count("output-method") ) {
    _options->outputMethod = vm["output-method"].as< vector<string> >();
  }

  if ( vm.count("output-address") ) {
    _options->outputAddress = vm["output-address"].as< vector<string> >();
  }

  return true;
}

int main(int argc, char** argv)
{
  s_catch_signals();

  DeviceOptions_t options;
  try
  {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  }
  catch (exception& e)
  {
    LOG(ERROR) << e.what();
    return 1;
  }

  LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  flp.SetTransport(transportFactory);

  flp.SetProperty(O2FLPex::Id, options.id);
  flp.SetProperty(O2FLPex::NumIoThreads, options.ioThreads);

  flp.SetProperty(O2FLPex::NumInputs, options.numInputs);
  flp.SetProperty(O2FLPex::NumOutputs, options.numOutputs);
  flp.SetProperty(O2FLPex::HeartbeatTimeoutInMs, options.heartbeatTimeoutInMs);
  flp.SetProperty(O2FLPex::SendOffset, options.sendOffset);

  flp.ChangeState(O2FLPex::INIT);

  for (int i = 0; i < options.numInputs; ++i) {
    flp.SetProperty(O2FLPex::InputSocketType, options.inputSocketType.at(i), i);
    flp.SetProperty(O2FLPex::InputRcvBufSize, options.inputBufSize.at(i), i);
    flp.SetProperty(O2FLPex::InputMethod, options.inputMethod.at(i), i);
    flp.SetProperty(O2FLPex::InputAddress, options.inputAddress.at(i), i);
  }

  for (int i = 0; i < options.numOutputs; ++i) {
    flp.SetProperty(O2FLPex::OutputSocketType, options.outputSocketType.at(i), i);
    flp.SetProperty(O2FLPex::OutputRcvBufSize, options.outputBufSize.at(i), i);
    flp.SetProperty(O2FLPex::OutputMethod, options.outputMethod.at(i), i);
    flp.SetProperty(O2FLPex::OutputAddress, options.outputAddress.at(i), i);
  }

  flp.ChangeState(O2FLPex::SETOUTPUT);
  flp.ChangeState(O2FLPex::SETINPUT);
  flp.ChangeState(O2FLPex::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(flp.fRunningMutex);
  while (!flp.fRunningFinished) {
    flp.fRunningCondition.wait(lock);
  }

  flp.ChangeState(O2FLPex::STOP);
  flp.ChangeState(O2FLPex::END);

  return 0;
}
