/**
 * runFLPSampler.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>
#include <csignal>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQTransportFactoryZMQ.h"

#include "FLPexSampler.h"

using namespace std;

using namespace AliceO2::Devices;

FLPexSampler sampler;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  sampler.ChangeState(FLPexSampler::STOP);
  sampler.ChangeState(FLPexSampler::END);

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
  string inputAddress;
  int logInputRate;
  string outputSocketType;
  int outputBufSize;
  string outputMethod;
  string outputAddress;
  int logOutputRate;
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
    ("input-address", bpo::value<string>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
    ("log-input-rate", bpo::value<int>()->required(), "Log input rate on socket, 1/0")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("log-output-rate", bpo::value<int>()->required(), "Log output rate on socket, 1/0")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "Test FLP Sampler" << endl << desc;
    return false;
  }

    bpo::notify(vm);

  if (vm.count("id")) {
    _options->id = vm["id"].as<string>();
  }

  if (vm.count("event-rate")) {
    _options->eventRate = vm["event-rate"].as<int>();
  }

  if (vm.count("io-threads")) {
    _options->ioThreads = vm["io-threads"].as<int>();
  }

  if (vm.count("input-socket-type")) {
    _options->inputSocketType = vm["input-socket-type"].as<string>();
  }

  if (vm.count("input-buff-size")) {
    _options->inputBufSize = vm["input-buff-size"].as<int>();
  }

  if (vm.count("input-method")) {
    _options->inputMethod = vm["input-method"].as<string>();
  }

  if (vm.count("input-address")) {
    _options->inputAddress = vm["input-address"].as<string>();
  }

  if (vm.count("log-input-rate")) {
    _options->logInputRate = vm["log-input-rate"].as<int>();
  }

  if (vm.count("output-socket-type")) {
    _options->outputSocketType = vm["output-socket-type"].as<string>();
  }

  if (vm.count("output-buff-size")) {
    _options->outputBufSize = vm["output-buff-size"].as<int>();
  }

  if (vm.count("output-method")) {
    _options->outputMethod = vm["output-method"].as<string>();
  }

  if (vm.count("output-address")) {
    _options->outputAddress = vm["output-address"].as<string>();
  }

  if (vm.count("log-output-rate")) {
    _options->logOutputRate = vm["log-output-rate"].as<int>();
  }

  return true;
}

int main(int argc, char** argv)
{
  s_catch_signals();

  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  } catch (const exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  LOG(INFO) << "PID: " << getpid();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  sampler.SetTransport(transportFactory);

  sampler.SetProperty(FLPexSampler::Id, options.id);
  sampler.SetProperty(FLPexSampler::NumIoThreads, options.ioThreads);
  sampler.SetProperty(FLPexSampler::EventRate, options.eventRate);

  sampler.SetProperty(FLPexSampler::NumInputs, 1);
  sampler.SetProperty(FLPexSampler::NumOutputs, 1);

  sampler.ChangeState(FLPexSampler::INIT);

  sampler.SetProperty(FLPexSampler::InputSocketType, options.inputSocketType);
  sampler.SetProperty(FLPexSampler::InputSndBufSize, options.inputBufSize);
  sampler.SetProperty(FLPexSampler::InputMethod, options.inputMethod);
  sampler.SetProperty(FLPexSampler::InputAddress, options.inputAddress);
  sampler.SetProperty(FLPexSampler::LogInputRate, options.logInputRate);

  sampler.SetProperty(FLPexSampler::OutputSocketType, options.outputSocketType);
  sampler.SetProperty(FLPexSampler::OutputSndBufSize, options.outputBufSize);
  sampler.SetProperty(FLPexSampler::OutputMethod, options.outputMethod);
  sampler.SetProperty(FLPexSampler::OutputAddress, options.outputAddress);
  sampler.SetProperty(FLPexSampler::LogOutputRate, options.logOutputRate);

  try {
    sampler.ChangeState(FLPexSampler::SETOUTPUT);
    sampler.ChangeState(FLPexSampler::SETINPUT);
// temporary check to allow compilation with older fairmq version
#ifdef FAIRMQ_INTERFACE_VERSION
    sampler.ChangeState(FLPexSampler::BIND);
    sampler.ChangeState(FLPexSampler::CONNECT);
#endif
    sampler.ChangeState(FLPexSampler::RUN);
  } catch (const exception& e) {
      LOG(ERROR) << e.what();
  }

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(sampler.fRunningMutex);
  while (!sampler.fRunningFinished) {
    sampler.fRunningCondition.wait(lock);
  }

  sampler.ChangeState(FLPexSampler::STOP);
  sampler.ChangeState(FLPexSampler::END);

  return 0;
}
