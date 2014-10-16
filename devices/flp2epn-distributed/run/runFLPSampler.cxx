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
#include "O2FLPexSampler.h"

#ifdef NANOMSG
#include "FairMQTransportFactoryNN.h"
#else
#include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;

O2FLPexSampler sampler;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  sampler.ChangeState(O2FLPexSampler::STOP);
  sampler.ChangeState(O2FLPexSampler::END);

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
  int eventRate;
  int ioThreads;
  string outputSocketType;
  int outputBufSize;
  string outputMethod;
  string outputAddress;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw std::runtime_error("Internal error: options' container is empty.");

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes")
    ("event-rate", bpo::value<int>()->default_value(0), "Event rate limit in maximum number of events per second")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
    ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
    ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if ( vm.count("help") ) {
    LOG(INFO) << "Test FLP Sampler" << endl << desc;
    return false;
  }

    bpo::notify(vm);

  if ( vm.count("id") ) {
    _options->id = vm["id"].as<string>();
  }

  if ( vm.count("event-size") ) {
    _options->eventSize = vm["event-size"].as<int>();
  }

  if ( vm.count("event-rate") ) {
    _options->eventRate = vm["event-rate"].as<int>();
  }

  if ( vm.count("io-threads") ) {
    _options->ioThreads = vm["io-threads"].as<int>();
  }

  if ( vm.count("output-socket-type") ) {
    _options->outputSocketType = vm["output-socket-type"].as<string>();
  }

  if ( vm.count("output-buff-size") ) {
    _options->outputBufSize = vm["output-buff-size"].as<int>();
  }

  if ( vm.count("output-method") ) {
    _options->outputMethod = vm["output-method"].as<string>();
  }

  if ( vm.count("output-address") ) {
    _options->outputAddress = vm["output-address"].as<string>();
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

  sampler.SetTransport(transportFactory);

  sampler.SetProperty(O2FLPexSampler::Id, options.id);
  sampler.SetProperty(O2FLPexSampler::NumIoThreads, options.ioThreads);
  sampler.SetProperty(O2FLPexSampler::EventSize, options.eventSize);
  sampler.SetProperty(O2FLPexSampler::EventRate, options.eventRate);

  sampler.SetProperty(O2FLPexSampler::NumInputs, 0);
  sampler.SetProperty(O2FLPexSampler::NumOutputs, 1);

  sampler.ChangeState(O2FLPexSampler::INIT);

  sampler.SetProperty(O2FLPexSampler::OutputSocketType, options.outputSocketType);
  sampler.SetProperty(O2FLPexSampler::OutputRcvBufSize, options.outputBufSize);
  sampler.SetProperty(O2FLPexSampler::OutputMethod, options.outputMethod);
  sampler.SetProperty(O2FLPexSampler::OutputAddress, options.outputAddress);

  sampler.ChangeState(O2FLPexSampler::SETOUTPUT);
  sampler.ChangeState(O2FLPexSampler::SETINPUT);
  sampler.ChangeState(O2FLPexSampler::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(sampler.fRunningMutex);
  while (!sampler.fRunningFinished)
  {
    sampler.fRunningCondition.wait(lock);
  }

  sampler.ChangeState(O2FLPexSampler::STOP);
  sampler.ChangeState(O2FLPexSampler::END);

  return 0;
}
