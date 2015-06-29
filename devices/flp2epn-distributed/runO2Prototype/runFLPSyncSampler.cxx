/**
 * runFLPSyncSampler.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>
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

typedef struct DeviceOptions
{
  string id;
  int eventRate;
  int ioThreads;

  string dataOutSocketType;
  int dataOutBufSize;
  string dataOutMethod;
  // string dataOutAddress;
  int dataOutRateLogging;

  string ackInSocketType;
  int ackInBufSize;
  string ackInMethod;
  // string ackInAddress;
  int ackInRateLogging;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw runtime_error("Internal error: options' container is empty.");

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("event-rate", bpo::value<int>()->default_value(100), "Event rate limit in maximum number of events per second")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")

    ("data-out-socket-type", bpo::value<string>()->default_value("pub"), "Data output socket type: pub/push")
    ("data-out-buff-size", bpo::value<int>()->default_value(100), "Data output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-out-method", bpo::value<string>()->default_value("bind"), "Data output method: bind/connect")
    // ("data-out-address", bpo::value<string>()->required(), "Data output address, e.g.: \"tcp://localhost:5555\"")
    ("data-out-rate-logging", bpo::value<int>()->default_value(0), "Log output rate on data socket, 1/0")

    ("ack-in-socket-type", bpo::value<string>()->default_value("pull"), "Acknowledgement Input socket type: sub/pull")
    ("ack-in-buff-size", bpo::value<int>()->default_value(100), "Acknowledgement Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("ack-in-method", bpo::value<string>()->default_value("bind"), "Acknowledgement Input method: bind/connect")
    // ("ack-in-address", bpo::value<string>()->required(), "Acknowledgement Input address, e.g.: \"tcp://localhost:5555\"")
    ("ack-in-rate-logging", bpo::value<int>()->default_value(0), "Log input rate on Acknowledgement socket, 1/0")

    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "FLP Sync Sampler" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                       { _options->id                = vm["id"].as<string>(); }
  if (vm.count("event-rate"))               { _options->eventRate         = vm["event-rate"].as<int>(); }
  if (vm.count("io-threads"))               { _options->ioThreads         = vm["io-threads"].as<int>(); }

  if (vm.count("data-out-socket-type"))  { _options->dataOutSocketType  = vm["data-out-socket-type"].as<string>(); }
  if (vm.count("data-out-buff-size"))    { _options->dataOutBufSize     = vm["data-out-buff-size"].as<int>(); }
  if (vm.count("data-out-method"))       { _options->dataOutMethod      = vm["data-out-method"].as<string>(); }
  // if (vm.count("data-out-address"))      { _options->dataOutAddress     = vm["data-out-address"].as<string>(); }
  if (vm.count("data-out-rate-logging")) { _options->dataOutRateLogging = vm["data-out-rate-logging"].as<int>(); }

  if (vm.count("ack-in-socket-type"))    { _options->ackInSocketType   = vm["ack-in-socket-type"].as<string>(); }
  if (vm.count("ack-in-buff-size"))      { _options->ackInBufSize      = vm["ack-in-buff-size"].as<int>(); }
  if (vm.count("ack-in-method"))         { _options->ackInMethod       = vm["ack-in-method"].as<string>(); }
  // if (vm.count("ack-in-address"))        { _options->ackInAddress      = vm["ack-in-address"].as<string>(); }
  if (vm.count("ack-in-rate-logging"))   { _options->ackInRateLogging  = vm["ack-in-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char** argv)
{
  FLPSyncSampler sampler;
  sampler.CatchSignals();

  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  } catch (exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  LOG(INFO) << "FLP Sync Sampler, ID: " << options.id << " (PID: " << getpid() << ")";

  map<string,string> IPs;
  FairMQ::tools::getHostIPs(IPs);

  stringstream ss;

  if (IPs.count("ib0")) {
    ss << "tcp://" << IPs["ib0"];
  } else if (IPs.count("eth0")) {
    ss << "tcp://" << IPs["eth0"];
  } else {
    LOG(ERROR) << "Could not find ib0 or eth0 interface";
    exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Running on " << ss.str();

  ss << ":5655";

  string ownAddress = ss.str();

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  sampler.SetTransport(transportFactory);

  sampler.SetProperty(FLPSyncSampler::Id, options.id);
  sampler.SetProperty(FLPSyncSampler::NumIoThreads, options.ioThreads);
  sampler.SetProperty(FLPSyncSampler::EventRate, options.eventRate);

  FairMQChannel dataOutChannel(options.dataOutSocketType, options.dataOutMethod, ownAddress);
  dataOutChannel.UpdateSndBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRcvBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRateLogging(options.dataOutRateLogging);
  sampler.fChannels["data-out"].push_back(dataOutChannel);

  FairMQChannel ackInChannel(options.ackInSocketType, options.ackInMethod, ownAddress);
  ackInChannel.UpdateSndBufSize(options.ackInBufSize);
  ackInChannel.UpdateRcvBufSize(options.ackInBufSize);
  ackInChannel.UpdateRateLogging(options.ackInRateLogging);
  sampler.fChannels["ack-in"].push_back(ackInChannel);

  sampler.ChangeState("INIT_DEVICE");
  sampler.WaitForInitialValidation();

  // Advertise the bound addresses via DDS properties
  dds::key_value::CKeyValue ddsKeyValue;
  ddsKeyValue.putValue("FLPSyncSamplerOutputAddress", sampler.fChannels["data-out"].at(0).GetAddress());
  ddsKeyValue.putValue("FLPSyncSamplerInputAddress", sampler.fChannels["ack-in"].at(0).GetAddress());

  sampler.WaitForEndOfState("INIT_DEVICE");

  sampler.ChangeState("INIT_TASK");
  sampler.WaitForEndOfState("INIT_TASK");

  sampler.ChangeState("RUN");
  sampler.WaitForEndOfState("RUN");

  sampler.ChangeState("RESET_TASK");
  sampler.WaitForEndOfState("RESET_TASK");

  sampler.ChangeState("RESET_DEVICE");
  sampler.WaitForEndOfState("RESET_DEVICE");

  sampler.ChangeState("END");

  return 0;
}
