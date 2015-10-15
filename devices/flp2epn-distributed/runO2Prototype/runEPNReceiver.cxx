/**
 * runEPNReceiver.cxx
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

#include "EPNReceiver.h"

#include "KeyValue.h" // DDS

using namespace std;
using namespace AliceO2::Devices;

typedef struct DeviceOptions
{
  string id;
  int ioThreads;
  int heartbeatIntervalInMs;
  int bufferTimeoutInMs;
  int numFLPs;
  int testMode;

  string dataInSocketType;
  int dataInBufSize;
  string dataInMethod;
  // string dataInAddress;
  int dataInRateLogging;

  string dataOutSocketType;
  int dataOutBufSize;
  string dataOutMethod;
  // string dataOutAddress;
  int dataOutRateLogging;

  string hbOutSocketType;
  int hbOutBufSize;
  string hbOutMethod;
  // string hbOutAddress;
  int hbOutRateLogging;

  string ackOutSocketType;
  int ackOutBufSize;
  string ackOutMethod;
  // string ackOutAddress;
  int ackOutRateLogging;
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
    ("heartbeat-interval", bpo::value<int>()->default_value(5000), "Heartbeat interval in milliseconds")
    ("buffer-timeout", bpo::value<int>()->default_value(5000), "Buffer timeout in milliseconds")
    ("num-flps", bpo::value<int>()->required(), "Number of FLPs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")

    ("data-in-socket-type", bpo::value<string>()->default_value("pull"), "Data input socket type: sub/pull")
    ("data-in-buff-size", bpo::value<int>()->default_value(10), "Data input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-in-method", bpo::value<string>()->default_value("bind"), "Data input method: bind/connect")
    // ("data-in-address", bpo::value<string>()->required(), "Data input address, e.g.: \"tcp://localhost:5555\"")
    ("data-in-rate-logging", bpo::value<int>()->default_value(1), "Log input rate on data socket, 1/0")

    ("data-out-socket-type", bpo::value<string>()->default_value("push"), "Data output socket type: pub/push")
    ("data-out-buff-size", bpo::value<int>()->default_value(10), "Data output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-out-method", bpo::value<string>()->default_value("bind"), "Data output method: bind/connect")
    // ("data-out-address", bpo::value<string>()->required(), "Data output address, e.g.: \"tcp://localhost:5555\"")
    ("data-out-rate-logging", bpo::value<int>()->default_value(1), "Log output rate on data socket, 1/0")

    ("hb-out-socket-type", bpo::value<string>()->default_value("pub"), "Heartbeat output socket type: pub/push")
    ("hb-out-buff-size", bpo::value<int>()->default_value(100), "Heartbeat output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("hb-out-method", bpo::value<string>()->default_value("connect"), "Heartbeat output method: bind/connect")
    // ("hb-out-address", bpo::value<string>()->required(), "Heartbeat output address, e.g.: \"tcp://localhost:5555\"")
    ("hb-out-rate-logging", bpo::value<int>()->default_value(0), "Log output rate on heartbeat socket, 1/0")

    ("ack-out-socket-type", bpo::value<string>()->default_value("push"), "Acknowledgement output socket type: pub/push")
    ("ack-out-buff-size", bpo::value<int>()->default_value(100), "Acknowledgement output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("ack-out-method", bpo::value<string>()->default_value("connect"), "Acknowledgement output method: bind/connect")
    // ("ack-out-address", bpo::value<string>()->required() "Acknowledgement output address, e.g.: \"tcp://localhost:5555\"")
    ("ack-out-rate-logging", bpo::value<int>()->default_value(0), "Log output rate on acknowledgement socket, 1/0")

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
  if (vm.count("heartbeat-interval"))    { _options->heartbeatIntervalInMs = vm["heartbeat-interval"].as<int>(); }
  if (vm.count("buffer-timeout"))        { _options->bufferTimeoutInMs     = vm["buffer-timeout"].as<int>(); }
  if (vm.count("num-flps"))              { _options->numFLPs               = vm["num-flps"].as<int>(); }
  if (vm.count("test-mode"))             { _options->testMode              = vm["test-mode"].as<int>(); }

  if (vm.count("data-in-socket-type"))   { _options->dataInSocketType      = vm["data-in-socket-type"].as<string>(); }
  if (vm.count("data-in-buff-size"))     { _options->dataInBufSize         = vm["data-in-buff-size"].as<int>(); }
  if (vm.count("data-in-method"))        { _options->dataInMethod          = vm["data-in-method"].as<string>(); }
  // if (vm.count("data-in-address"))       { _options->dataInAddress         = vm["data-in-address"].as<string>(); }
  if (vm.count("data-in-rate-logging"))  { _options->dataInRateLogging     = vm["data-in-rate-logging"].as<int>(); }

  if (vm.count("data-out-socket-type"))  { _options->dataOutSocketType     = vm["data-out-socket-type"].as<string>(); }
  if (vm.count("data-out-buff-size"))    { _options->dataOutBufSize        = vm["data-out-buff-size"].as<int>(); }
  if (vm.count("data-out-method"))       { _options->dataOutMethod         = vm["data-out-method"].as<string>(); }
  // if (vm.count("data-out-address"))      { _options->dataOutAddress        = vm["data-out-address"].as<string>(); }
  if (vm.count("data-out-rate-logging")) { _options->dataOutRateLogging    = vm["data-out-rate-logging"].as<int>(); }

  if (vm.count("hb-out-socket-type"))    { _options->hbOutSocketType       = vm["hb-out-socket-type"].as<string>(); }
  if (vm.count("hb-out-buff-size"))      { _options->hbOutBufSize          = vm["hb-out-buff-size"].as<int>(); }
  if (vm.count("hb-out-method"))         { _options->hbOutMethod           = vm["hb-out-method"].as<string>(); }
  // if (vm.count("hb-out-address"))        { _options->hbOutAddress          = vm["hb-out-address"].as<string>(); }
  if (vm.count("hb-out-rate-logging"))   { _options->hbOutRateLogging      = vm["hb-out-rate-logging"].as<int>(); }

  if (vm.count("ack-out-socket-type"))   { _options->ackOutSocketType      = vm["ack-out-socket-type"].as<string>(); }
  if (vm.count("ack-out-buff-size"))     { _options->ackOutBufSize         = vm["ack-out-buff-size"].as<int>(); }
  if (vm.count("ack-out-method"))        { _options->ackOutMethod          = vm["ack-out-method"].as<string>(); }
  // if (vm.count("ack-out-address"))       { _options->ackOutAddress         = vm["ack-out-address"].as<string>(); }
  if (vm.count("ack-out-rate-logging"))  { _options->ackOutRateLogging     = vm["ack-out-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char** argv)
{
  // create the device
  EPNReceiver epn;
  // let the device catch interrupt signals (SIGINT, SIGTERM)
  epn.CatchSignals();

  // create container for command line options and fill it
  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  } catch (exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  LOG(INFO) << "EPN Receiver, ID: " << options.id << " (PID: " << getpid() << ")";

  // container to hold the IP address of the node we are running on
  map<string,string> IPs;
  FairMQ::tools::getHostIPs(IPs);

  stringstream ss;

  // With TCP, we want to run either one Eth or Infiniband, try to find available interfaces
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

  // store the IP addresses to be given to device for initialization
  string ownAddress = ss.str();

  // configure the transport interface
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
  epn.SetTransport(transportFactory);

  // configure device
  epn.SetProperty(EPNReceiver::Id, options.id);
  epn.SetProperty(EPNReceiver::NumIoThreads, options.ioThreads);
  epn.SetProperty(EPNReceiver::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);
  epn.SetProperty(EPNReceiver::BufferTimeoutInMs, options.bufferTimeoutInMs);
  epn.SetProperty(EPNReceiver::NumFLPs, options.numFLPs);
  epn.SetProperty(EPNReceiver::TestMode, options.testMode);

  // configure data input channel (port will be configured dynamically)
  FairMQChannel dataInChannel(options.dataInSocketType, options.dataInMethod, ownAddress);
  dataInChannel.UpdateSndBufSize(options.dataInBufSize);
  dataInChannel.UpdateRcvBufSize(options.dataInBufSize);
  dataInChannel.UpdateRateLogging(options.dataInRateLogging);
  epn.fChannels["data-in"].push_back(dataInChannel);

  // configure data output channel (port will be configured dynamically)
  FairMQChannel dataOutChannel(options.dataOutSocketType, options.dataOutMethod, ownAddress);
  dataOutChannel.UpdateSndBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRcvBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRateLogging(options.dataOutRateLogging);
  epn.fChannels["data-out"].push_back(dataOutChannel);

  // configure heartbeat output channels
  for (int i = 0; i < options.numFLPs; ++i) {
    FairMQChannel hbOutChannel(options.hbOutSocketType, options.hbOutMethod, "");
    hbOutChannel.UpdateSndBufSize(options.hbOutBufSize);
    hbOutChannel.UpdateRcvBufSize(options.hbOutBufSize);
    hbOutChannel.UpdateRateLogging(options.hbOutRateLogging);
    epn.fChannels["heartbeat-out"].push_back(hbOutChannel);
  }

  // In test mode, configure acknowledgement channel to the FLPSyncSampler
  if (options.testMode == 1) {
    FairMQChannel ackOutChannel(options.ackOutSocketType, options.ackOutMethod, "");
    ackOutChannel.UpdateSndBufSize(options.ackOutBufSize);
    ackOutChannel.UpdateRcvBufSize(options.ackOutBufSize);
    ackOutChannel.UpdateRateLogging(options.ackOutRateLogging);
    epn.fChannels["ack-out"].push_back(ackOutChannel);
  }

  // Initialize the device with the configured properties (asynchronous).
  epn.ChangeState("INIT_DEVICE");
  // Wait for initial validation.
  // Missing properties (such as addresses for connecting) will be revalidated asynchronously
  epn.WaitForInitialValidation();

  // create DDS key value store
  dds::key_value::CKeyValue ddsKeyValue;

  // Advertise the bound data input address via DDS.
  ddsKeyValue.putValue("EPNReceiverInputAddress", epn.fChannels["data-in"].at(0).GetAddress());
  // In regular mode, advertise the bound data output address via DDS.
  if (options.testMode == 0) {
    ddsKeyValue.putValue("EPNReceiverOutputAddress", epn.fChannels["data-out"].at(0).GetAddress());
  }

  // Initialize DDS store to receive properties
  dds::key_value::CKeyValue::valuesMap_t values;
  {
  mutex keyMutex;
  condition_variable keyCondition;

  ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

  // Receive FLPSender heartbeat input addresses from DDS.
  LOG(DEBUG) << "Waiting for FLPSender heartbeat input addresses from DDS...";
  ddsKeyValue.getValues("FLPSenderHeartbeatInputAddress", &values);
  while (values.size() != options.numFLPs) {
    unique_lock<mutex> lock(keyMutex);
    keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("FLPSenderHeartbeatInputAddress", &values);
  }
  }
  LOG(DEBUG) << "Received all " << options.numFLPs << " addresses from DDS.";

  // Update device properties with the received addresses.
  auto it_values = values.begin();
  for (int i = 0; i < options.numFLPs; ++i) {
    epn.fChannels["heartbeat-out"].at(i).UpdateAddress(it_values->second);
    it_values++;
  }

  // In test mode, get the value of the FLPSyncSampler input address for the feedback socket.
  if (options.testMode == 1) {
    values.clear();
    {
    mutex keyMutex;
    condition_variable keyCondition;

    ddsKeyValue.subscribe([&keyCondition](const string& _key, const string& _value) {keyCondition.notify_all();});

    LOG(DEBUG) << "Waiting for FLPSyncSampler Input Address from DDS...";
    ddsKeyValue.getValues("FLPSyncSamplerInputAddress", &values);
    while (values.empty()) {
      unique_lock<mutex> lock(keyMutex);
      keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
      ddsKeyValue.getValues("FLPSyncSamplerInputAddress", &values);
    }
    }
    LOG(DEBUG) << "Received FLPSyncSampler Input Address from DDS.";

    // Update device properties with the received address.
    epn.fChannels["ack-out"].at(0).UpdateAddress(values.begin()->second);
  }

  // Wait for the device initialization to finish.
  epn.WaitForEndOfState("INIT_DEVICE");

  epn.ChangeState("INIT_TASK");
  epn.WaitForEndOfState("INIT_TASK");

  epn.ChangeState("RUN");
  epn.WaitForEndOfState("RUN");

  epn.ChangeState("RESET_TASK");
  epn.WaitForEndOfState("RESET_TASK");

  epn.ChangeState("RESET_DEVICE");
  epn.WaitForEndOfState("RESET_DEVICE");

  epn.ChangeState("END");

  return 0;
}
