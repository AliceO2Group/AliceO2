/**
 * runFLPSender.cxx
 *
 * @since 2013-04-23
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

#include "FLPSender.h"

#include "KeyValue.h" // DDS

using namespace std;
using namespace AliceO2::Devices;

typedef struct DeviceOptions
{
  string id;
  int flpIndex;
  int eventSize;
  int ioThreads;
  int numEPNs;
  int heartbeatTimeoutInMs;
  int testMode;
  int sendOffset;
  int sendDelay;

  string dataInSocketType;
  int dataInBufSize;
  string dataInMethod;
  // string dataInAddress;
  int dataInRateLogging;

  string dataOutSocketType;
  int dataOutBufSize;
  string dataOutMethod;
  // vector<string> dataOutAddress;
  int dataOutRateLogging;

  string hbInSocketType;
  int hbInBufSize;
  string hbInMethod;
  // string hbInAddress;
  int hbInRateLogging;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL)
    throw runtime_error("Internal error: options' container is empty.");

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("flp-index", bpo::value<int>()->default_value(0), "FLP Index (for debugging in test mode)")
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes (test mode)")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("num-epns", bpo::value<int>()->required(), "Number of EPNs")
    ("heartbeat-timeout", bpo::value<int>()->default_value(20000), "Heartbeat timeout in milliseconds")
    ("test-mode", bpo::value<int>()->default_value(0),"Run in test mode")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("send-delay", bpo::value<int>()->default_value(0), "Delay for staggered sending")

    ("data-in-socket-type", bpo::value<string>()->default_value("pull"), "Data input socket type: sub/pull")
    ("data-in-buff-size", bpo::value<int>()->default_value(10), "Data input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-in-method", bpo::value<string>()->default_value("bind"), "Data input method: bind/connect")
    // ("data-in-address", bpo::value<string>()->required(), "Data input address, e.g.: \"tcp://localhost:5555\"")
    ("data-in-rate-logging", bpo::value<int>()->default_value(1), "Log data input rate on socket, 1/0")

    ("data-out-socket-type", bpo::value<string>()->default_value("push"), "Output socket type: pub/push")
    ("data-out-buff-size", bpo::value<int>()->default_value(10), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-out-method", bpo::value<string>()->default_value("connect"), "Output method: bind/connect")
    // ("data-out-address", bpo::value<vector<string>>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("data-out-rate-logging", bpo::value<int>()->default_value(1), "Log output rate on socket, 1/0")

    ("hb-in-socket-type", bpo::value<string>()->default_value("sub"), "Heartbeat in socket type: sub/pull")
    ("hb-in-buff-size", bpo::value<int>()->default_value(100), "Heartbeat in buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("hb-in-method", bpo::value<string>()->default_value("bind"), "Heartbeat in method: bind/connect")
    // ("hb-in-address", bpo::value<string>()->required(), "Heartbeat in address, e.g.: \"tcp://localhost:5555\"")
    ("hb-in-rate-logging", bpo::value<int>()->default_value(0), "Log heartbeat in rate on socket, 1/0")

    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "FLP Sender" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                    { _options->id                   = vm["id"].as<string>(); }
  if (vm.count("flp-index"))             { _options->flpIndex             = vm["flp-index"].as<int>(); }
  if (vm.count("event-size"))            { _options->eventSize            = vm["event-size"].as<int>(); }
  if (vm.count("io-threads"))            { _options->ioThreads            = vm["io-threads"].as<int>(); }
  if (vm.count("num-epns"))              { _options->numEPNs              = vm["num-epns"].as<int>(); }
  if (vm.count("heartbeat-timeout"))     { _options->heartbeatTimeoutInMs = vm["heartbeat-timeout"].as<int>(); }
  if (vm.count("test-mode"))             { _options->testMode             = vm["test-mode"].as<int>(); }
  if (vm.count("send-offset"))           { _options->sendOffset           = vm["send-offset"].as<int>(); }
  if (vm.count("send-delay"))            { _options->sendDelay            = vm["send-delay"].as<int>(); }

  if (vm.count("data-in-socket-type"))   { _options->dataInSocketType     = vm["data-in-socket-type"].as<string>(); }
  if (vm.count("data-in-buff-size"))     { _options->dataInBufSize        = vm["data-in-buff-size"].as<int>(); }
  if (vm.count("data-in-method"))        { _options->dataInMethod         = vm["data-in-method"].as<string>(); }
  // if (vm.count("data-in-address"))       { _options->dataInAddress        = vm["data-in-address"].as<string>(); }
  if (vm.count("data-in-rate-logging"))  { _options->dataInRateLogging    = vm["data-in-rate-logging"].as<int>(); }

  if (vm.count("data-out-socket-type"))  { _options->dataOutSocketType    = vm["data-out-socket-type"].as<string>(); }
  if (vm.count("data-out-buff-size"))    { _options->dataOutBufSize       = vm["data-out-buff-size"].as<int>(); }
  if (vm.count("data-out-method"))       { _options->dataOutMethod        = vm["data-out-method"].as<string>(); }
  // if (vm.count("data-out-address"))      { _options->dataOutAddress       = vm["data-out-address"].as<vector<string>>(); }
  if (vm.count("data-out-rate-logging")) { _options->dataOutRateLogging   = vm["data-out-rate-logging"].as<int>(); }

  if (vm.count("hb-in-socket-type"))     { _options->hbInSocketType       = vm["hb-in-socket-type"].as<string>(); }
  if (vm.count("hb-in-buff-size"))       { _options->hbInBufSize          = vm["hb-in-buff-size"].as<int>(); }
  if (vm.count("hb-in-method"))          { _options->hbInMethod           = vm["hb-in-method"].as<string>(); }
  // if (vm.count("hb-in-address"))         { _options->hbInAddress          = vm["hb-in-address"].as<string>(); }
  if (vm.count("hb-in-rate-logging"))    { _options->hbInRateLogging      = vm["hb-in-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char** argv)
{
  FLPSender flp;
  flp.CatchSignals();

  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options))
      return 0;
  } catch (exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  if (options.numEPNs <= 0) {
    LOG(ERROR) << "Configured with 0 EPNs, exiting. Use --num-epns program option.";
    exit(EXIT_FAILURE);
  }

  LOG(INFO) << "FLP Sender, ID: " << options.id << " (PID: " << getpid() << ")";

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

  string ownAddress  = ss.str();

  // DDS
  // Waiting for properties
  dds::key_value::CKeyValue ddsKeyValue;
  dds::key_value::CKeyValue::valuesMap_t values;

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

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  flp.SetTransport(transportFactory);

  // Configure device.
  flp.SetProperty(FLPSender::Id, options.id);
  flp.SetProperty(FLPSender::Index, options.flpIndex);
  flp.SetProperty(FLPSender::NumIoThreads, options.ioThreads);
  flp.SetProperty(FLPSender::EventSize, options.eventSize);
  flp.SetProperty(FLPSender::HeartbeatTimeoutInMs, options.heartbeatTimeoutInMs);
  flp.SetProperty(FLPSender::TestMode, options.testMode);
  flp.SetProperty(FLPSender::SendOffset, options.sendOffset);

  // Configure data input channel (address is set later when received from DDS).
  FairMQChannel dataInChannel(options.dataInSocketType, options.dataInMethod, "");
  dataInChannel.UpdateSndBufSize(options.dataInBufSize);
  dataInChannel.UpdateRcvBufSize(options.dataInBufSize);
  dataInChannel.UpdateRateLogging(options.dataInRateLogging);
  flp.fChannels["data-in"].push_back(dataInChannel);

  // Configure data output channels (address is set later when received from DDS).
  for (int i = 0; i < options.numEPNs; ++i) {
    FairMQChannel dataOutChannel(options.dataOutSocketType, options.dataOutMethod, "");
    dataOutChannel.UpdateSndBufSize(options.dataOutBufSize);
    dataOutChannel.UpdateRcvBufSize(options.dataOutBufSize);
    dataOutChannel.UpdateRateLogging(options.dataOutRateLogging);
    flp.fChannels["data-out"].push_back(dataOutChannel);
  }

  // configure heartbeat input channel
  FairMQChannel hbInChannel(options.hbInSocketType, options.hbInMethod, ownAddress);
  hbInChannel.UpdateSndBufSize(options.hbInBufSize);
  hbInChannel.UpdateRcvBufSize(options.hbInBufSize);
  hbInChannel.UpdateRateLogging(options.hbInRateLogging);
  flp.fChannels["heartbeat-in"].push_back(hbInChannel);

  if (options.testMode == 1) {
    // In test mode, assign address that was received from the FLPSyncSampler via DDS.
    flp.fChannels["data-in"].at(0).UpdateAddress(values.begin()->second); // FLPSyncSampler signal
  } else {
    // In regular mode, assign placeholder address, that will be set when binding.
    flp.fChannels["data-in"].at(0).UpdateAddress(ownAddress); // data
  }

  flp.ChangeState("INIT_DEVICE");
  flp.WaitForInitialValidation();

  if (options.testMode == 0) {
    // In regular mode, advertise the bound data input address to the DDS.
    ddsKeyValue.putValue("FLPSenderInputAddress", flp.fChannels["data-in"].at(0).GetAddress());
  }

  ddsKeyValue.putValue("FLPSenderHeartbeatInputAddress", flp.fChannels["heartbeat-in"].at(0).GetAddress());

  dds::key_value::CKeyValue::valuesMap_t epn_addr_values;

  // Receive the EPNReceiver input addresses from DDS.
  {
  mutex keyMutex;
  condition_variable keyCondition;

  ddsKeyValue.subscribe([&keyCondition](const string& /*_key*/, const string& /*_value*/) {keyCondition.notify_all();});
  ddsKeyValue.getValues("EPNReceiverInputAddress", &epn_addr_values);
  while (epn_addr_values.size() != options.numEPNs) {
    unique_lock<mutex> lock(keyMutex);
    keyCondition.wait_until(lock, chrono::system_clock::now() + chrono::milliseconds(1000));
    ddsKeyValue.getValues("EPNReceiverInputAddress", &epn_addr_values);
  }
  }

  // Assign the received EPNReceiver input addresses to the device.
  auto it_epn_addr_values = epn_addr_values.begin();
  for (int i = 0; i < options.numEPNs; ++i) {
    flp.fChannels["data-out"].at(i).UpdateAddress(it_epn_addr_values->second);
    it_epn_addr_values++;
  }

  // TODO: sort the data channels

  flp.WaitForEndOfState("INIT_DEVICE");

  flp.ChangeState("INIT_TASK");
  flp.WaitForEndOfState("INIT_TASK");

  flp.ChangeState("RUN");
  flp.WaitForEndOfState("RUN");

  flp.ChangeState("RESET_TASK");
  flp.WaitForEndOfState("RESET_TASK");

  flp.ChangeState("RESET_DEVICE");
  flp.WaitForEndOfState("RESET_DEVICE");

  flp.ChangeState("END");

  return 0;
}
