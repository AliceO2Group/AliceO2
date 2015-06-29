/**
 * runFLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "FairMQTransportFactoryZMQ.h"

#include "FLPSender.h"

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
  string dataInAddress;
  int dataInRateLogging;

  string dataOutSocketType;
  int dataOutBufSize;
  string dataOutMethod;
  vector<string> dataOutAddress;
  int dataOutRateLogging;

  string hbInSocketType;
  int hbInBufSize;
  string hbInMethod;
  string hbInAddress;
  int hbInRateLogging;
} DeviceOptions_t;

inline bool parse_cmd_line(int _argc, char* _argv[], DeviceOptions* _options)
{
  if (_options == NULL) {
    throw runtime_error("Internal error: options' container is empty.");
  }

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("flp-index", bpo::value<int>()->default_value(0), "FLP Index (for debugging in test mode)")
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes (test mode)")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("num-epns", bpo::value<int>()->required(), "Number of EPNs")
    ("heartbeat-timeout", bpo::value<int>()->default_value(20000), "Heartbeat timeout in milliseconds")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("send-delay", bpo::value<int>()->default_value(8), "Delay for staggered sending")

    ("data-in-socket-type", bpo::value<string>()->default_value("pull"), "Data input socket type: sub/pull")
    ("data-in-buff-size", bpo::value<int>()->default_value(10), "Data input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-in-method", bpo::value<string>()->default_value("bind"), "Data input method: bind/connect")
    ("data-in-address", bpo::value<string>()->required(), "Data input address, e.g.: \"tcp://localhost:5555\"")
    ("data-in-rate-logging", bpo::value<int>()->default_value(1), "Log data input rate on socket, 1/0")

    ("data-out-socket-type", bpo::value<string>()->default_value("push"), "Output socket type: pub/push")
    ("data-out-buff-size", bpo::value<int>()->default_value(10), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-out-method", bpo::value<string>()->default_value("connect"), "Output method: bind/connect")
    ("data-out-address", bpo::value<vector<string>>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("data-out-rate-logging", bpo::value<int>()->default_value(1), "Log output rate on socket, 1/0")

    ("hb-in-socket-type", bpo::value<string>()->default_value("sub"), "Heartbeat in socket type: sub/pull")
    ("hb-in-buff-size", bpo::value<int>()->default_value(100), "Heartbeat in buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("hb-in-method", bpo::value<string>()->default_value("bind"), "Heartbeat in method: bind/connect")
    ("hb-in-address", bpo::value<string>()->required(), "Heartbeat in address, e.g.: \"tcp://localhost:5555\"")
    ("hb-in-rate-logging", bpo::value<int>()->default_value(0), "Log heartbeat in rate on socket, 1/0")

    ("help", "Print help messages");

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

  if (vm.count("help")) {
    LOG(INFO) << "FLP Sender" << endl << desc;
    return false;
  }

  bpo::notify(vm);

  if (vm.count("id"))                      { _options->id                        = vm["id"].as<string>(); }
  if (vm.count("flp-index"))               { _options->flpIndex                  = vm["flp-index"].as<int>(); }
  if (vm.count("event-size"))              { _options->eventSize                 = vm["event-size"].as<int>(); }
  if (vm.count("io-threads"))              { _options->ioThreads                 = vm["io-threads"].as<int>(); }

  if (vm.count("num-epns"))                { _options->numEPNs                   = vm["num-epns"].as<int>(); }

  if (vm.count("heartbeat-timeout"))       { _options->heartbeatTimeoutInMs      = vm["heartbeat-timeout"].as<int>(); }
  if (vm.count("test-mode"))               { _options->testMode                  = vm["test-mode"].as<int>(); }
  if (vm.count("send-offset"))             { _options->sendOffset                = vm["send-offset"].as<int>(); }
  if (vm.count("send-delay"))              { _options->sendDelay                 = vm["send-delay"].as<int>(); }

  if (vm.count("data-in-socket-type"))  { _options->dataInSocketType       = vm["data-in-socket-type"].as<string>(); }
  if (vm.count("data-in-buff-size"))    { _options->dataInBufSize          = vm["data-in-buff-size"].as<int>(); }
  if (vm.count("data-in-method"))       { _options->dataInMethod           = vm["data-in-method"].as<string>(); }
  if (vm.count("data-in-address"))      { _options->dataInAddress          = vm["data-in-address"].as<string>(); }
  if (vm.count("data-in-rate-logging")) { _options->dataInRateLogging      = vm["data-in-rate-logging"].as<int>(); }

  if (vm.count("data-out-socket-type"))      { _options->dataOutSocketType          = vm["data-out-socket-type"].as<string>(); }
  if (vm.count("data-out-buff-size"))        { _options->dataOutBufSize             = vm["data-out-buff-size"].as<int>(); }
  if (vm.count("data-out-method"))           { _options->dataOutMethod              = vm["data-out-method"].as<string>(); }
  if (vm.count("data-out-address"))          { _options->dataOutAddress             = vm["data-out-address"].as<vector<string>>(); }
  if (vm.count("data-out-rate-logging"))     { _options->dataOutRateLogging         = vm["data-out-rate-logging"].as<int>(); }

  if (vm.count("hb-in-socket-type"))    { _options->hbInSocketType         = vm["hb-in-socket-type"].as<string>(); }
  if (vm.count("hb-in-buff-size"))      { _options->hbInBufSize            = vm["hb-in-buff-size"].as<int>(); }
  if (vm.count("hb-in-method"))         { _options->hbInMethod             = vm["hb-in-method"].as<string>(); }
  if (vm.count("hb-in-address"))        { _options->hbInAddress            = vm["hb-in-address"].as<string>(); }
  if (vm.count("hb-in-rate-logging"))   { _options->hbInRateLogging        = vm["hb-in-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char** argv)
{
  FLPSender flp;
  flp.CatchSignals();

  DeviceOptions_t options;
  try {
    if (!parse_cmd_line(argc, argv, &options)) {
      return 0;
    }
  } catch (const exception& e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  if (options.numEPNs != options.dataOutAddress.size()) {
    LOG(ERROR) << "Number of EPNs does not match the number of provided data output addresses.";
    return 1;
  }

  LOG(INFO) << "FLP Sender, ID: " << options.id << " (PID: " << getpid() << ")";

  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();

  flp.SetTransport(transportFactory);

  flp.SetProperty(FLPSender::Id, options.id);
  flp.SetProperty(FLPSender::Index, options.flpIndex);
  flp.SetProperty(FLPSender::NumIoThreads, options.ioThreads);
  flp.SetProperty(FLPSender::EventSize, options.eventSize);
  flp.SetProperty(FLPSender::HeartbeatTimeoutInMs, options.heartbeatTimeoutInMs);
  flp.SetProperty(FLPSender::TestMode, options.testMode);
  flp.SetProperty(FLPSender::SendOffset, options.sendOffset);
  flp.SetProperty(FLPSender::SendDelay, options.sendDelay);

  // configure data input channel
  FairMQChannel dataInChannel(options.dataInSocketType, options.dataInMethod, options.dataInAddress);
  dataInChannel.UpdateSndBufSize(options.dataInBufSize);
  dataInChannel.UpdateRcvBufSize(options.dataInBufSize);
  dataInChannel.UpdateRateLogging(options.dataInRateLogging);
  flp.fChannels["data-in"].push_back(dataInChannel);

  // configure data output channels
  for (int i = 0; i < options.numEPNs; ++i) {
    FairMQChannel dataOutChannel(options.dataOutSocketType, options.dataOutMethod, options.dataOutAddress.at(i));
    dataOutChannel.UpdateSndBufSize(options.dataOutBufSize);
    dataOutChannel.UpdateRcvBufSize(options.dataOutBufSize);
    dataOutChannel.UpdateRateLogging(options.dataOutRateLogging);
    flp.fChannels["data-out"].push_back(dataOutChannel);
  }

  // configure heartbeat input channel
  FairMQChannel hbInChannel(options.hbInSocketType, options.hbInMethod, options.hbInAddress);
  hbInChannel.UpdateSndBufSize(options.hbInBufSize);
  hbInChannel.UpdateRcvBufSize(options.hbInBufSize);
  hbInChannel.UpdateRateLogging(options.hbInRateLogging);
  flp.fChannels["heartbeat-in"].push_back(hbInChannel);

  flp.ChangeState("INIT_DEVICE");
  flp.WaitForEndOfState("INIT_DEVICE");

  flp.ChangeState("INIT_TASK");
  flp.WaitForEndOfState("INIT_TASK");

  flp.ChangeState("RUN");
  flp.InteractiveStateLoop();

  return 0;
}
