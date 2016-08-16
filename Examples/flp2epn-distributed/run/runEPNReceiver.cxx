/**
 * runEPNReceiver.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"

#include "FLP2EPNex_distributed/EPNReceiver.h"

using namespace std;
using namespace AliceO2::Devices;

struct DeviceOptions
{
  DeviceOptions() :
    id(), ioThreads(0), heartbeatIntervalInMs(0), bufferTimeoutInMs(0), numFLPs(0), testMode(0), interactive(0),
    dataInSocketType(), dataInBufSize(1000), dataInMethod(), dataInAddress(), dataInRateLogging(0),
    dataOutSocketType(), dataOutBufSize(1000), dataOutMethod(), dataOutAddress(), dataOutRateLogging(0),
    hbOutSocketType(), hbOutBufSize(1000), hbOutMethod(), hbOutAddress(), hbOutRateLogging(0),
    ackOutSocketType(), ackOutBufSize(1000), ackOutMethod(), ackOutAddress(), ackOutRateLogging(0) {}

  string id;
  int ioThreads;
  int heartbeatIntervalInMs;
  int bufferTimeoutInMs;
  int numFLPs;
  int testMode;
  int interactive;
  string transport;

  string dataInSocketType;
  int dataInBufSize;
  string dataInMethod;
  string dataInAddress;
  int dataInRateLogging;

  string dataOutSocketType;
  int dataOutBufSize;
  string dataOutMethod;
  string dataOutAddress;
  int dataOutRateLogging;

  string hbOutSocketType;
  int hbOutBufSize;
  string hbOutMethod;
  vector<string> hbOutAddress;
  int hbOutRateLogging;

  string ackOutSocketType;
  int ackOutBufSize;
  string ackOutMethod;
  string ackOutAddress;
  int ackOutRateLogging;
};

inline bool parse_cmd_line(int _argc, char *_argv[], DeviceOptions *_options)
{
  if (_options == NULL) {
    throw runtime_error("Internal error: options' container is empty.");
  }

  namespace bpo = boost::program_options;
  bpo::options_description desc("Options");
  desc.add_options()
    ("id", bpo::value<string>()->required(), "Device ID")
    ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
    ("heartbeat-interval", bpo::value<int>()->default_value(5000), "Heartbeat interval in milliseconds")
    ("buffer-timeout", bpo::value<int>()->default_value(1000), "Buffer timeout in milliseconds")
    ("num-flps", bpo::value<int>()->required(), "Number of FLPs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("interactive", bpo::value<int>()->default_value(1), "Run in interactive mode (1/0)")
    ("transport", bpo::value<string>()->default_value("zeromq"), "Transport (zeromq/nanomsg)")

    ("data-in-socket-type", bpo::value<string>()->default_value("pull"), "Data input socket type: sub/pull")
    ("data-in-buff-size", bpo::value<int>()->default_value(10),
     "Data input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-in-method", bpo::value<string>()->default_value("bind"), "Data input method: bind/connect")
    ("data-in-address", bpo::value<string>()->required(), "Data input address, e.g.: \"tcp://localhost:5555\"")
    ("data-in-rate-logging", bpo::value<int>()->default_value(1), "Log input rate on data socket, 1/0")

    ("data-out-socket-type", bpo::value<string>()->default_value("push"), "Output socket type: pub/push")
    ("data-out-buff-size", bpo::value<int>()->default_value(10),
     "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
    ("data-out-method", bpo::value<string>()->default_value("bind"), "Output method: bind/connect")
    ("data-out-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
    ("data-out-rate-logging", bpo::value<int>()->default_value(1), "Log output rate on data socket, 1/0")

    ("hb-out-socket-type", bpo::value<string>()->default_value("pub"), "Heartbeat output socket type: pub/push")
    ("hb-out-buff-size", bpo::value<int>()->default_value(10), "Heartbeat output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")

    ("hb-out-method", bpo::value<string>()->default_value("connect"), "Heartbeat output method: bind/connect")
    ("hb-out-address", bpo::value<vector<string>>()->required(),
     "Heartbeat output address, e.g.: \"tcp://localhost:5555\"")
    ("hb-out-rate-logging", bpo::value<int>()->default_value(0), "Log output rate on heartbeat socket, 1/0")

    ("ack-out-socket-type", bpo::value<string>()->default_value("push"), "Acknowledgement output socket type: pub/push")
    ("ack-out-buff-size", bpo::value<int>()->default_value(10), "Acknowledgement output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")

    ("ack-out-method", bpo::value<string>()->default_value("connect"), "Acknowledgement output method: bind/connect")
    ("ack-out-address", bpo::value<string>()->required(),
     "Acknowledgement output address, e.g.: \"tcp://localhost:5555\"")
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
  if (vm.count("interactive"))           { _options->interactive           = vm["interactive"].as<int>(); }
  if (vm.count("transport"))             { _options->transport             = vm["transport"].as<string>(); }

  if (vm.count("data-in-socket-type"))   { _options->dataInSocketType      = vm["data-in-socket-type"].as<string>(); }
  if (vm.count("data-in-buff-size"))     { _options->dataInBufSize         = vm["data-in-buff-size"].as<int>(); }
  if (vm.count("data-in-method"))        { _options->dataInMethod          = vm["data-in-method"].as<string>(); }
  if (vm.count("data-in-address"))       { _options->dataInAddress         = vm["data-in-address"].as<string>(); }
  if (vm.count("data-in-rate-logging"))  { _options->dataInRateLogging     = vm["data-in-rate-logging"].as<int>(); }

  if (vm.count("data-out-socket-type"))  { _options->dataOutSocketType     = vm["data-out-socket-type"].as<string>(); }
  if (vm.count("data-out-buff-size"))    { _options->dataOutBufSize        = vm["data-out-buff-size"].as<int>(); }
  if (vm.count("data-out-method"))       { _options->dataOutMethod         = vm["data-out-method"].as<string>(); }
  if (vm.count("data-out-address"))      { _options->dataOutAddress        = vm["data-out-address"].as<string>(); }
  if (vm.count("data-out-rate-logging")) { _options->dataOutRateLogging    = vm["data-out-rate-logging"].as<int>(); }

  if (vm.count("hb-out-socket-type"))    { _options->hbOutSocketType       = vm["hb-out-socket-type"].as<string>(); }
  if (vm.count("hb-out-buff-size"))      { _options->hbOutBufSize          = vm["hb-out-buff-size"].as<int>(); }
  if (vm.count("hb-out-method"))         { _options->hbOutMethod           = vm["hb-out-method"].as<string>(); }
  if (vm.count("hb-out-address"))        { _options->hbOutAddress          = vm["hb-out-address"].as<vector<string>>(); }
  if (vm.count("hb-out-rate-logging"))   { _options->hbOutRateLogging      = vm["hb-out-rate-logging"].as<int>(); }

  if (vm.count("ack-out-socket-type"))   { _options->ackOutSocketType      = vm["ack-out-socket-type"].as<string>(); }
  if (vm.count("ack-out-buff-size"))     { _options->ackOutBufSize         = vm["ack-out-buff-size"].as<int>(); }
  if (vm.count("ack-out-method"))        { _options->ackOutMethod          = vm["ack-out-method"].as<string>(); }
  if (vm.count("ack-out-address"))       { _options->ackOutAddress         = vm["ack-out-address"].as<string>(); }
  if (vm.count("ack-out-rate-logging"))  { _options->ackOutRateLogging     = vm["ack-out-rate-logging"].as<int>(); }

  return true;
}

int main(int argc, char **argv)
{
  // create the device
  EPNReceiver epn;
  // let the device handle the interrupt signals (SIGINT, SIGTERM)
  epn.CatchSignals();

  // container for the command line options
  DeviceOptions options;
  // parse the command line options and fill the container
  try {
    if (!parse_cmd_line(argc, argv, &options)) {
      return 0;
    }
  } catch (const exception &e) {
    LOG(ERROR) << e.what();
    return 1;
  }

  // check if the number of provided heartbeat addresses matches the configured number of flpSenders
  if (options.numFLPs != options.hbOutAddress.size()) {
    LOG(ERROR) << "Number of FLPs does not match the number of provided heartbeat output addresses.";
    return 1;
  }

  LOG(INFO) << "EPN Receiver, ID: " << options.id << " (PID: " << getpid() << ")";

  // configure the transport interface
  epn.SetTransport(options.transport);

  // set device properties
  epn.SetProperty(EPNReceiver::Id, options.id);
  epn.SetProperty(EPNReceiver::NumIoThreads, options.ioThreads);
  epn.SetProperty(EPNReceiver::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);
  epn.SetProperty(EPNReceiver::BufferTimeoutInMs, options.bufferTimeoutInMs);
  epn.SetProperty(EPNReceiver::NumFLPs, options.numFLPs);
  epn.SetProperty(EPNReceiver::TestMode, options.testMode);

  // configure data input channel
  FairMQChannel dataInChannel(options.dataInSocketType, options.dataInMethod, options.dataInAddress);
  dataInChannel.UpdateSndBufSize(options.dataInBufSize);
  dataInChannel.UpdateRcvBufSize(options.dataInBufSize);
  dataInChannel.UpdateRateLogging(options.dataInRateLogging);
  epn.fChannels["data-in"].push_back(dataInChannel);

  // configure data output channel
  FairMQChannel dataOutChannel(options.dataOutSocketType, options.dataOutMethod, options.dataOutAddress);
  dataOutChannel.UpdateSndBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRcvBufSize(options.dataOutBufSize);
  dataOutChannel.UpdateRateLogging(options.dataOutRateLogging);
  epn.fChannels["data-out"].push_back(dataOutChannel);

  // configure heartbeats output channels
  for (int i = 0; i < options.numFLPs; ++i) {
    FairMQChannel hbOutChannel(options.hbOutSocketType, options.hbOutMethod, options.hbOutAddress.at(i));
    hbOutChannel.UpdateSndBufSize(options.hbOutBufSize);
    hbOutChannel.UpdateRcvBufSize(options.hbOutBufSize);
    hbOutChannel.UpdateRateLogging(options.hbOutRateLogging);
    epn.fChannels["heartbeat-out"].push_back(hbOutChannel);
  }

  // configure acknowledgement channel
  FairMQChannel ackOutChannel(options.ackOutSocketType, options.ackOutMethod, options.ackOutAddress);
  ackOutChannel.UpdateSndBufSize(options.ackOutBufSize);
  ackOutChannel.UpdateRcvBufSize(options.ackOutBufSize);
  ackOutChannel.UpdateRateLogging(options.ackOutRateLogging);
  epn.fChannels["ack-out"].push_back(ackOutChannel);

  // init the device
  epn.ChangeState("INIT_DEVICE");
  epn.WaitForEndOfState("INIT_DEVICE");

  // init the task (user code)
  epn.ChangeState("INIT_TASK");
  epn.WaitForEndOfState("INIT_TASK");

  // run the device
  epn.ChangeState("RUN");
  if (options.interactive > 0) {
    epn.InteractiveStateLoop();
  } else {
    epn.WaitForEndOfState("RUN");

    epn.ChangeState("RESET_TASK");
    epn.WaitForEndOfState("RESET_TASK");

    epn.ChangeState("RESET_DEVICE");
    epn.WaitForEndOfState("RESET_DEVICE");

    epn.ChangeState("END");
  }

  return 0;
}
