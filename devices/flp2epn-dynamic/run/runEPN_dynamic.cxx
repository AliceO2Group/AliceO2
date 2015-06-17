/**
 * runEPN_dynamic.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <iostream>
#include <csignal>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "O2EPNex.h"

#ifdef NANOMSG
    #include "FairMQTransportFactoryNN.h"
#else
    #include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;

O2EPNex epn;

static void s_signal_handler (int signal)
{
    cout << endl << "Caught signal " << signal << endl;

    epn.ChangeState(O2EPNex::END);

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
    string inputSocketType;
    int inputBufSize;
    string inputMethod;
    string inputAddress;
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
        ("num-outputs", bpo::value<int>()->required(), "Number of EPN output sockets")
        ("heartbeat-interval", bpo::value<int>()->default_value(5000), "Heartbeat interval in milliseconds")
        ("input-socket-type", bpo::value<string>()->required(), "Input socket type: sub/pull")
        ("input-buff-size", bpo::value<int>()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
        ("input-method", bpo::value<string>()->required(), "Input method: bind/connect")
        ("input-address", bpo::value<string>()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
        ("output-socket-type", bpo::value< vector<string> >()->required(), "Output socket type: pub/push")
        ("output-buff-size", bpo::value< vector<int> >()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
        ("output-method", bpo::value< vector<string> >()->required(), "Output method: bind/connect")
        ("output-address", bpo::value< vector<string> >()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
        ("help", "Print help messages");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

    if ( vm.count("help") )
    {
        LOG(INFO) << "EPN" << endl << desc;
        return false;
    }

    bpo::notify(vm);

    if ( vm.count("id") )
        _options->id = vm["id"].as<string>();

    if ( vm.count("io-threads") )
        _options->ioThreads = vm["io-threads"].as<int>();

    if ( vm.count("num-outputs") )
        _options->numOutputs = vm["num-outputs"].as<int>();

    if ( vm.count("heartbeat-interval") )
        _options->heartbeatIntervalInMs = vm["heartbeat-interval"].as<int>();

    if ( vm.count("input-socket-type") )
        _options->inputSocketType = vm["input-socket-type"].as<string>();

    if ( vm.count("input-buff-size") )
        _options->inputBufSize = vm["input-buff-size"].as<int>();

    if ( vm.count("input-method") )
        _options->inputMethod = vm["input-method"].as<string>();

    if ( vm.count("input-address") )
        _options->inputAddress = vm["input-address"].as<string>();

    if ( vm.count("output-socket-type") )
        _options->outputSocketType = vm["output-socket-type"].as< vector<string> >();

    if ( vm.count("output-buff-size") )
        _options->outputBufSize = vm["output-buff-size"].as< vector<int> >();

    if ( vm.count("output-method") )
        _options->outputMethod = vm["output-method"].as< vector<string> >();

    if ( vm.count("output-address") )
        _options->outputAddress = vm["output-address"].as< vector<string> >();

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

    epn.SetTransport(transportFactory);

    epn.SetProperty(O2EPNex::Id, options.id);
    epn.SetProperty(O2EPNex::NumIoThreads, options.ioThreads);

    epn.SetProperty(O2EPNex::HeartbeatIntervalInMs, options.heartbeatIntervalInMs);

    FairMQChannel inputChannel(options.inputSocketType, options.inputMethod, options.inputAddress);
    inputChannel.UpdateSndBufSize(options.inputBufSize);
    inputChannel.UpdateRcvBufSize(options.inputBufSize);
    inputChannel.UpdateRateLogging(1);

    epn.fChannels["data-in"].push_back(inputChannel);

    for (int i = 0; i < options.outputAddress.size(); ++i)
    {
        FairMQChannel outputChannel(options.outputSocketType.at(i), options.outputMethod.at(i), options.outputAddress.at(i));
        outputChannel.UpdateSndBufSize(options.outputBufSize.at(i));
        outputChannel.UpdateRcvBufSize(options.outputBufSize.at(i));
        outputChannel.UpdateRateLogging(1);

        epn.fChannels["data-out"].push_back(outputChannel);
    }

    epn.ChangeState("INIT_DEVICE");
    epn.WaitForEndOfState("INIT_DEVICE");

    epn.ChangeState("INIT_TASK");
    epn.WaitForEndOfState("INIT_TASK");

    epn.ChangeState("RUN");
    epn.WaitForEndOfState("RUN");

    epn.ChangeState("STOP");

    epn.ChangeState("RESET_TASK");
    epn.WaitForEndOfState("RESET_TASK");

    epn.ChangeState("RESET_DEVICE");
    epn.WaitForEndOfState("RESET_DEVICE");

    epn.ChangeState("END");

    return 0;
}
