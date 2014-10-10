/**
 * runFLP.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
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
    int eventSize;
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
        ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
        ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
        ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
        ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
        ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
        ("help", "Print help messages");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

    if ( vm.count("help") )
    {
        LOG(INFO) << "FLP" << endl << desc;
        return false;
    }

    bpo::notify(vm);

    if ( vm.count("id") )
        _options->id = vm["id"].as<string>();

    if ( vm.count("event-size") )
        _options->eventSize = vm["event-size"].as<int>();

    if ( vm.count("io-threads") )
        _options->ioThreads = vm["io-threads"].as<int>();

    if ( vm.count("output-socket-type") )
        _options->outputSocketType = vm["output-socket-type"].as<string>();

    if ( vm.count("output-buff-size") )
        _options->outputBufSize = vm["output-buff-size"].as<int>();

    if ( vm.count("output-method") )
        _options->outputMethod = vm["output-method"].as<string>();

    if ( vm.count("output-address") )
        _options->outputAddress = vm["output-address"].as<string>();

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
    flp.SetProperty(O2FLPex::EventSize, options.eventSize);

    flp.SetProperty(O2FLPex::NumInputs, 0);
    flp.SetProperty(O2FLPex::NumOutputs, 1);

    flp.ChangeState(O2FLPex::INIT);

    flp.SetProperty(O2FLPex::OutputSocketType, options.outputSocketType);
    flp.SetProperty(O2FLPex::OutputRcvBufSize, options.outputBufSize);
    flp.SetProperty(O2FLPex::OutputMethod, options.outputMethod);
    flp.SetProperty(O2FLPex::OutputAddress, options.outputAddress);

    flp.ChangeState(O2FLPex::SETOUTPUT);
    flp.ChangeState(O2FLPex::SETINPUT);
    flp.ChangeState(O2FLPex::RUN);

    // wait until the running thread has finished processing.
    boost::unique_lock<boost::mutex> lock(flp.fRunningMutex);
    while (!flp.fRunningFinished)
    {
      flp.fRunningCondition.wait(lock);
    }

    flp.ChangeState(O2FLPex::STOP);
    flp.ChangeState(O2FLPex::END);

    return 0;
}
