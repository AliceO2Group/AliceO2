/**
 * runMerger.cxx
 *
 * @since 2012-12-06
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <iostream>
#include <csignal>

#include "boost/program_options.hpp"

#include "FairMQLogger.h"
#include "O2Merger.h"

#ifdef NANOMSG
    #include "FairMQTransportFactoryNN.h"
#else
    #include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;

O2Merger merger;

static void s_signal_handler (int signal)
{
    cout << endl << "Caught signal " << signal << endl;

    merger.ChangeState(O2Merger::STOP);
    merger.ChangeState(O2Merger::END);

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
    vector<string> inputSocketType;
    vector<int> inputBufSize;
    vector<string> inputMethod;
    vector<string> inputAddress;
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
        ("io-threads", bpo::value<int>()->default_value(1), "Number of I/O threads")
        ("num-inputs", bpo::value<int>()->required(), "Number of input sockets")
        ("input-socket-type", bpo::value< vector<string> >()->required(), "Input socket type: sub/pull")
        ("input-buff-size", bpo::value< vector<int> >()->required(), "Input buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
        ("input-method", bpo::value< vector<string> >()->required(), "Input method: bind/connect")
        ("input-address", bpo::value< vector<string> >()->required(), "Input address, e.g.: \"tcp://localhost:5555\"")
        ("output-socket-type", bpo::value<string>()->required(), "Output socket type: pub/push")
        ("output-buff-size", bpo::value<int>()->required(), "Output buffer size in number of messages (ZeroMQ)/bytes(nanomsg)")
        ("output-method", bpo::value<string>()->required(), "Output method: bind/connect")
        ("output-address", bpo::value<string>()->required(), "Output address, e.g.: \"tcp://localhost:5555\"")
        ("help", "Print help messages");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(_argc, _argv, desc), vm);

    if ( vm.count("help") )
    {
        LOG(INFO) << "MERGER" << endl << desc;
        return false;
    }

    bpo::notify(vm);

    if ( vm.count("id") )
        _options->id = vm["id"].as<string>();

    if ( vm.count("io-threads") )
        _options->ioThreads = vm["io-threads"].as<int>();

    if ( vm.count("num-inputs") )
        _options->numInputs = vm["num-inputs"].as<int>();

    if ( vm.count("input-socket-type") )
        _options->inputSocketType = vm["input-socket-type"].as< vector<string> >();

    if ( vm.count("input-buff-size") )
        _options->inputBufSize = vm["input-buff-size"].as< vector<int> >();

    if ( vm.count("input-method") )
        _options->inputMethod = vm["input-method"].as< vector<string> >();

    if ( vm.count("input-address") )
        _options->inputAddress = vm["input-address"].as< vector<string> >();

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

    merger.SetTransport(transportFactory);

    merger.SetProperty(O2Merger::Id, options.id);
    merger.SetProperty(O2Merger::NumIoThreads, options.ioThreads);

    merger.SetProperty(O2Merger::NumInputs, options.numInputs);
    merger.SetProperty(O2Merger::NumOutputs, 1);

    merger.ChangeState(O2Merger::INIT);

    for (int i = 0; i < options.numInputs; ++i)
    {
        merger.SetProperty(O2Merger::InputSocketType, options.inputSocketType.at(i), i);
        merger.SetProperty(O2Merger::InputSndBufSize, options.inputBufSize.at(i), i);
        merger.SetProperty(O2Merger::InputMethod, options.inputMethod.at(i), i);
        merger.SetProperty(O2Merger::InputAddress, options.inputAddress.at(i), i);
    }

    merger.SetProperty(O2Merger::OutputSocketType, options.outputSocketType);
    merger.SetProperty(O2Merger::OutputRcvBufSize, options.outputBufSize);
    merger.SetProperty(O2Merger::OutputMethod, options.outputMethod);
    merger.SetProperty(O2Merger::OutputAddress, options.outputAddress);

    merger.ChangeState(O2Merger::SETOUTPUT);
    merger.ChangeState(O2Merger::SETINPUT);
    merger.ChangeState(O2Merger::RUN);

    // wait until the running thread has finished processing.
    boost::unique_lock<boost::mutex> lock(merger.fRunningMutex);
    while (!merger.fRunningFinished)
    {
      merger.fRunningCondition.wait(lock);
    }

    merger.ChangeState(O2Merger::STOP);
    merger.ChangeState(O2Merger::END);

    return 0;
}

