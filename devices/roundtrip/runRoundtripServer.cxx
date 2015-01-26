/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * runExampleServer.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <iostream>
#include <csignal>

#include "FairMQLogger.h"
#include "RoundtripServer.h"

#ifdef NANOMSG
#include "FairMQTransportFactoryNN.h"
#else
#include "FairMQTransportFactoryZMQ.h"
#endif

using namespace std;

RoundtripServer server;

static void s_signal_handler(int signal)
{
    cout << endl << "Caught signal " << signal << endl;

    server.ChangeState(RoundtripServer::STOP);
    server.ChangeState(RoundtripServer::END);

    cout << "Shutdown complete. Bye!" << endl;
    exit(1);
}

static void s_catch_signals(void)
{
    struct sigaction action;
    action.sa_handler = s_signal_handler;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
}

int main(int argc, char** argv)
{
    s_catch_signals();

    LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
    FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
    FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

    server.SetTransport(transportFactory);

    server.SetProperty(RoundtripServer::Id, "server");
    server.SetProperty(RoundtripServer::NumIoThreads, 1);
    server.SetProperty(RoundtripServer::NumInputs, 1);
    server.SetProperty(RoundtripServer::NumOutputs, 0);

    server.ChangeState(RoundtripServer::INIT);

    server.SetProperty(RoundtripServer::InputSocketType, "rep", 0);
    server.SetProperty(RoundtripServer::InputSndBufSize, 10000, 0);
    server.SetProperty(RoundtripServer::InputRcvBufSize, 10000, 0);
    server.SetProperty(RoundtripServer::InputMethod, "bind", 0);
    server.SetProperty(RoundtripServer::InputAddress, "tcp://*:5005", 0);

    server.ChangeState(RoundtripServer::SETOUTPUT);
    server.ChangeState(RoundtripServer::SETINPUT);
    server.ChangeState(RoundtripServer::BIND);
    server.ChangeState(RoundtripServer::CONNECT);

    LOG(INFO) << "Listening for requests!";

    server.ChangeState(RoundtripServer::RUN);

    // wait until the running thread has finished processing.
    boost::unique_lock<boost::mutex> lock(server.fRunningMutex);
    while (!server.fRunningFinished)
    {
        server.fRunningCondition.wait(lock);
    }

    server.ChangeState(RoundtripServer::STOP);
    server.ChangeState(RoundtripServer::END);

    return 0;
}
