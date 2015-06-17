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

    FairMQChannel channel("rep", "bind", "tcp://*:5005");
    channel.UpdateSndBufSize(10000);
    channel.UpdateRcvBufSize(10000);
    channel.UpdateRateLogging(1);
    server.fChannels["data"].push_back(channel);

    server.ChangeState("INIT_DEVICE");
    server.WaitForEndOfState("INIT_DEVICE");

    server.ChangeState("INIT_TASK");
    server.WaitForEndOfState("INIT_TASK");

    server.ChangeState("RUN");
    server.WaitForEndOfState("RUN");

    server.ChangeState("STOP");

    server.ChangeState("RESET_TASK");
    server.WaitForEndOfState("RESET_TASK");

    server.ChangeState("RESET_DEVICE");
    server.WaitForEndOfState("RESET_DEVICE");

    server.ChangeState("END");

    return 0;
}
