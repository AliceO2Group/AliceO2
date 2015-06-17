/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * RoundtripServer.cxx
 *
 * @since 2014-10-10
 * @author A. Rybalchenko
 */

#include "RoundtripServer.h"
#include "FairMQLogger.h"

using namespace std;

void RoundtripServer::Run()
{
    while (GetCurrentState() == RUNNING)
    {
        FairMQMessage* request = fTransportFactory->CreateMessage();
        fChannels["data"].at(0).Receive(request);
        delete request;

        void* buffer = operator new[](1);
        FairMQMessage* reply = fTransportFactory->CreateMessage(buffer, 1);
        fChannels["data"].at(0).Send(reply);
        delete reply;
    }
}
