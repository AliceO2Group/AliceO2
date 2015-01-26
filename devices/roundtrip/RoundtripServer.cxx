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
    while (fState == RUNNING)
    {
        FairMQMessage* request = fTransportFactory->CreateMessage();
        fPayloadInputs->at(0)->Receive(request);
        delete request;

        void* buffer = operator new[](1);
        FairMQMessage* reply = fTransportFactory->CreateMessage(buffer, 1);
        fPayloadInputs->at(0)->Send(reply);
        delete reply;
    }

    FairMQDevice::Shutdown();

    boost::lock_guard<boost::mutex> lock(fRunningMutex);
    fRunningFinished = true;
    fRunningCondition.notify_one();
}
