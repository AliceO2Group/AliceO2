/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * RoundtripClient.cpp
 *
 * @since 2014-10-10
 * @author A. Rybalchenko
 */

#include <map>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "RoundtripClient.h"
#include "FairMQLogger.h"

using namespace std;
using boost::posix_time::ptime;

RoundtripClient::RoundtripClient()
    : fText()
{
}

void RoundtripClient::Run()
{
    ptime startTime;
    ptime endTime;

    map<int,int> averageTimes;

    int size = 0;

    while (size < 20000000)
    {
        size += 1000000;
        void* buffer = operator new[](size);
        FairMQMessage* msg = fTransportFactory->CreateMessage(buffer, size);
        FairMQMessage* reply = fTransportFactory->CreateMessage();

        startTime = boost::posix_time::microsec_clock::local_time();

        fPayloadOutputs->at(0)->Send(msg);
        fPayloadOutputs->at(0)->Receive(reply);

        endTime = boost::posix_time::microsec_clock::local_time();

        averageTimes[size] = (endTime - startTime).total_microseconds();

        delete msg;
        delete reply;
    }

    for (auto it = averageTimes.begin(); it != averageTimes.end(); ++it)
    {
        std::cout << it->first << ": " << it->second << " microseconds" << std::endl;
    }

    FairMQDevice::Shutdown();

    boost::lock_guard<boost::mutex> lock(fRunningMutex);
    fRunningFinished = true;
    fRunningCondition.notify_one();
}


void RoundtripClient::SetProperty(const int key, const string& value, const int slot /*= 0*/)
{
    switch (key)
    {
        case Text:
            fText = value;
            break;
        default:
            FairMQDevice::SetProperty(key, value, slot);
            break;
    }
}

string RoundtripClient::GetProperty(const int key, const string& default_ /*= ""*/, const int slot /*= 0*/)
{
    switch (key)
    {
        case Text:
            return fText;
            break;
        default:
            return FairMQDevice::GetProperty(key, default_, slot);
    }
}

void RoundtripClient::SetProperty(const int key, const int value, const int slot /*= 0*/)
{
    switch (key)
    {
        default:
            FairMQDevice::SetProperty(key, value, slot);
            break;
    }
}

int RoundtripClient::GetProperty(const int key, const int default_ /*= 0*/, const int slot /*= 0*/)
{
    switch (key)
    {
        default:
            return FairMQDevice::GetProperty(key, default_, slot);
    }
}
