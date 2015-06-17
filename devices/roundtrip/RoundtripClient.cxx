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

        fChannels["data"].at(0).Send(msg);
        fChannels["data"].at(0).Receive(reply);

        endTime = boost::posix_time::microsec_clock::local_time();

        averageTimes[size] = (endTime - startTime).total_microseconds();

        delete msg;
        delete reply;
    }

    for (auto it = averageTimes.begin(); it != averageTimes.end(); ++it)
    {
        std::cout << it->first << ": " << it->second << " microseconds" << std::endl;
    }
}


void RoundtripClient::SetProperty(const int key, const string& value)
{
    switch (key)
    {
        case Text:
            fText = value;
            break;
        default:
            FairMQDevice::SetProperty(key, value);
            break;
    }
}

string RoundtripClient::GetProperty(const int key, const string& default_ /*= ""*/)
{
    switch (key)
    {
        case Text:
            return fText;
            break;
        default:
            return FairMQDevice::GetProperty(key, default_);
    }
}

void RoundtripClient::SetProperty(const int key, const int value)
{
    switch (key)
    {
        default:
            FairMQDevice::SetProperty(key, value);
            break;
    }
}

int RoundtripClient::GetProperty(const int key, const int default_ /*= 0*/)
{
    switch (key)
    {
        default:
            return FairMQDevice::GetProperty(key, default_);
    }
}
