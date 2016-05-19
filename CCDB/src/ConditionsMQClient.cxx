/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * ConditionsMQClient.cpp
 *
 * @since 2016-01-11
 * @author R. Grosso, from examples/MQ/7-parameters/FairMQExample7Client.cxx
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "CCDB/ConditionsMQClient.h"

#include "TMessage.h"
#include "Rtypes.h"

#include "CCDB/Condition.h"

using namespace AliceO2::CDB;
using namespace std;

ConditionsMQClient::ConditionsMQClient()
  : fRunId(0),
    fParameterName()
{
}

ConditionsMQClient::~ConditionsMQClient()
{
}

void ConditionsMQClient::CustomCleanup(void *data, void *hint)
{
    delete (string*)hint;
}

// special class to expose protected TMessage constructor
class WrapTMessage : public TMessage
{
  public:
    WrapTMessage(void* buf, Int_t len)
        : TMessage(buf, len)
    {
        ResetBit(kIsOwner);
    }
};

void ConditionsMQClient::Run()
{
    int runId = 2000;

    while (CheckCurrentState(RUNNING))
    {
        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));

        string* reqStr = new string(fParameterName + "," + to_string(runId));

        LOG(INFO) << "Requesting parameter \"" << fParameterName << "\" for Run ID " << runId << ".";

        unique_ptr<FairMQMessage> req(fTransportFactory->CreateMessage(const_cast<char*>(reqStr->c_str()), reqStr->length(), CustomCleanup, reqStr));
        unique_ptr<FairMQMessage> rep(fTransportFactory->CreateMessage());

        if (fChannels.at("data").at(0).Send(req) > 0)
        {
            if (fChannels.at("data").at(0).Receive(rep) > 0)
            {
                WrapTMessage tmsg(rep->GetData(), rep->GetSize());
                Condition* aCondition = (Condition*)(tmsg.ReadObject(tmsg.GetClass()));
                LOG(INFO) << "Received a condition from the server:";
                aCondition->printConditionMetaData();
            }
        }

        runId++;
        if (runId == 2101)
        {
            runId = 2001;
        }
    }
}


void ConditionsMQClient::SetProperty(const int key, const string& value)
{
    switch (key)
    {
        case ParameterName:
            fParameterName = value;
            break;
        default:
            FairMQDevice::SetProperty(key, value);
            break;
    }
}

string ConditionsMQClient::GetProperty(const int key, const string& default_ /*= ""*/)
{
    switch (key)
    {
        case ParameterName:
            return fParameterName;
            break;
        default:
            return FairMQDevice::GetProperty(key, default_);
    }
}

void ConditionsMQClient::SetProperty(const int key, const int value)
{
    switch (key)
    {
        default:
            FairMQDevice::SetProperty(key, value);
            break;
    }
}

int ConditionsMQClient::GetProperty(const int key, const int default_ /*= 0*/)
{
    switch (key)
    {
        default:
            return FairMQDevice::GetProperty(key, default_);
    }
}
