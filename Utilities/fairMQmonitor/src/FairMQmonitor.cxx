/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * FairMQmonitor.cpp
 *
 * @since 2014-10-10
 * @author M.Krzewicki <mkrzewic@cern.ch>
 */

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "fairMQmonitor/FairMQmonitor.h"
#include "FairMQLogger.h"

using namespace std;

FairMQmonitor::FairMQmonitor()
    : fCounter(0)
{
}

bool FairMQmonitor::ConditionalRun()
{
    // Wait a second to keep the output readable.
    this_thread::sleep_for(chrono::seconds(1));

    FairMQParts parts;

    // NewSimpleMessage creates a copy of the data and takes care of its destruction (after the transfer takes place).
    // Should only be used for small data because of the cost of an additional copy
    parts.AddPart(NewMessage(100));
    parts.AddPart(NewMessage(1000));

    LOG(INFO) << "Sending body of size: " << parts.At(1)->GetSize();

    Send(parts, "data-out");

    // Go out of the sending loop if the stopFlag was sent.
    if (fCounter++ == 5)
    {
        return false;
    }

    return true;
}

FairMQmonitor::~FairMQmonitor()
{
}
