/********************************************************************************
 *    Copyright (C) 2014 Frankfurt Institute for Advanced Studies               *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * FairMQmonitor.cpp
 *
 * @since 2014-12-10
 * @author M.Krzewicki <mkrzewic@cern.ch>
 */

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "fairMQmonitor/FairMQmonitor.h"
#include "FairMQLogger.h"
#include "Headers/DataHeader.h"

using namespace std;
using namespace AliceO2;

FairMQmonitor::FairMQmonitor()
{
}

bool FairMQmonitor::ConditionalRun()
{
    this_thread::sleep_for(chrono::seconds(1));

    FairMQParts parts;

    // NewSimpleMessage creates a copy of the data and takes care of its destruction (after the transfer takes place).
    // Should only be used for small data because of the cost of an additional copy
    Base::DataHeader h;
    h = AliceO2::Base::gDataOriginAny;
    h = AliceO2::Base::gDataDescriptionInfo;
    h = AliceO2::Base::gSerializationMethodNone;

    //AddMessage(parts,O2Header(h),"some info");

    LOG(INFO) << "Sending body of size: " << parts.At(1)->GetSize();

    Send(parts, "data-out");

    return true;
}

FairMQmonitor::~FairMQmonitor()
{
}
