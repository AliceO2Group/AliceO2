// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @file O2MessageMonitor.cxx
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include <chrono>
#include <thread> // this_thread::sleep_for

#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>
#include "O2MessageMonitor/O2MessageMonitor.h"
#include "O2Device/Compatibility.h"

using namespace std;
using namespace o2::header;
using namespace o2::base;
using namespace o2::compatibility;

//__________________________________________________________________________________________________
void O2MessageMonitor::InitTask()
{
  mDelay = GetConfig()->GetValue<int>("sleep");
  mIterations = GetConfig()->GetValue<int>("n");
  mPayload = GetConfig()->GetValue<std::string>("payload");
  std::string tmp = GetConfig()->GetValue<std::string>("name");
  if (!tmp.empty())
    mName = tmp;
  mLimitOutputCharacters = GetConfig()->GetValue<int>("limit");
}

//__________________________________________________________________________________________________
void O2MessageMonitor::Run()
{
  // check socket type of data channel
  std::string type;
  std::vector<FairMQChannel>& subChannels = fChannels["data"];
  if (subChannels.size() > 0) {
    type = subChannels[0].GetType();
  }

  auto dataResource = o2::pmr::getTransportAllocator(subChannels[0].Transport());

  while (FairMQ13<FairMQDevice>::IsRunning(this) && (--mIterations) != 0) {
    this_thread::sleep_for(chrono::milliseconds(mDelay));

    O2Message message;

    // maybe send a request
    if (type == "req") {
      addDataBlock(message, {dataResource, DataHeader{gDataDescriptionInfo, gDataOriginAny, DataHeader::SubSpecificationType{0}}},
                   NewSimpleMessageFor("data", 0, mPayload));
      Send(message, "data");
      message.fParts.clear();
    }

    // message in;
    Receive(message, "data");
    LOG(INFO) << "== New message=============================";
    o2::base::forEach(message, [&](auto header, auto data) {
      hexDump("headerBuffer", header.data(), header.size());
      hexDump("dataBuffer", data.data(), data.size(), mLimitOutputCharacters);
    });
    message.fParts.clear();

    // maybe a reply message
    if (type == "rep") {
      o2::base::addDataBlock(message,
                             {dataResource, DataHeader{gDataDescriptionInfo, gDataOriginAny, DataHeader::SubSpecificationType{0}}},
                             NewSimpleMessageFor("data", 0, ""));
      Send(message, "data");
    }
  }
}
