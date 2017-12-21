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

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "O2MessageMonitor/O2MessageMonitor.h"
#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>
#include "Headers/DataHeader.h"
#include "Headers/NameHeader.h"

using namespace std;
using namespace o2::header;
using namespace o2::Base;

using NameHeader48 = NameHeader<48>; //header holding 16 characters

//__________________________________________________________________________________________________
O2MessageMonitor::O2MessageMonitor()
  : mDataHeader()
  , mPayload("I am the info payload")
  , mName(R"(My name is "gDataDescriptionInfo")")
  , mDelay(1000)
  , mIterations(10)
  , mLimitOutputCharacters(1024)
{
  mDataHeader = gDataOriginAny;
  mDataHeader = gDataDescriptionInfo;
  mDataHeader = gSerializationMethodNone;
}

//__________________________________________________________________________________________________
void O2MessageMonitor::InitTask()
{
  mDelay = GetConfig()->GetValue<int>("sleep");
  mIterations = GetConfig()->GetValue<int>("n");
  mPayload = GetConfig()->GetValue<std::string>("payload");
  std::string tmp = GetConfig()->GetValue<std::string>("name");
  if (!tmp.empty()) mName = tmp;
  mLimitOutputCharacters = GetConfig()->GetValue<int>("limit");
}

//__________________________________________________________________________________________________
void O2MessageMonitor::Run()
{
  //check socket type of data channel
  std::string type;
  std::vector<FairMQChannel> subChannels = fChannels["data"];
  if (subChannels.size()>0) {
    type = subChannels[0].GetType();
  }

  while (CheckCurrentState(RUNNING) && (--mIterations)!=0) {
    this_thread::sleep_for(chrono::milliseconds(mDelay));

    O2Message message;
    NameHeader48 nameHeader;

    //maybe send a request
    nameHeader = mName;
    AddMessage(message,{mDataHeader,nameHeader},NewSimpleMessage(mPayload));
    Send(message, "data");
    message.fParts.clear();

    //message in;
    Receive(message, "data");
    LOG(INFO) << "== New message=============================";
    ForEach(message, &O2MessageMonitor::HandleO2frame);
    message.fParts.clear();

    //maybe a reply message
    if (type=="rep") {
      nameHeader = "My name is reply";
      AddMessage(message,{mDataHeader,nameHeader},NewSimpleMessage("I am a reply"));
      Send(message, "data");
    }
  }
}

//__________________________________________________________________________________________________
bool O2MessageMonitor::HandleO2frame(const byte* headerBuffer, size_t headerBufferSize,
    const byte* dataBuffer,   size_t dataBufferSize)
{

  hexDump("headerBuffer", headerBuffer, headerBufferSize);
  hexDump("dataBuffer", dataBuffer, dataBufferSize, mLimitOutputCharacters);

  const DataHeader* dataHeader = get<DataHeader>(headerBuffer);
  if (!dataHeader) { LOG(INFO) << "data header empty!"; return false; }
  if ( (*dataHeader)==gDataDescriptionInfo ) {}

  const NameHeader<0>* nameHeader = get<NameHeader<0>>(headerBuffer);
  if (nameHeader) {size_t sizeNameHeader=nameHeader->size();}

  return true;
}

