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

/// @file FairMQmonitor.cxx
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "fairMQmonitor/FairMQmonitor.h"
#include "FairMQProgOptions.h"
#include "FairMQLogger.h"
#include "Headers/DataHeader.h"

using namespace std;
using namespace AliceO2::Header;
using namespace AliceO2::Base;

using NameHeader48 = NameHeader<48>; //header holding 16 characters

//__________________________________________________________________________________________________
FairMQmonitor::FairMQmonitor()
  : mDataHeader()
  , mPayload("I am the info payload")
  , mName("My name is \"gDataDescriptionInfo\"")
  , mDelay(1000)
  , mIterations(10)
  , mLimitOutputCharacters(1024)
{
  mDataHeader = gDataOriginAny;
  mDataHeader = gDataDescriptionInfo;
  mDataHeader = gSerializationMethodNone;
}

//__________________________________________________________________________________________________
void FairMQmonitor::InitTask()
{
  mDelay = fConfig->GetValue<int>("sleep");
  mIterations = fConfig->GetValue<int>("n");
  mPayload = fConfig->GetValue<std::string>("payload");
  std::string tmp = fConfig->GetValue<std::string>("name");
  if (!tmp.empty()) mName = tmp;
  mLimitOutputCharacters = fConfig->GetValue<int>("limit");
}

//__________________________________________________________________________________________________
void FairMQmonitor::Run()
{
  //check socket type of data channel
  std::string type;
  std::vector<FairMQChannel> subChannels = fChannels["data"];
  if (subChannels.size()>0) {
    type = subChannels[0].GetType();
  }

  while (CheckCurrentState(RUNNING) && (--mIterations)!=0) {
    this_thread::sleep_for(chrono::milliseconds(mDelay));

    O2message message;
    NameHeader48 nameHeader;

    //maybe send a request
    nameHeader = mName;
    AddMessage(message,{mDataHeader,nameHeader},NewSimpleMessage(mPayload));
    Send(message, "data");
    message.fParts.clear();

    //message in;
    Receive(message, "data");
    LOG(INFO) << "== New message=============================";
    ForEach(message, &FairMQmonitor::HandleO2frame);
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
bool FairMQmonitor::HandleO2frame(const byte* headerBuffer, size_t headerBufferSize,
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

