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
using namespace AliceO2::Header;

using NameHeader16 = NameHeader<16>; //header holding 16 characters

FairMQmonitor::FairMQmonitor()
{
}

bool FairMQmonitor::ConditionalRun()
{
  this_thread::sleep_for(chrono::seconds(1));

  FairMQParts parts;

  DataHeader dataHeader;
  dataHeader = gDataOriginAny;
  dataHeader = gDataDescriptionInfo;
  dataHeader = gSerializationMethodNone;

  NameHeader16 nameHeader;
  nameHeader = "some name";

  AddMessage(parts,{dataHeader},NewSimpleMessage("foo"));
  AddMessage(parts,{dataHeader,nameHeader},NewSimpleMessage("bar"));

  dataHeader = gDataOriginTPC;
  AddMessage(parts,{dataHeader},NewSimpleMessage("foo"));


  Send(parts, "data-out");

  return true;
}

bool FairMQmonitor::HandleData(FairMQParts& parts, int /*index*/)
{
  ForEach(parts, &FairMQmonitor::HandleBuffers);
  return true;
}

bool FairMQmonitor::HandleBuffers(byte* headerBuffer, size_t headerBufferSize,
                                  byte* dataBuffer,   size_t dataBufferSize)
{
  const DataHeader* dataHeader = get<DataHeader>(headerBuffer);
  if (!dataHeader) return false;

  if ( (*dataHeader)==gDataDescriptionTracks ) return true;
  if ( (*dataHeader)==gDataOriginTPC ) return true;

  const NameHeader16* nameHeader = get<NameHeader16>(headerBuffer);
  if (!nameHeader) return false;

  return true;
}

FairMQmonitor::~FairMQmonitor()
{
}
