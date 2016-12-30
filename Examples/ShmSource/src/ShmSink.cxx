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

/// @file ShmSink.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include <thread> // this_thread::sleep_for
#include <chrono>
#include <cassert>

#include "ShmSource/ShmSink.h"
#include "Headers/DataHeader.h"
#include "FairMQLogger.h"
#include "FairMQProgOptions.h"
#include <random>

using namespace std;

//_______________________________________________________________________________________
ShmSink::ShmSink() : mCounter{ 0 }
{
  // register callback for channel "data-in"
  OnData("data-in", &ShmSink::HandleData);
}

//_______________________________________________________________________________________
ShmSink::~ShmSink()
{
  LOG(INFO) << "exiting ShmSink";
}

//_______________________________________________________________________________________
void ShmSink::InitTask()
{
  mCounter = fConfig->GetValue<int>("n");
}

//_______________________________________________________________________________________
bool ShmSink::HandleData(FairMQParts& message, int /*index*/)
{
  ForEach(message, &ShmSink::HandleO2frame);
  return true;
}

//_______________________________________________________________________________________
bool ShmSink::HandleO2frame(const byte* headerBuffer, size_t headerBufferSize, const byte* dataBuffer,
                            size_t dataBufferSize)
{
  auto dataHeader = AliceO2::Header::get<AliceO2::Header::DataHeader>(headerBuffer);
  auto shmHeader = AliceO2::Header::get<AliceO2::Header::BoostShmHeader>(headerBuffer);

  if (!dataHeader)
    LOG(WARN) << "no data header\n";

  // in this example we have put the shared memory message id in the payload
  // crap out if something is not right
  if (shmHeader) {
    assert((*reinterpret_cast<const AliceO2::SharedMemory::IdType*>(dataBuffer) == shmHeader->id));
  }

  return true;
}
