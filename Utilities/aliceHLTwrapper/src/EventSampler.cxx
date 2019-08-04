// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   EventSampler.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  Sampler device for Alice HLT events in FairRoot/ALFA

#include "aliceHLTwrapper/EventSampler.h"
#include "O2Device/Compatibility.h"
#include <FairMQLogger.h>
#include <FairMQPoller.h>
#include "aliceHLTwrapper/AliHLTDataTypes.h"
#include <options/FairMQProgOptions.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

using std::string;
using std::vector;
using std::unique_ptr;
using std::cerr;
using std::endl;

// time reference for the timestamp of events is the beginning of the day
using std::chrono::system_clock;
typedef std::chrono::duration<int,std::ratio<60*60*24> > PeriodDay;
const std::chrono::time_point<system_clock, PeriodDay> dayref=std::chrono::time_point_cast<PeriodDay>(system_clock::now());

using namespace o2::alice_hlt;

EventSampler::EventSampler(int verbosity)
  : mEventPeriod(-1)
  , mInitialDelay(-1)
  , mNEvents(-1)
  , mPollingTimeout(-1)
  , mSkipProcessing(0)
  , mVerbosity(verbosity)
  , mLatencyLogFileName()
{
}

EventSampler::~EventSampler()
= default;

constexpr const char* EventSampler::OptionKeys[];

bpo::options_description EventSampler::GetOptionsDescription()
{
  // assemble the options for the device class and component
  bpo::options_description od("EventSampler options");
  // clang-format off
  od.add_options()
    (OptionKeys[OptionKeyEventPeriod],
     bpo::value<int>()->default_value(1000),
     "event period in us")
    (OptionKeys[OptionKeyInitialDelay],
     bpo::value<int>()->default_value(1000),
     "initial delay before the first event in ms")
    (OptionKeys[OptionKeyPollTimeout],
     bpo::value<int>()->default_value(10),
     "timeout for the input socket poller in ms")
    ((std::string(OptionKeys[OptionKeyLatencyLog]) + ",l").c_str(),
     bpo::value<std::string>()->default_value(""),
     "output file name for logging of latency")
    ((std::string(OptionKeys[OptionKeyDryRun]) + ",n").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "skip component processing");
  // clang-format on
  return od;
}

void EventSampler::InitTask()
{
  /// inherited from FairMQDevice
  const auto * config = GetConfig();
  if (config) {
    mEventPeriod = config->GetValue<int>(OptionKeys[OptionKeyEventPeriod]);
    mInitialDelay =  config->GetValue<int>(OptionKeys[OptionKeyInitialDelay]);
    mPollingTimeout = config->GetValue<int>(OptionKeys[OptionKeyPollTimeout]);
    mSkipProcessing = config->GetValue<bool>(OptionKeys[OptionKeyDryRun]);
    mLatencyLogFileName = config->GetValue<std::string>(OptionKeys[OptionKeyLatencyLog]);
  }
  mNEvents=0;
}

void EventSampler::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

  boost::thread samplerThread(boost::bind(&EventSampler::samplerLoop, this));

  unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels["data-in"]));

  bool received = false;

  // inherited variables of FairMQDevice:
  // fChannels
  // fTransportFactory
  int numInputs = fChannels["data-in"].size();
  int NoOfMsgParts=numInputs-1;
  int errorCount=0;
  const int maxError=10;

  vector<unique_ptr<FairMQMessage>> inputMessages;
  vector<int> inputMessageCntPerSocket(numInputs, 0);
  int nReadCycles=0;

  std::ofstream latencyLog(mLatencyLogFileName);

  while (compatibility::FairMQ13<FairMQDevice>::IsRunning(this)) {

    // read input messages
    poller->Poll(mPollingTimeout);
    for(int i = 0; i < numInputs; i++) {
      if (poller->CheckInput(i)) {
        if (fChannels.at("data-in").at(i).Receive(inputMessages)) {
          inputMessageCntPerSocket[i] = inputMessages.size();
          if (mVerbosity > 3) {
            LOG(INFO) << " |---- receive Msgs from socket " << i;
          }
        }
        if (mVerbosity > 2) {
          LOG(INFO) << "------ received " << inputMessageCntPerSocket[i] << " message(s) from socket " << i;
        }
      }
    }

    system_clock::time_point timestamp = system_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timestamp - dayref);
    auto useconds = std::chrono::duration_cast<std::chrono::microseconds>(timestamp  - dayref - seconds);

    for (vector<unique_ptr<FairMQMessage>>::iterator mit=inputMessages.begin();
         mit!=inputMessages.end(); mit++) {
      AliHLTComponentEventData* evtData=reinterpret_cast<AliHLTComponentEventData*>((*mit)->GetData());
      if ((*mit)->GetSize() >= sizeof(AliHLTComponentEventData) &&
          evtData && evtData->fStructSize == sizeof(AliHLTComponentEventData) &&
          (evtData->fEventCreation_s>0 || evtData->fEventCreation_us>0)) {
        unsigned latencySeconds=seconds.count() - evtData->fEventCreation_s;
        unsigned latencyUSeconds=0;
        if (useconds.count() < evtData->fEventCreation_us) {
          latencySeconds--;
          latencyUSeconds=(1000000 + useconds.count()) - evtData->fEventCreation_us;
        } else {
          latencyUSeconds=useconds.count() - evtData->fEventCreation_us;
        }
        if (mVerbosity>0) {
          const char* unit="";
          unsigned value=0;
          if (latencySeconds>=10) {
            unit=" s";
            value=latencySeconds;
          } else if (latencySeconds>0 || latencyUSeconds>10000) {
            value=latencySeconds*1000 + latencyUSeconds/1000;
            unit=" ms";
          } else {
            value=latencyUSeconds;
            unit=" us";
          }
          LOG(DEBUG) << "received event " << evtData->fEventID << " at " << seconds.count() << "s  " << useconds.count() << "us - latency " << value << unit;
        }
        latencyUSeconds+=latencySeconds*1000000; // max 4294s, should be enough for latency
        if (latencyLog.is_open()) {
          latencyLog << evtData->fEventID << " " << latencyUSeconds << endl;
        }
      }
    }
    inputMessages.clear();
    for (vector<int>::iterator mcit=inputMessageCntPerSocket.begin();
         mcit!=inputMessageCntPerSocket.end(); mcit++) {
      *mcit=0;
    }
  }

  if (latencyLog.is_open()) {
    latencyLog.close();
  }

  samplerThread.interrupt();
  samplerThread.join();
}

void EventSampler::samplerLoop()
{
  /// sampler loop
  LOG(INFO) << "initializing sampler loop, then waiting for " << mInitialDelay << " ms";
  // wait until the first event is sent
  // usleep expects arguments in the range [0,1000000]
  unsigned initialDelayInSeconds=mInitialDelay/1000;
  unsigned initialDelayInUSeconds=mInitialDelay%1000;
  unsigned eventPeriodInSeconds=mEventPeriod/1000000;
  if (initialDelayInSeconds>0) sleep(initialDelayInSeconds);
  usleep(initialDelayInUSeconds);

  unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
  msg->Rebuild(sizeof(AliHLTComponentEventData));
  if (msg->GetSize() < sizeof(AliHLTComponentEventData)) {
    // fatal error
    LOG(ERROR) << "failed to allocate message of size " << sizeof(AliHLTComponentEventData) << ", aborting event generation";
    return;
  }
  AliHLTComponentEventData* evtData = reinterpret_cast<AliHLTComponentEventData*>(msg->GetData());
  memset(evtData, 0, sizeof(AliHLTComponentEventData));
  evtData->fStructSize = sizeof(AliHLTComponentEventData);

  int numOutputs = (fChannels.find("data-out") == fChannels.end() ? 0 : fChannels["data-out"].size());

  LOG(INFO) << "starting sampler loop, period " << mEventPeriod << " us";
  while (compatibility::FairMQ13<FairMQDevice>::IsRunning(this)) {
    msg->Rebuild(sizeof(AliHLTComponentEventData));
    evtData = reinterpret_cast<AliHLTComponentEventData*>(msg->GetData());
    memset(evtData, 0, sizeof(AliHLTComponentEventData));
    evtData->fStructSize = sizeof(AliHLTComponentEventData);
    evtData->fEventID=mNEvents;
    system_clock::time_point timestamp = system_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timestamp - dayref);
    evtData->fEventCreation_s=seconds.count();
    auto useconds = std::chrono::duration_cast<std::chrono::microseconds>(timestamp  - dayref - seconds);
    evtData->fEventCreation_us=useconds.count();
    if (mVerbosity>0) {
      LOG(DEBUG) << "send     event " << evtData->fEventID << " at " << evtData->fEventCreation_s << "s  " << evtData->fEventCreation_us << "us";
    }

    for (int iOutput=0; iOutput<numOutputs; iOutput++) {
      fChannels["data-out"].at(iOutput).Send(msg);
    }

    mNEvents++;
    if (eventPeriodInSeconds>0) sleep(eventPeriodInSeconds);
    else usleep(mEventPeriod);
  }
}
