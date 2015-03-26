//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
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

#include "EventSampler.h"
#include "FairMQLogger.h"
#include "FairMQPoller.h"
#include "AliHLTDataTypes.h"

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

using namespace std;

// time reference for the timestamp of events is the beginning of the day
using std::chrono::system_clock;
typedef std::chrono::duration<int,std::ratio<60*60*24> > PeriodDay;
const std::chrono::time_point<system_clock, PeriodDay> dayref=std::chrono::time_point_cast<PeriodDay>(system_clock::now());

using namespace ALICE::HLT;

EventSampler::EventSampler(int verbosity)
  : mEventPeriod(1000)
  , mInitialDelay(1000)
  , mNEvents(-1)
  , mPollingTimeout(10)
  , mSkipProcessing(0)
  , mVerbosity(verbosity)
  , mOutputFile()
{
}

EventSampler::~EventSampler()
{
}

void EventSampler::Init()
{
  /// inherited from FairMQDevice
  mNEvents=0;
  FairMQDevice::Init();
}

void EventSampler::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
  boost::thread samplerThread(boost::bind(&EventSampler::samplerLoop, this));

  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);

  bool received = false;

  // inherited variables of FairMQDevice:
  // fNumInputs
  // fTransportFactory
  // fPayloadInputs
  // fPayloadOutputs
  int NoOfMsgParts=fNumInputs-1;
  int errorCount=0;
  const int maxError=10;

  vector</*const*/ FairMQMessage*> inputMessages;
  vector<int> inputMessageCntPerSocket(fNumInputs, 0);
  int nReadCycles=0;

  std::ofstream latencyLog(mOutputFile);

  while (fState == RUNNING) {

    // read input messages
    poller->Poll(mPollingTimeout);
    for(int i = 0; i < fNumInputs; i++) {
      if (poller->CheckInput(i)) {
        int64_t more = 0;
        do {
          more = 0;
          unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
          received = fPayloadInputs->at(i)->Receive(msg.get());
          if (received) {
            inputMessages.push_back(msg.release());
            inputMessageCntPerSocket[i]++;
            if (mVerbosity > 3) {
              LOG(INFO) << " |---- receive Msg from socket " << i;
            }
            size_t more_size = sizeof(more);
            fPayloadInputs->at(i)->GetOption("rcv-more", &more, &more_size);
          }
        } while (more);
        if (mVerbosity > 2) {
          LOG(INFO) << "------ received " << inputMessageCntPerSocket[i] << " message(s) from socket " << i;
        }
      }
    }

    system_clock::time_point timestamp = system_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timestamp - dayref);
    auto useconds = std::chrono::duration_cast<std::chrono::microseconds>(timestamp  - dayref - seconds);

    for (vector<FairMQMessage*>::iterator mit=inputMessages.begin();
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

      delete *mit;
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

  delete poller;

  rateLogger.interrupt();
  rateLogger.join();
  samplerThread.interrupt();
  samplerThread.join();

  Shutdown();

  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void EventSampler::Pause()
{
  /// inherited from FairMQDevice

  // nothing to do
  FairMQDevice::Pause();
}

void EventSampler::Shutdown()
{
  /// inherited from FairMQDevice

  int iResult=0;
  // TODO: shutdown component and delete instance

  FairMQDevice::Shutdown();
}

void EventSampler::InitOutput()
{
  /// inherited from FairMQDevice

  FairMQDevice::InitOutput();
}

void EventSampler::InitInput()
{
  /// inherited from FairMQDevice

  FairMQDevice::InitInput();
}

void EventSampler::SetProperty(const int key, const string& value, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  switch (key) {
  case OutputFile:
    mOutputFile = value;
    return;
  }
  return FairMQDevice::SetProperty(key, value, slot);
}

string EventSampler::GetProperty(const int key, const string& default_, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  return FairMQDevice::GetProperty(key, default_, slot);
}

void EventSampler::SetProperty(const int key, const int value, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  switch (key) {
  case EventPeriod:
    mEventPeriod = value;
    return;
  case InitialDelay:
    mInitialDelay = value;
    return;
  case PollingTimeout:
    mPollingTimeout = value;
    return;
  case SkipProcessing:
    mSkipProcessing = value;
    return;
  }
  return FairMQDevice::SetProperty(key, value, slot);
}

int EventSampler::GetProperty(const int key, const int default_, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  switch (key) {
  case EventPeriod:
    return mEventPeriod;
  case InitialDelay:
    return mInitialDelay;
  case PollingTimeout:
    return mPollingTimeout;
  case SkipProcessing:
    return mSkipProcessing;
  }
  return FairMQDevice::GetProperty(key, default_, slot);
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

  LOG(INFO) << "starting sampler loop, period " << mEventPeriod << " us";
  while (fState == RUNNING) {
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

    for (int iOutput=0; iOutput<fNumOutputs; iOutput++) {
      fPayloadOutputs->at(iOutput)->Send(msg.get());
    }

    mNEvents++;
    if (eventPeriodInSeconds>0) sleep(eventPeriodInSeconds);
    else usleep(mEventPeriod);
  }
}
