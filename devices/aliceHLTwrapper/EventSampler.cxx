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

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
using namespace ALICE::HLT;

// the chrono lib needs C++11
#if __cplusplus < 201103L
#warning statistics measurement for EventSampler disabled: need C++11 standard
#else
#define USE_CHRONO
#endif
#ifdef USE_CHRONO
#include <chrono>
using std::chrono::system_clock;
typedef std::chrono::milliseconds TimeScale;
#endif // USE_CHRONO

EventSampler::EventSampler(int verbosity)
  : mEventRate(1000)
  , mNEvents(-1)
  , mPollingTimeout(10)
  , mSkipProcessing(0)
  , mVerbosity(verbosity)
{
}

EventSampler::~EventSampler()
{
}

void EventSampler::Init()
{
  /// inherited from FairMQDevice
  FairMQDevice::Init();
}

void EventSampler::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

#ifdef USE_CHRONO
  static system_clock::time_point refTime = system_clock::now();
#endif // USE_CHRONO
  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

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
  while (fState == RUNNING) {

    // read input messages
    poller->Poll(mPollingTimeout);
    int inputsReceived=0;
    bool receivedAtLeastOneMessage=false;
    for(int i = 0; i < fNumInputs; i++) {
      if (inputMessageCntPerSocket[i]>0) {
	inputsReceived++;
	continue;
      }
      received = false;
      if (poller->CheckInput(i)) {
        int64_t more = 0;
        do {
          more = 0;
          unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
          received = fPayloadInputs->at(i)->Receive(msg.get());
          if (received) {
            receivedAtLeastOneMessage = true;
            inputMessages.push_back(msg.release());
            if (inputMessageCntPerSocket[i] == 0)
              inputsReceived++; // count only the first message on that socket
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
    if (receivedAtLeastOneMessage) nReadCycles++;
    if (inputsReceived<fNumInputs) {
      continue;
    }
#ifdef USE_CHRONO
    auto duration = std::chrono::duration_cast<TimeScale>(std::chrono::system_clock::now() - refTime);
#endif //USE_CHRONO

    // if (!mSkipProcessing) {
    //   // prepare input from messages
    //   vector<AliceO2::AliceHLT::MessageFormat::BufferDesc_t> dataArray;
    //   for (vector</*const*/ FairMQMessage*>::iterator msg=inputMessages.begin();
    // 	   msg!=inputMessages.end(); msg++) {
    // 	void* buffer=(*msg)->GetData();
    // 	dataArray.push_back(AliceO2::AliceHLT::MessageFormat::BufferDesc_t(reinterpret_cast<unsigned char*>(buffer), (*msg)->GetSize()));
    //   }

    //   // call the component
    //   if ((iResult=mComponent->process(dataArray))<0) {
    // 	LOG(ERROR) << "component processing failed with error code " << iResult;
    //   }

    //   // build messages from output data
    //   if (dataArray.size() > 0) {
    //     if (mVerbosity > 2) {
    //       LOG(INFO) << "processing " << dataArray.size() << " buffer(s)";
    //     }
    //     unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
    //     if (msg.get() && fPayloadOutputs != NULL && fPayloadOutputs->size() > 0) {
    //       vector<AliceO2::AliceHLT::MessageFormat::BufferDesc_t>::iterator data = dataArray.begin();
    //       while (data != dataArray.end()) {
    //         if (mVerbosity > 2) {
    //           LOG(INFO) << "sending message of size " << data->mSize;
    //         }
    //         msg->Rebuild(data->mSize);
    //         if (msg->GetSize() < data->mSize) {
    //           iResult = -ENOSPC;
    //           break;
    //         }
    //         AliHLTUInt8_t* pTarget = reinterpret_cast<AliHLTUInt8_t*>(msg->GetData());
    //         memcpy(pTarget, data->mP, data->mSize);
    //         if (data + 1 == dataArray.end()) {
    //           // this is the last data block
    //           fPayloadOutputs->at(0)->Send(msg.get());
    //         } else {
    //           fPayloadOutputs->at(0)->Send(msg.get(), "snd-more");
    //         }

    //         data = dataArray.erase(data);
    //       }
    //     } else if (fPayloadOutputs == NULL || fPayloadOutputs->size() == 0) {
    //       if (errorCount == maxError && errorCount++ > 0)
    //         LOG(ERROR) << "persistent error, suppressing further output";
    //       else if (errorCount++ < maxError)
    //         LOG(ERROR) << "no output slot available (" << (fPayloadOutputs == NULL ? "uninitialized" : "0 slots")
    //                    << ")";
    //     } else {
    //       if (errorCount == maxError && errorCount++ > 0)
    //         LOG(ERROR) << "persistent error, suppressing further output";
    //       else if (errorCount++ < maxError)
    //         LOG(ERROR) << "can not get output message from framework";
    //       iResult = -ENOMSG;
    //     }
    //   }
    // }

    // cleanup
    for (vector<FairMQMessage*>::iterator mit=inputMessages.begin();
	 mit!=inputMessages.end(); mit++) {
      delete *mit;
    }
    inputMessages.clear();
    for (vector<int>::iterator mcit=inputMessageCntPerSocket.begin();
	 mcit!=inputMessageCntPerSocket.end(); mcit++) {
      *mcit=0;
    }
  }

  delete poller;

  rateLogger.interrupt();
  rateLogger.join();

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
  case EventRate:
    mEventRate = value;
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
  case EventRate:
    return mEventRate;
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
  while (1) {
  }
}
