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

//  @file   WrapperDevice.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include "aliceHLTwrapper/WrapperDevice.h"
#include "aliceHLTwrapper/Component.h"
#include <FairMQLogger.h>
#include <FairMQPoller.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>

using std::string;
using std::vector;
using std::unique_ptr;
using namespace ALICE::HLT;

// the chrono lib needs C++11
#if __cplusplus < 201103L
#warning statistics measurement for WrapperDevice disabled: need C++11 standard
#else
#define USE_CHRONO
#endif
#ifdef USE_CHRONO
#include <chrono>
using std::chrono::system_clock;
using TimeScale = std::chrono::milliseconds;
#endif // USE_CHRONO

WrapperDevice::WrapperDevice(int argc, char** argv, int verbosity)
  : mComponent(nullptr)
  , mArgv()
  , mMessages()
  , mPollingPeriod(10)
  , mSkipProcessing(0)
  , mLastCalcTime(-1)
  , mLastSampleTime(-1)
  , mMinTimeBetweenSample(-1)
  , mMaxTimeBetweenSample(-1)
  , mTotalReadCycles(-1)
  , mMaxReadCycles(-1)
  , mNSamples(-1)
  , mVerbosity(verbosity)
{
  mArgv.insert(mArgv.end(), argv, argv+argc);
}

WrapperDevice::~WrapperDevice()
= default;

void WrapperDevice::Init()
{
}

void WrapperDevice::InitTask()
{
  /// inherited from FairMQDevice

  int iResult=0;
  std::unique_ptr<Component> component(new ALICE::HLT::Component);
  if (!component.get()) return /*-ENOMEM*/;

  string idkey="--instance-id";
  string id="";
  id=GetProperty(FairMQDevice::Id, id);
  vector<char*> argv;
  argv.push_back(mArgv[0]);
  argv.push_back(&idkey[0]);
  argv.push_back(&id[0]);
  if (mArgv.size()>1)
    argv.insert(argv.end(), mArgv.begin()+1, mArgv.end());
  if ((iResult=component->init(argv.size(), &argv[0]))<0) {
    LOG(ERROR) << "component init failed with error code " << iResult;
    throw std::runtime_error("component init failed");
    return /*iResult*/;
  }

  mComponent=component.release();
  mLastCalcTime=-1;
  mLastSampleTime=-1;
  mMinTimeBetweenSample=-1;
  mMaxTimeBetweenSample=-1;
  mTotalReadCycles=0;
  mMaxReadCycles=-1;
  mNSamples=0;
}

void WrapperDevice::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

#ifdef USE_CHRONO
  static system_clock::time_point refTime = system_clock::now();
#endif // USE_CHRONO

  unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels["data-in"]));

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
  while (CheckCurrentState(RUNNING)) {

    // read input messages
    poller->Poll(mPollingPeriod);
    int inputsReceived=0;
    bool receivedAtLeastOneMessage=false;
    for(int i = 0; i < numInputs; i++) {
      if (inputMessageCntPerSocket[i]>0) {
        inputsReceived++;
        continue;
      }
      if (poller->CheckInput(i)) {
        if (fChannels.at("data-in").at(i).Receive(inputMessages)) {
          receivedAtLeastOneMessage = true;
          inputsReceived++; // count only the first message on that socket
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
    if (receivedAtLeastOneMessage) nReadCycles++;
    if (inputsReceived<numInputs) {
      continue;
    }
    mNSamples++;
    mTotalReadCycles+=nReadCycles;
    if (mMaxReadCycles<0 || mMaxReadCycles<nReadCycles)
      mMaxReadCycles=nReadCycles;
    // if (nReadCycles>1) {
    //   LOG(INFO) << "------ recieved complete Msg from " << numInputs << " input(s) after " << nReadCycles << " read cycles" ;
    // }
    nReadCycles=0;
#ifdef USE_CHRONO
    auto duration = std::chrono::duration_cast<TimeScale>(std::chrono::system_clock::now() - refTime);

    if (mLastSampleTime>=0) {
      int sampleTimeDiff=duration.count()-mLastSampleTime;
      if (mMinTimeBetweenSample < 0 || sampleTimeDiff<mMinTimeBetweenSample)
        mMinTimeBetweenSample=sampleTimeDiff;
      if (mMaxTimeBetweenSample < 0 || sampleTimeDiff>mMaxTimeBetweenSample)
        mMaxTimeBetweenSample=sampleTimeDiff;
    }
    mLastSampleTime=duration.count();
    if (duration.count()-mLastCalcTime>1000) {
      LOG(INFO) << "------ processed  " << mNSamples << " sample(s) - total " 
                << mComponent->getEventCount() << " sample(s)";
      if (mNSamples > 0) {
        LOG(INFO) << "------ min  " << mMinTimeBetweenSample << "ms, max " << mMaxTimeBetweenSample << "ms avrg "
                  << (duration.count() - mLastCalcTime) / mNSamples << "ms ";
        LOG(INFO) << "------ avrg number of read cycles " << mTotalReadCycles / mNSamples
                  << "  max number of read cycles " << mMaxReadCycles;
      }
      mNSamples=0;
      mTotalReadCycles=0;
      mMinTimeBetweenSample=-1;
      mMaxTimeBetweenSample=-1;
      mMaxReadCycles=-1;
      mLastCalcTime=duration.count();
    }
#endif //USE_CHRONO

    if (!mSkipProcessing) {
      // prepare input from messages
      vector<o2::AliceHLT::MessageFormat::BufferDesc_t> dataArray;
      for (vector<unique_ptr<FairMQMessage>>::iterator msg=inputMessages.begin();
           msg!=inputMessages.end(); msg++) {
        void* buffer=(*msg)->GetData();
        dataArray.emplace_back(reinterpret_cast<unsigned char*>(buffer), (*msg)->GetSize());
      }

      // create a signal with the callback to the buffer allocation, the component
      // can create messages via the callback and writes data directly to buffer
      cballoc_signal_t cbsignal;
      cbsignal.connect([this](unsigned int size){return this->createMessageBuffer(size);} );
      mMessages.clear();

      // call the component
      if ((iResult=mComponent->process(dataArray, &cbsignal))<0) {
        LOG(ERROR) << "component processing failed with error code " << iResult;
      }

      // build messages from output data
      if (dataArray.size() > 0) {
        if (mVerbosity > 2) {
          LOG(INFO) << "processing " << dataArray.size() << " buffer(s)";
        }
        for (auto opayload : dataArray) {
          FairMQMessage* omsg=nullptr;
          // loop over pre-allocated messages
          for (auto premsg = begin(mMessages); premsg != end(mMessages); premsg++) {
            if ((*premsg)->GetData() == opayload.mP &&
                (*premsg)->GetSize() == opayload.mSize) {
              omsg=(*premsg).get();
              if (mVerbosity > 2) {
                LOG(DEBUG) << "using pre-allocated message of size " << opayload.mSize;
              }
              break;
            }
          }
          if (omsg==nullptr) {
            unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
            if (msg.get()) {
              msg->Rebuild(opayload.mSize);
              if (msg->GetSize() < opayload.mSize) {
                iResult = -ENOSPC;
                break;
              }
              if (mVerbosity > 2) {
                LOG(DEBUG) << "scheduling message of size " << opayload.mSize;
              }
              AliHLTUInt8_t* pTarget = reinterpret_cast<AliHLTUInt8_t*>(msg->GetData());
              memcpy(pTarget, opayload.mP, opayload.mSize);
              mMessages.push_back(move(msg));
            } else {
              if (errorCount == maxError && errorCount++ > 0)
                LOG(ERROR) << "persistent error, suppressing further output";
              else if (errorCount++ < maxError)
                LOG(ERROR) << "can not get output message from framework";
              iResult = -ENOMSG;
            }
          }
        }
      }

      if (mMessages.size()>0) {
        if (fChannels.find("data-out") != fChannels.end() && fChannels["data-out"].size() > 0) {
          fChannels["data-out"].at(0).Send(mMessages);
          if (mVerbosity > 2) {
            LOG(DEBUG) << "sending multipart message with " << mMessages.size() << " parts";
          }
        } else {
          if (errorCount == maxError && errorCount++ > 0)
            LOG(ERROR) << "persistent error, suppressing further output";
          else if (errorCount++ < maxError)
            LOG(ERROR) << "no output slot available (" << (fChannels.find("data-out") == fChannels.end() ? "uninitialized" : "0 slots")
                       << ")";
        }
        mMessages.clear();
      }
    }

    // cleanup
    inputMessages.clear();
    for (vector<int>::iterator mcit=inputMessageCntPerSocket.begin();
         mcit!=inputMessageCntPerSocket.end(); mcit++) {
      *mcit=0;
    }
  }
}

void WrapperDevice::Pause()
{
  /// inherited from FairMQDevice

  // nothing to do
  FairMQDevice::Pause();
}

void WrapperDevice::SetProperty(const int key, const string& value)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  return FairMQDevice::SetProperty(key, value);
}

string WrapperDevice::GetProperty(const int key, const string& default_)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  return FairMQDevice::GetProperty(key, default_);
}

void WrapperDevice::SetProperty(const int key, const int value)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  switch (key) {
  case PollingPeriod:
    mPollingPeriod = value;
    return;
  case SkipProcessing:
    mSkipProcessing = value;
    return;
  }
  return FairMQDevice::SetProperty(key, value);
}

int WrapperDevice::GetProperty(const int key, const int default_)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  switch (key) {
  case PollingPeriod:
    return mPollingPeriod;
  case SkipProcessing:
    return mSkipProcessing;
  }
  return FairMQDevice::GetProperty(key, default_);
}

unsigned char* WrapperDevice::createMessageBuffer(unsigned size)
{
  /// create a new message with data buffer of specified size
  unique_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
  if (msg.get()==nullptr) return nullptr;
  msg->Rebuild(size);
  if (msg->GetSize() < size) {
    return nullptr;
  }

  if (mVerbosity > 2) {
    LOG(DEBUG) << "allocating message of size " << size;
  }
  mMessages.push_back(move(msg));
  return reinterpret_cast<AliHLTUInt8_t*>(mMessages.back()->GetData());
}
