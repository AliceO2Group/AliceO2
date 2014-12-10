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

#include "WrapperDevice.h"
#include "Component.h"
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
#  warning statistics measurement for WrapperDevice disabled: need C++11 standard
#else
#  define USE_CHRONO
#endif
#ifdef USE_CHRONO
#include <chrono>
using std::chrono::system_clock;
typedef std::chrono::milliseconds TimeScale;
#endif //USE_CHRONO

WrapperDevice::WrapperDevice(int argc, char** argv, int verbosity)
  : mComponent(NULL)
  , mArgv()
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
{
}

void WrapperDevice::Init()
{
  /// inherited from FairMQDevice

  int iResult=0;
  std::auto_ptr<Component> component(new ALICE::HLT::Component);
  if (!component.get()) return /*-ENOMEM*/;

  if ((iResult=component->Init(mArgv.size(), &mArgv[0]))<0) {
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

  FairMQDevice::Init();
}

void WrapperDevice::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

#ifdef USE_CHRONO
  static system_clock::time_point refTime = system_clock::now();
#endif //USE_CHRONO
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
  while ( fState == RUNNING ) {

    // read input messages
    poller->Poll(mPollingPeriod);
    int inputsReceived=0;
    bool receivedAtLeastOneMessage=false;
    for(int i = 0; i < fNumInputs; i++) {
      if (inputMessageCntPerSocket[i]>0) {
	inputsReceived++;
	continue;
      }
      received = false;
      if (poller->CheckInput(i)){
	int64_t more = 0;
	do {
	  more=0;
	  auto_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
	  received = fPayloadInputs->at(i)->Receive(msg.get());
	  if (received) {
	    receivedAtLeastOneMessage=true;
	    inputMessages.push_back(msg.release());
	    if (inputMessageCntPerSocket[i]==0)
	      inputsReceived++; // count only the first message on that socket
	    inputMessageCntPerSocket[i]++;
	    if (mVerbosity>3) {
	      LOG(INFO) << " |---- recieve Msg from socket no" << i ;
	    }
	    size_t more_size = sizeof(more);
	    fPayloadInputs->at(i)->GetOption("rcv-more", &more, &more_size);
	  }
	} while (more);
	if (mVerbosity>2) {
	  LOG(INFO) << "------ recieved " << inputMessageCntPerSocket[i] << " message(s) from socket no" << i ;
	}
      }
    }
    if (receivedAtLeastOneMessage) nReadCycles++;
    if (inputsReceived<fNumInputs) {
      continue;
    }
    mNSamples++;
    mTotalReadCycles+=nReadCycles;
    if (mMaxReadCycles<0 || mMaxReadCycles<nReadCycles)
      mMaxReadCycles=nReadCycles;
    // if (nReadCycles>1) {
    //   LOG(INFO) << "------ recieved complete Msg from " << fNumInputs << " input(s) after " << nReadCycles << " read cycles" ;
    // }
    nReadCycles=0;
#ifdef USE_CHRONO
    auto duration = std::chrono::duration_cast< TimeScale>(std::chrono::system_clock::now() - refTime);

    if (mLastSampleTime>=0) {
      int sampleTimeDiff=duration.count()-mLastSampleTime;
      if (mMinTimeBetweenSample < 0 || sampleTimeDiff<mMinTimeBetweenSample)
    	mMinTimeBetweenSample=sampleTimeDiff;
      if (mMaxTimeBetweenSample < 0 || sampleTimeDiff>mMaxTimeBetweenSample)
    	mMaxTimeBetweenSample=sampleTimeDiff;
    }
    mLastSampleTime=duration.count();
    if (duration.count()-mLastCalcTime>fLogIntervalInMs) {
      LOG(INFO) << "------ processed  " << mNSamples << " sample(s) ";
      if (mNSamples>0) {
    	LOG(INFO) << "------ min  " << mMinTimeBetweenSample << "ms, max " << mMaxTimeBetweenSample << "ms avrg " << (duration.count()-mLastCalcTime)/mNSamples << "ms ";
    	LOG(INFO) << "------ avrg number of read cycles " << mTotalReadCycles/mNSamples << "  max number of read cycles " << mMaxReadCycles;
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
    vector<ALICE::HLT::Component::BufferDesc_t> dataArray;
    for (vector</*const*/ FairMQMessage*>::iterator msg=inputMessages.begin();
	 msg!=inputMessages.end(); msg++) {
      void* buffer=(*msg)->GetData();
      dataArray.push_back(ALICE::HLT::Component::BufferDesc_t(reinterpret_cast<unsigned char*>(buffer), (*msg)->GetSize()));
    }

    // call the component
    if ((iResult=mComponent->Process(dataArray))<0) {
      LOG(ERROR) << "component processing failed with error code " << iResult;
    }

    // build messages from output data
    if (dataArray.size()>0) {
      if (mVerbosity>2) {
	LOG(INFO) << "processing " << dataArray.size() << " buffer(s)";
      }
    auto_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
    if (msg.get() && fPayloadOutputs!=NULL && fPayloadOutputs->size()>0) {
      vector<ALICE::HLT::Component::BufferDesc_t>::iterator data=dataArray.begin();
      while (data!=dataArray.end()) {
	if (mVerbosity>2) {
	  LOG(INFO) << "sending message of size " << data->mSize;
	}
	msg->Rebuild(data->mSize);
	if (msg->GetSize()<data->mSize) {
	  iResult=-ENOSPC;
	  break;
	}
	AliHLTUInt8_t* pTarget=reinterpret_cast<AliHLTUInt8_t*>(msg->GetData());
	memcpy(pTarget, data->mP, data->mSize);
	if (data+1==dataArray.end()) {
	  // this is the last data block
	  fPayloadOutputs->at(0)->Send(msg.get());
	} else {
	  fPayloadOutputs->at(0)->Send(msg.get(), "snd-more");
	}
      
	data=dataArray.erase(data);
      }
    } else if (fPayloadOutputs==NULL || fPayloadOutputs->size()==0) {
      if (errorCount==maxError && errorCount++>0)
	LOG(ERROR) << "persistent error, suppressing further output";
      else if (errorCount++<maxError)
	LOG(ERROR) << "no output slot available (" << (fPayloadOutputs==NULL?"uninitialized":"0 slots") << ")";
    } else {
      if (errorCount==maxError && errorCount++>0)
	LOG(ERROR) << "persistent error, suppressing further output";
      else if (errorCount++<maxError)
	LOG(ERROR) << "can not get output message from framework";
      iResult=-ENOMSG;
    }
    }
    }

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

void WrapperDevice::Pause()
{
  /// inherited from FairMQDevice

  // nothing to do
  FairMQDevice::Pause();
}

void WrapperDevice::Shutdown()
{
  /// inherited from FairMQDevice

  int iResult=0;
  // TODO: shutdown component and delte instance 

  FairMQDevice::Shutdown();
}

void WrapperDevice::InitOutput()
{
  /// inherited from FairMQDevice

  FairMQDevice::InitOutput();
}

void WrapperDevice::InitInput()
{
  /// inherited from FairMQDevice

  FairMQDevice::InitInput();
}

void WrapperDevice::SetProperty(const int key, const string& value, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  return FairMQDevice::SetProperty(key, value, slot);
}

string WrapperDevice::GetProperty(const int key, const string& default_, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  return FairMQDevice::GetProperty(key, default_, slot);
}

void WrapperDevice::SetProperty(const int key, const int value, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::SetProperty
  switch (key) {
  case PollingPeriod:
    mPollingPeriod=value;
    return;
  case SkipProcessing:
    mSkipProcessing=value;
    return;
  }
  return FairMQDevice::SetProperty(key, value, slot);
}

int WrapperDevice::GetProperty(const int key, const int default_, const int slot)
{
  /// inherited from FairMQDevice
  /// handle device specific properties and forward to FairMQDevice::GetProperty
  switch (key) {
  case PollingPeriod:
    return mPollingPeriod;
  case SkipProcessing:
    return mSkipProcessing;
  }
  return FairMQDevice::GetProperty(key, default_, slot);
}
