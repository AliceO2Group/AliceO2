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

WrapperDevice::WrapperDevice(int argc, char** argv)
  : mComponent(NULL)
  , mArgv()
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
  FairMQDevice::Init();
}

void WrapperDevice::Run()
{
  /// inherited from FairMQDevice
  int iResult=0;

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
  while ( fState == RUNNING ) {

    // read input messages
    poller->Poll(100);
    for(int i = 0; i < fNumInputs; i++) {
      received = false;
      if (poller->CheckInput(i)){
	auto_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
        received = fPayloadInputs->at(i)->Receive(msg.get());
	if (received) {
	  inputMessages.push_back(msg.release());
          //LOG(INFO) << "------ recieve Msg from " << i ;
	}
      }
    }

    // prepare input from messages
    vector<ALICE::HLT::Component::BufferDesc_t> dataArray;
    for (vector</*const*/ FairMQMessage*>::iterator msg=inputMessages.begin();
	 msg!=inputMessages.end(); msg++) {
      void* buffer=(*msg)->GetData();
      dataArray.push_back(ALICE::HLT::Component::BufferDesc_t(reinterpret_cast<unsigned char*>(buffer), (*msg)->GetSize()));
    }

    // call the component
    if ((iResult=mComponent->Process(dataArray))<0) {
      LOG(ERROR) << "component processing failed with error code" << iResult;
    }

    // build messages from output data
    if (dataArray.size()>0) {
    auto_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
    if (msg.get() && fPayloadOutputs!=NULL && fPayloadOutputs->size()>0) {
      vector<ALICE::HLT::Component::BufferDesc_t>::iterator data=dataArray.begin();
      while (data!=dataArray.end()) {
	msg->Rebuild(data->mSize);
	if (msg->GetSize()<data->mSize) {
	  iResult=-ENOSPC;
	  break;
	}
	AliHLTUInt8_t* pTarget=reinterpret_cast<AliHLTUInt8_t*>(msg->GetData());
	memcpy(pTarget, data->mP, data->mSize);
	if (data+1==dataArray.end()) {
	  // that is the last data block
	  // TODO: replace this with the corresponding FairMQ flag if that becomes available
	  fPayloadOutputs->at(0)->Send(msg.get()/*, ZMQ_SNDMORE*/);
	} else {
	  fPayloadOutputs->at(0)->Send(msg.get());
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

    // cleanup
    for (vector<FairMQMessage*>::iterator mit=inputMessages.begin();
	 mit!=inputMessages.end(); mit++)
      delete *mit;
    inputMessages.clear();
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
