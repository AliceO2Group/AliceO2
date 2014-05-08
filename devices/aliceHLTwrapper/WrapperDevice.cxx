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
#include "AliHLTDataTypes.h"
#include "SystemInterface.h"
#include "HOMERFactory.h"
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

WrapperDevice::WrapperDevice(const char* library, const char* id, const char* parameter, unsigned runNumber)
  : mComponentLibrary(library)
  , mComponentId(id)
  , mComponentParameter(parameter)
  , mRunNumber(runNumber)
  , mOutputBuffer()
  , mOutputBlocks()
  , mpSystem(NULL)
  , mpFactory(NULL)
  , mpReader(NULL)
  , mpWriter(NULL)
  , mProcessor(kEmptyHLTComponentHandle)
{
}

WrapperDevice::~WrapperDevice()
{
}

void WrapperDevice::Init()
{
  /// inherited from FairMQDevice

  int iResult=0;
  // TODO: make the SystemInterface a singleton
  auto_ptr<ALICE::HLT::SystemInterface> iface(new SystemInterface);
  if (iface.get()==NULL || ((iResult=iface->InitSystem(mRunNumber)))<0) {
    LOG(ERROR) << "failed to set up SystemInterface " << iface.get() << " (" << iResult << ")";
    return;
  }
  auto_ptr<ALICE::HLT::HOMERFactory> homerfact(new HOMERFactory);
  if (!homerfact.get()) {
    LOG(ERROR) << "failed to set up HOMERFactory " << homerfact.get();
    return;
  }

  // basic initialization succeeded, make the instances persistent
  mpSystem=iface.release();
  mpFactory=homerfact.release();

  // load the component library
  if ((iResult=mpSystem->LoadLibrary(mComponentLibrary.c_str()))<0)
    return;

  // chop the parameter string in order to provide parameters in the argc/argv format
  vector<const char*> parameters;
  auto_ptr<char> parameterBuffer(new char[mComponentParameter.length()+1]);
  if (mComponentParameter.length()>0 && parameterBuffer.get()!=NULL) {
    strcpy(parameterBuffer.get(), mComponentParameter.c_str());
    char* iterator=parameterBuffer.get();
    parameters.push_back(iterator);
    for (; *iterator!=0; iterator++) {
      if (*iterator!=' ') continue;
      *iterator=0; // separate strings
      if (*(iterator+1)!=' ' && *(iterator+1)!=0)
	parameters.push_back(iterator+1);
    }
  }

  // create component
  if ((iResult=mpSystem->CreateComponent(mComponentId.c_str(), NULL, parameters.size(), &parameters[0], &mProcessor, ""))<0)
    return;
    
}

void WrapperDevice::Run()
{
  /// inherited from FairMQDevice
  if (!mpSystem) return;
  int iResult=0;

  // TODO: make the initial size configurable
  unsigned outputBufferSize=10000;
  mOutputBuffer.reserve(outputBufferSize);

  AliHLTComponentEventData evtData;
  memset(&evtData, 0, sizeof(evtData));
  evtData.fStructSize=sizeof(evtData);

  AliHLTComponentTriggerData trigData;
  memset(&trigData, 0, sizeof(trigData));
  trigData.fStructSize=sizeof(trigData);

  AliHLTUInt32_t outputBlockCnt=0;
  AliHLTComponentBlockData* pOutputBlocks=NULL;
  AliHLTComponentEventDoneData* pEventDoneData=NULL;

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);

  bool received = false;

  // inherited variables of FairMQDevice:
  // fNumInputs
  // fTransportFactory
  // fPayloadInputs
  // fPayloadOutputs
  int NoOfMsgParts=fNumInputs-1;

  vector<FairMQMessage*> inputMessages(fNumInputs);
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
          LOG(INFO) << "------ recieve Msg from " << i ;
	}
      }
    }

    // call HLT component

    // prepare input from messages
    vector<AliHLTComponentBlockData> inputBlocks;

    // process
    evtData.fBlockCnt=inputBlocks.size();
    int nofTrials=2;
    do {
      outputBufferSize=mOutputBuffer.capacity();
      outputBlockCnt=0;
      // TODO: check if that is working with the corresponding allocation method of the 
      // component environment
      if (pOutputBlocks) delete [] pOutputBlocks;
      pOutputBlocks=NULL;
      if (pEventDoneData) delete pEventDoneData;
      pEventDoneData=NULL;

      iResult=mpSystem->ProcessEvent(mProcessor, &evtData, &inputBlocks[0], &trigData,
				     &mOutputBuffer[0], &outputBufferSize,
				     &outputBlockCnt, &pOutputBlocks,
				     &pEventDoneData);

      if (iResult==-ENOSPC) {
	if (mOutputBuffer.capacity()<outputBufferSize) {
	  // resize according to what the component returned
	  mOutputBuffer.reserve(outputBufferSize);
          LOG(INFO) << "increasing output buffer to " << outputBufferSize << " and processing again" ;
	} else {
	  // component did not return what it would need
	  nofTrials=0; // stop
	}
      }
    } while (iResult==-ENOSPC && --nofTrials>0);
    
    // build the message from the output block
    auto_ptr<FairMQMessage> msg(fTransportFactory->CreateMessage());
    if (msg.get()) {
      AliHLTComponentBlockData* pOutputBlock=pOutputBlocks;
      for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++) {
	int msgSize=sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize;
	msg->Rebuild(msgSize);
	if (msg->GetSize()<msgSize) {
	  continue;
	}
	AliHLTUInt8_t* pTarget=reinterpret_cast<AliHLTUInt8_t*>(msg->GetData());
	memcpy(pTarget, pOutputBlock, sizeof(AliHLTComponentBlockData));
	pTarget+=sizeof(AliHLTComponentBlockData);
	memcpy(pTarget, &mOutputBuffer[pOutputBlock->fOffset], pOutputBlock->fSize);
	if (blockIndex+1<outputBlockCnt) {
	  // TODO: replace this with the corresponding FairMQ flag if that becomes available
	  fPayloadOutputs->at(0)->Send(msg.get()/*, ZMQ_SNDMORE*/);
	} else {
	  fPayloadOutputs->at(0)->Send(msg.get());
	}
      }
    }

    // cleanup
    inputBlocks.clear();
    for (vector<FairMQMessage*>::iterator mit=inputMessages.begin();
	 mit!=inputMessages.end(); mit++)
      delete *mit;
    inputMessages.clear();
    mOutputBuffer.clear();
    outputBlockCnt=0;
    if (pOutputBlocks) delete [] pOutputBlocks;
    pOutputBlocks=NULL;
    if (pEventDoneData) delete pEventDoneData;
    pEventDoneData=NULL;
  }

  delete poller;

  rateLogger.interrupt();
  rateLogger.join();
}

void WrapperDevice::Pause()
{
  /// inherited from FairMQDevice

  // nothing to do
}

void WrapperDevice::Shutdown()
{
  /// inherited from FairMQDevice

  int iResult=0;
  if (!mpSystem) return;
  
  iResult=mpSystem->DestroyComponent(mProcessor);
  iResult=mpSystem->ReleaseSystem();

}

void WrapperDevice::InitOutput()
{
  /// inherited from FairMQDevice

}

void WrapperDevice::InitInput()
{
  /// inherited from FairMQDevice

}

