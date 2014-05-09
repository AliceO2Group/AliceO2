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

//  @file   Component.cxx
//  @author Matthias Richter
//  @since  2014-05-07 
//  @brief  A component running ALICE HLT code

#include "Component.h"
#include "AliHLTDataTypes.h"
#include "SystemInterface.h"
#include "HOMERFactory.h"
#include "AliHLTHOMERData.h"
#include "AliHLTHOMERWriter.h"
#include "AliHLTHOMERReader.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <memory>
using namespace ALICE::HLT;

Component::Component()
  : mOutputBuffer()
  , mpSystem(NULL)
  , mpFactory(NULL)
  , mpWriter(NULL)
  , mProcessor(kEmptyHLTComponentHandle)
{
}

Component::~Component()
{
}

int Component::Init(int argc, char** argv)
{
  /// initialize: scan arguments, setup system interface and create component

  // parse options
  static struct option programOptions[] = {
    {"library",     required_argument, 0, 'l'},
    {"component",   required_argument, 0, 'c'},
    {"parameter",   optional_argument, 0, 'p'},
    {"run",         required_argument, 0, 'r'},
    {"msgsize",     optional_argument, 0, 's'},
    {0, 0, 0, 0}
  };

  /* getopt_long stores the option index here. */
  char c=0;
  int iOption = 0;

  // HLT components are implemented in shared libraries, the library name
  // and component id are used to factorize a component
  // optionally, a list of configuration parameters can be specified as
  // one single string which is translated in an array of string in the
  // argc/argv format
  const char* componentLibrary="";
  const char* componentId="";
  const char* componentParameter="";

  // the configuration and calibration is fixed for every run and identified
  // by the run no
  int runNumber=0;

  while ((c=getopt_long(argc, argv, "l:c:p:r:s:", programOptions, &iOption)) != -1) {
    switch (c) {
    case 'l':
      componentLibrary=optarg;
      break;
    case 'c':
      componentId=optarg;
      break;
    case 'p':
      componentParameter=optarg;
      break;
    case 'r':
      std::stringstream(optarg) >> runNumber;
      break;
    case 's':
      {
	unsigned size=0;
	std::stringstream(optarg) >> size;
	if (mOutputBuffer.capacity()<size)
	  mOutputBuffer.reserve(size);
      }
      break;
    case '?':
      // TODO: more error handling
      break;
    default:
      cerr << "unknown option: '"<< c << "'" << endl;
    }
  }

  if (componentLibrary==NULL || componentLibrary[0]==0 ||
      componentId==NULL || componentId[0]==0 ||
      runNumber<0) {
    cerr << "missing argument" << endl;
    return -EINVAL;
  }

  int iResult=0;
  // TODO: make the SystemInterface a singleton
  auto_ptr<ALICE::HLT::SystemInterface> iface(new SystemInterface);
  if (iface.get()==NULL || ((iResult=iface->InitSystem(runNumber)))<0) {
    //LOG(ERROR) << "failed to set up SystemInterface " << iface.get() << " (" << iResult << ")";
    return -ENOSYS;
  }
  auto_ptr<ALICE::HLT::HOMERFactory> homerfact(new HOMERFactory);
  if (!homerfact.get()) {
    //LOG(ERROR) << "failed to set up HOMERFactory " << homerfact.get();
    return -ENOSYS;
  }

  // basic initialization succeeded, make the instances persistent
  mpSystem=iface.release();
  mpFactory=homerfact.release();

  // load the component library
  if ((iResult=mpSystem->LoadLibrary(componentLibrary))!=0)
    return iResult>0?-iResult:iResult;

  // chop the parameter string in order to provide parameters in the argc/argv format
  vector<const char*> parameters;
  unsigned parameterLength=strlen(componentParameter);
  auto_ptr<char> parameterBuffer(new char[parameterLength+1]);
  if (parameterLength>0 && parameterBuffer.get()!=NULL) {
    strcpy(parameterBuffer.get(), componentParameter);
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
  if ((iResult=mpSystem->CreateComponent(componentId, NULL, parameters.size(), &parameters[0], &mProcessor, ""))<0)
    return iResult>0?-iResult:iResult;

  return iResult;
}

int Component::Process(vector<BufferDesc_t>& dataArray)
{
  if (!mpSystem) return -ENOSYS;
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


  // prepare input structure for the ALICE HLT component
  vector<AliHLTComponentBlockData> inputBlocks;
  for (vector<BufferDesc_t>::iterator data=dataArray.begin();
       data!=dataArray.end(); data++) {
    if (ReadSingleBlock(data->p, data->size, inputBlocks)<0) {
      // not in the format of a single block, check if its a HOMER block
      if (ReadHOMERFormat(data->p, data->size, inputBlocks)<0) {
	// not in HOMER format either, ignore the input
	// TODO: decide if that should be an error
      }
    }
  }
  unsigned nofInputBlocks=inputBlocks.size();

  // add event type data block
  AliHLTComponentBlockData eventTypeBlock;
  memset(&eventTypeBlock, 0, sizeof(eventTypeBlock));
  eventTypeBlock.fStructSize=sizeof(eventTypeBlock);
  // Note: no payload!
  eventTypeBlock.fDataType=AliHLTComponentDataTypeInitializer("EVENTTYP", "PRIV");
  eventTypeBlock.fSpecification=gkAliEventTypeData;
  inputBlocks.push_back(eventTypeBlock);

  // process
  evtData.fBlockCnt=inputBlocks.size();
  int nofTrials=2;
  do {
    unsigned long constEventBase=0;
    unsigned long constBlockBase=0;
    double inputBlockMultiplier=0.;
    mpSystem->GetOutputSize(mProcessor, &constEventBase, &constBlockBase, &inputBlockMultiplier);
    outputBufferSize=constEventBase+nofInputBlocks*constBlockBase;
    if (mOutputBuffer.capacity()<outputBufferSize) {
      mOutputBuffer.reserve(outputBufferSize);
    } else if (nofTrials<2) {
      // component did not update the output size
      break;
    }
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

  } while (iResult==ENOSPC && --nofTrials>0);
    
  // build the message from the output block

  // cleanup
  // NOTE: don't cleanup mOutputBuffer as the data is going to be used outside the class
  // until released.
  inputBlocks.clear();
  outputBlockCnt=0;
  if (pOutputBlocks) delete [] pOutputBlocks;
  pOutputBlocks=NULL;
  if (pEventDoneData) delete pEventDoneData;
  pEventDoneData=NULL;

  return -iResult;
}

AliHLTHOMERWriter* Component::CreateHOMERFormat(AliHLTComponentBlockData* pOutputBlocks, AliHLTUInt32_t outputBlockCnt)
{
  // send data blocks in HOMER format in one message
  int iResult=0;
  if (!mpFactory) return NULL;
  auto_ptr<AliHLTHOMERWriter> writer(mpFactory->OpenWriter());
  if (writer.get()==NULL) return NULL;

  homer_uint64 homerHeader[kCount_64b_Words];
  HOMERBlockDescriptor homerDescriptor(homerHeader);

  AliHLTComponentBlockData* pOutputBlock=pOutputBlocks;
  for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++) {
    memset( homerHeader, 0, sizeof(homer_uint64)*kCount_64b_Words );
    homerDescriptor.Initialize();
    homer_uint64 id=0;
    homer_uint64 origin=0;
    memcpy(&id, pOutputBlock->fDataType.fID, sizeof(homer_uint64));
    memcpy(((AliHLTUInt8_t*)&origin)+sizeof(homer_uint32), pOutputBlock->fDataType.fOrigin, sizeof(homer_uint32));
    homerDescriptor.SetType((id));
    homerDescriptor.SetSubType1((origin));
    homerDescriptor.SetSubType2(pOutputBlock->fSpecification);
    homerDescriptor.SetBlockSize(pOutputBlock->fSize);
    writer->AddBlock(homerHeader, &mOutputBuffer[pOutputBlock->fOffset]);
  }
  return writer.release();
}

int Component::ReadSingleBlock(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks)
{
  // read a single block from message payload consisting of AliHLTComponentBlockData followed by
  // the block data
  return 0;
}

int Component::ReadHOMERFormat(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks)
{
  // read message payload in HOMER format
  return 0;
}
