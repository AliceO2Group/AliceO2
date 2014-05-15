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
  , mOutputMode(kOutputModeSequence)
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
    {"parameter",   required_argument, 0, 'p'},
    {"run",         required_argument, 0, 'r'},
    {"msgsize",     required_argument, 0, 's'},
    {"output-mode", required_argument, 0, 'm'},
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

  optind=1; // indicate new start of scanning, especially when getop has been used in a higher layer already
  while ((c=getopt_long(argc, argv, "l:c:p:r:s:m:", programOptions, &iOption)) != -1) {
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
	mOutputBuffer.resize(size);
      }
      break;
    case 'm':
      std::stringstream(optarg) >> mOutputMode;
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

  unsigned outputBufferSize=0;

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
    if (ReadBlockSequence(data->mP, data->mSize, inputBlocks)<0) {
      // not in the format of a single block, check if its a HOMER block
      if (ReadHOMERFormat(data->mP, data->mSize, inputBlocks)<0) {
	// not in HOMER format either, ignore the input
	// TODO: decide if that should be an error
      }
    }
  }
  unsigned nofInputBlocks=inputBlocks.size();
  if (dataArray.size()>0 && nofInputBlocks==0) {
    cerr << "warning: none of " << dataArray.size() << " input buffers recognized as valid input" << endl;
  }
  dataArray.clear();

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
    if (mOutputBuffer.size()<outputBufferSize) {
      mOutputBuffer.resize(outputBufferSize);
    } else if (nofTrials<2) {
      // component did not update the output size
      break;
    }
    outputBufferSize=mOutputBuffer.size();
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
    if (outputBufferSize>0) {
      if (outputBufferSize>mOutputBuffer.size()) {
	cerr << "fatal error: component writing beyond buffer capacity" << endl;
	return -EFAULT;
      } else if (outputBufferSize<mOutputBuffer.size()) {
	mOutputBuffer.resize(outputBufferSize);
      }
    } else {
      mOutputBuffer.clear();
    }

  } while (iResult==ENOSPC && --nofTrials>0);
    
  // prepare output
  if (outputBlockCnt>0) {
  if (mOutputMode==kOutputModeHOMER) {
    AliHLTHOMERWriter* pWriter=CreateHOMERFormat(pOutputBlocks, outputBlockCnt);
    if (pWriter) {
      AliHLTUInt32_t position=mOutputBuffer.size();
      AliHLTUInt32_t payloadSize=pWriter->GetTotalMemorySize();
      if (mOutputBuffer.capacity()<position+payloadSize) {
	mOutputBuffer.reserve(position+payloadSize);
      }
      pWriter->Copy(&mOutputBuffer[position], 0, 0, 0, 0);
      mpFactory->DeleteWriter(pWriter);
      dataArray.push_back(BufferDesc_t(&mOutputBuffer[position], payloadSize));
    }
  } else if (mOutputMode==kOutputModeMultiPart) {
    AliHLTUInt32_t position=mOutputBuffer.size();
    AliHLTComponentBlockData* pOutputBlock=pOutputBlocks;
    for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++, pOutputBlock++) {
      if (mOutputBuffer.capacity()<position+sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize)
	mOutputBuffer.reserve(position+sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize);
      dataArray.push_back(BufferDesc_t(&mOutputBuffer[position], sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize));
      AliHLTUInt32_t sourcePosition=pOutputBlock->fOffset;
      pOutputBlock->fOffset=0;
      pOutputBlock->fPtr=NULL;
      memcpy(&mOutputBuffer[position], pOutputBlock, sizeof(AliHLTComponentBlockData));
      position+=sizeof(AliHLTComponentBlockData);
      memcpy(&mOutputBuffer[position], &mOutputBuffer[sourcePosition], pOutputBlock->fSize);
      position+=pOutputBlock->fSize;
    }
  } else if (mOutputMode==kOutputModeSequence) {
    AliHLTUInt32_t position=mOutputBuffer.size();
    AliHLTUInt32_t startPosition=position;
    AliHLTComponentBlockData* pOutputBlock=pOutputBlocks;
    for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++, pOutputBlock++) {
      if (mOutputBuffer.capacity()<position+sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize)
	mOutputBuffer.reserve(position+sizeof(AliHLTComponentBlockData)+pOutputBlock->fSize);
      AliHLTUInt32_t sourcePosition=pOutputBlock->fOffset;
      pOutputBlock->fOffset=0;
      pOutputBlock->fPtr=NULL;
      memcpy(&mOutputBuffer[position], pOutputBlock, sizeof(AliHLTComponentBlockData));
      position+=sizeof(AliHLTComponentBlockData);
      memcpy(&mOutputBuffer[position], &mOutputBuffer[sourcePosition], pOutputBlock->fSize);
      position+=pOutputBlock->fSize;
    }
    dataArray.push_back(BufferDesc_t(&mOutputBuffer[startPosition], position-startPosition));
  } else {
    // invalid output mode
    cerr << "error ALICE::HLT::Component: invalid output mode " << mOutputMode << endl; 
  }
  }

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
  for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++, pOutputBlock++) {
    memset( homerHeader, 0, sizeof(homer_uint64)*kCount_64b_Words );
    homerDescriptor.Initialize();
    homer_uint64 id=0;
    homer_uint64 origin=0;
    memcpy(&id, pOutputBlock->fDataType.fID, sizeof(homer_uint64));
    memcpy(((AliHLTUInt8_t*)&origin)+sizeof(homer_uint32), pOutputBlock->fDataType.fOrigin, sizeof(homer_uint32));
    homerDescriptor.SetType(ByteSwap64(id));
    homerDescriptor.SetSubType1(ByteSwap64(origin));
    homerDescriptor.SetSubType2(pOutputBlock->fSpecification);
    homerDescriptor.SetBlockSize(pOutputBlock->fSize);
    writer->AddBlock(homerHeader, &mOutputBuffer[pOutputBlock->fOffset]);
  }
  return writer.release();
}

int Component::ReadBlockSequence(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks)
{
  // read a sequence of blocks consisting of AliHLTComponentBlockData followed by payload
  // from a buffer
  if (buffer==NULL) return 0;
  unsigned position=0;
  vector<AliHLTComponentBlockData> input;
  while (position+sizeof(AliHLTComponentBlockData)<size) {
    AliHLTComponentBlockData* p=reinterpret_cast<AliHLTComponentBlockData*>(buffer+position);
    if (p->fStructSize==0 ||                     // no valid header
	p->fStructSize+position>size ||          // no space for the header
	p->fStructSize+p->fSize+position>size) { // no space for the payload
      // the buffer is only a valid sequence of data blocks if payload
      // of the last block exacly matches the buffer boundary
      // otherwize all blocks added until now are ignored
      return -ENODATA;
    }
    // insert a new block
    input.push_back(*p);
    position+=p->fStructSize;
    if (p->fSize>0) {
      input.back().fPtr=buffer+position;
      position+=p->fSize;
    } else {
      // Note: also a valid block, payload is optional
      input.back().fPtr=NULL;
    }
    // offset always 0 for iput blocks
    input.back().fOffset=0;
  }

  inputBlocks.insert(inputBlocks.end(), input.begin(), input.end());
  return input.size();
}

int Component::ReadHOMERFormat(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks)
{
  // read message payload in HOMER format
  if (buffer==NULL || mpFactory==NULL) return -EINVAL;
  auto_ptr<AliHLTHOMERReader> reader(mpFactory->OpenReaderBuffer(buffer, size));
  if (reader.get()==NULL) return -ENOMEM;

  unsigned nofBlocks=0;
  if (reader->ReadNextEvent()==0) {
    nofBlocks=reader->GetBlockCnt();
    for (unsigned i=0; i<nofBlocks; i++) {
      AliHLTComponentBlockData block;
      memset(&block, 0, sizeof(AliHLTComponentBlockData));
      block.fStructSize=sizeof(AliHLTComponentBlockData);
      block.fDataType.fStructSize=sizeof(AliHLTComponentDataType);
      homer_uint64 id=ByteSwap64(reader->GetBlockDataType( i ));
      homer_uint32 origin=ByteSwap32(reader->GetBlockDataOrigin( i ));
      memcpy(&block.fDataType.fID, &id, sizeof(id)>kAliHLTComponentDataTypefIDsize?kAliHLTComponentDataTypefIDsize:sizeof(id));
      memcpy(&block.fDataType.fOrigin, &origin, sizeof(origin)>kAliHLTComponentDataTypefOriginSize?kAliHLTComponentDataTypefOriginSize:sizeof(origin));
      block.fSpecification=reader->GetBlockDataSpec( i );
      block.fPtr=const_cast<void*>(reader->GetBlockData( i ));
      block.fSize=reader->GetBlockDataLength( i );
      inputBlocks.push_back(block);
    }
  }

  return nofBlocks;
}

AliHLTUInt64_t Component::ByteSwap64(AliHLTUInt64_t src)
{
  // swap a 64 bit number
  return ((src & 0xFFULL) << 56) | 
    ((src & 0xFF00ULL) << 40) | 
    ((src & 0xFF0000ULL) << 24) | 
    ((src & 0xFF000000ULL) << 8) | 
    ((src & 0xFF00000000ULL) >> 8) | 
    ((src & 0xFF0000000000ULL) >> 24) | 
    ((src & 0xFF000000000000ULL) >>  40) | 
    ((src & 0xFF00000000000000ULL) >> 56);
}

AliHLTUInt32_t Component::ByteSwap32(AliHLTUInt32_t src)
{
  // swap a 32 bit number
  return ((src & 0xFFULL) << 24) | 
    ((src & 0xFF00ULL) << 8) | 
    ((src & 0xFF0000ULL) >> 8) | 
    ((src & 0xFF000000ULL) >> 24);
}
