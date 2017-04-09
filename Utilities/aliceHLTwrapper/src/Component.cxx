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

#include "aliceHLTwrapper/Component.h"
#include "aliceHLTwrapper/AliHLTDataTypes.h"
#include "aliceHLTwrapper/SystemInterface.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <memory>
using namespace ALICE::HLT;
using namespace o2::AliceHLT;

using std::cerr;
using std::endl;
using std::string;
using std::unique_ptr;
using std::vector;

Component::Component()
  : mOutputBuffer()
  , mpSystem(nullptr)
  , mProcessor(kEmptyHLTComponentHandle)
  , mFormatHandler()
  , mEventCount(-1)
{
}

Component::~Component()
= default;

int Component::init(int argc, char** argv)
{
  /// initialize: scan arguments, setup system interface and create component

  // parse options
  static struct option programOptions[] = {
    {"library",     required_argument, nullptr, 'l'},
    {"component",   required_argument, nullptr, 'c'},
    {"parameter",   required_argument, nullptr, 'p'},
    {"run",         required_argument, nullptr, 'r'},
    {"msgsize",     required_argument, nullptr, 's'},
    {"output-mode", required_argument, nullptr, 'm'},
    {"instance-id", required_argument, nullptr, 'i'},
    {nullptr, 0, nullptr, 0}
  };

  /* getopt_long stores the option index here. */
  char c = 0;
  int iOption = 0;

  // HLT components are implemented in shared libraries, the library name
  // and component id are used to factorize a component
  // optionally, a list of configuration parameters can be specified as
  // one single string which is translated in an array of string in the
  // argc/argv format
  const char* componentLibrary = "";
  const char* componentId = "";
  const char* componentParameter = "";
  const char* instanceId="";

  // the configuration and calibration is fixed for every run and identified
  // by the run no
  int runNumber = 0;

  optind = 1; // indicate new start of scanning, especially when getop has been used in a higher layer already
  while ((c = getopt_long(argc, argv, "l:c:p:r:s:m:i:", programOptions, &iOption)) != -1) {
    switch (c) {
      case 'l':
        componentLibrary = optarg;
        break;
      case 'c':
        componentId = optarg;
        break;
      case 'p':
        componentParameter = optarg;
        break;
      case 'r':
        std::stringstream(optarg) >> runNumber;
        break;
      case 's': {
        unsigned size = 0;
        std::stringstream(optarg) >> size;
        mOutputBuffer.resize(size);
      } break;
      case 'm': {
        unsigned outputMode;
        std::stringstream(optarg) >> outputMode;
        mFormatHandler.setOutputMode(outputMode);
      } break;
      case 'i': {
        instanceId=optarg;
        break;
      }
      case '?':
        // TODO: more error handling
        break;
      default:
        cerr << "unknown option: '" << c << "'" << endl;
    }
  }

  if (componentLibrary == nullptr || componentLibrary[0] == 0 || componentId == nullptr || componentId[0] == 0 ||
      runNumber < 0) {
    cerr << "missing argument" << endl;
    return -EINVAL;
  }

  int iResult = 0;
  // TODO: make the SystemInterface a singleton
  unique_ptr<ALICE::HLT::SystemInterface> iface(new SystemInterface);
  if (iface.get() == nullptr || ((iResult = iface->initSystem(runNumber))) < 0) {
    // LOG(ERROR) << "failed to set up SystemInterface " << iface.get() << " (" << iResult << ")";
    return -ENOSYS;
  }

  // basic initialization succeeded, make the instances persistent
  mpSystem = iface.release();

  // load the component library
  if ((iResult = mpSystem->loadLibrary(componentLibrary)) != 0) return iResult > 0 ? -iResult : iResult;

  // chop the parameter string in order to provide parameters in the argc/argv format
  vector<const char*> parameters;
  unsigned parameterLength = strlen(componentParameter);
  unique_ptr<char> parameterBuffer(new char[parameterLength + 1]);
  if (parameterLength > 0 && parameterBuffer.get() != nullptr) {
    strcpy(parameterBuffer.get(), componentParameter);
    char* iterator = parameterBuffer.get();
    parameters.push_back(iterator);
    for (; *iterator != 0; iterator++) {
      if (*iterator != ' ') continue;
      *iterator = 0; // separate strings
      if (*(iterator + 1) != ' ' && *(iterator + 1) != 0) parameters.push_back(iterator + 1);
    }
  }

  // create component
  string description;
  description+=" chainid="; description+=instanceId;
  if ((iResult=mpSystem->createComponent(componentId, nullptr, parameters.size(), &parameters[0], &mProcessor, description.c_str()))<0) {
    // the ALICE HLT external interface uses the following error definition
    // 0 success
    // >0 error number
    return iResult > 0 ? -iResult : iResult;
  }

  mEventCount = 0;

  return iResult;
}

int Component::process(vector<MessageFormat::BufferDesc_t>& dataArray,
                       cballoc_signal_t* cbAllocate)
{
  if (!mpSystem) return -ENOSYS;
  int iResult = 0;

  unsigned outputBufferSize = 0;

  AliHLTComponentEventData evtData;
  memset(&evtData, 0, sizeof(evtData));
  evtData.fStructSize = sizeof(evtData);
  if (mEventCount >= 0) {
    // very simple approach to provide an event ID
    // TODO: adjust to the relevant format if available
    evtData.fEventID = mEventCount++;
  }

  AliHLTComponentTriggerData trigData;
  memset(&trigData, 0, sizeof(trigData));
  trigData.fStructSize = sizeof(trigData);

  AliHLTUInt32_t outputBlockCnt = 0;
  AliHLTComponentBlockData* pOutputBlocks = nullptr;
  AliHLTComponentEventDoneData* pEventDoneData = nullptr;

  // prepare input structure for the ALICE HLT component
  mFormatHandler.clear();
  mFormatHandler.addMessages(dataArray);
  vector<AliHLTComponentBlockData>& inputBlocks = mFormatHandler.getBlockDescriptors();
  unsigned nofInputBlocks = inputBlocks.size();
  if (dataArray.size() > 0 && nofInputBlocks == 0 && mFormatHandler.getEvtDataList().size() == 0) {
    cerr << "warning: none of " << dataArray.size() << " input buffer(s) recognized as valid input" << endl;
  }
  dataArray.clear();

  if (mFormatHandler.getEvtDataList().size()>0) {
    // copy the oldest event header
    memcpy(&evtData, &mFormatHandler.getEvtDataList().front(), sizeof(AliHLTComponentEventData));
  }

  // determine the total input size, needed later on for the calculation of the output buffer size
  int totalInputSize = 0;
  for (vector<AliHLTComponentBlockData>::const_iterator ci = inputBlocks.begin(); ci != inputBlocks.end(); ci++) {
    totalInputSize += ci->fSize;
  }

  // add event type data block
  AliHLTComponentBlockData eventTypeBlock;
  memset(&eventTypeBlock, 0, sizeof(eventTypeBlock));
  eventTypeBlock.fStructSize = sizeof(eventTypeBlock);
  // Note: no payload!
  eventTypeBlock.fDataType = AliHLTComponentDataTypeInitializer("EVENTTYP", "PRIV");
  eventTypeBlock.fSpecification = gkAliEventTypeData;
  inputBlocks.push_back(eventTypeBlock);

  // process
  evtData.fBlockCnt = inputBlocks.size();
  int nofTrials = 2;
  do {
    unsigned long constEventBase = 0;
    unsigned long constBlockBase = 0;
    double inputBlockMultiplier = 0.;
    mpSystem->getOutputSize(mProcessor, &constEventBase, &constBlockBase, &inputBlockMultiplier);
    outputBufferSize = constEventBase + nofInputBlocks * constBlockBase + totalInputSize * inputBlockMultiplier;
    outputBufferSize+=sizeof(AliHLTComponentStatistics) + sizeof(AliHLTComponentTableEntry);
    // take the full available buffer and increase if that
    // is too little
    mOutputBuffer.resize(mOutputBuffer.capacity());
    if (mOutputBuffer.size() < outputBufferSize) {
      mOutputBuffer.resize(outputBufferSize);
    } else if (nofTrials < 2) {
      // component did not update the output size
      break;
    }
    outputBufferSize = mOutputBuffer.size();
    outputBlockCnt = 0;
    // TODO: check if that is working with the corresponding allocation method of the
    // component environment
    if (pOutputBlocks) delete[] pOutputBlocks;
    pOutputBlocks = nullptr;
    if (pEventDoneData) delete pEventDoneData;
    pEventDoneData = nullptr;

    iResult = mpSystem->processEvent(mProcessor, &evtData, &inputBlocks[0], &trigData,
                                     &mOutputBuffer[0], &outputBufferSize,
                                     &outputBlockCnt, &pOutputBlocks,
                                     &pEventDoneData);
    if (outputBufferSize > 0) {
      if (outputBufferSize > mOutputBuffer.size()) {
        cerr << "fatal error: component writing beyond buffer capacity" << endl;
        return -EFAULT;
      } else if (outputBufferSize < mOutputBuffer.size()) {
        mOutputBuffer.resize(outputBufferSize);
      }
    } else {
      mOutputBuffer.clear();
    }

  } while (iResult == ENOSPC && --nofTrials > 0);

  // prepare output
  { // keep this after removing condition to preserve formatting
    AliHLTUInt8_t* pOutputBufferStart = &mOutputBuffer[0];
    AliHLTUInt8_t* pOutputBufferEnd = pOutputBufferStart + mOutputBuffer.size();
    // consistency check for data blocks
    // 1) all specified data must be either inside the output buffer given
    //    to the component or in one of the input buffers
    // 2) filter out special data blocks if they have been forwarded
    //
    // Some components set both pointer and offset in the block descriptor as data
    // reference for blocks in the output buffer. Due to a misinterpretation in the
    // early days of development that is only allowed if fPtr is set to the beginning
    // of output buffer. The block is however fully described by offset relative to
    // beginning of output buffer.
    unsigned validBlocks = 0;
    unsigned totalPayloadSize = 0;
    AliHLTComponentBlockData* pOutputBlock = pOutputBlocks;
    AliHLTComponentBlockData* pFiltered = pOutputBlocks;
    for (unsigned blockIndex = 0; blockIndex < outputBlockCnt; blockIndex++, pOutputBlock++) {
      // filter special data blocks
      if (pOutputBlock->fDataType == eventTypeBlock.fDataType) continue;

      // block descriptors without any attached payload are propagated
      bool bValid = pOutputBlock->fSize == 0;

      // calculate the data reference
      AliHLTUInt8_t* pStart =
        pOutputBlock->fPtr != nullptr ? reinterpret_cast<AliHLTUInt8_t*>(pOutputBlock->fPtr) : &mOutputBuffer[0];
      pStart += pOutputBlock->fOffset;
      AliHLTUInt8_t* pEnd = pStart + pOutputBlock->fSize;
      pOutputBlock->fPtr = pStart;
      pOutputBlock->fOffset = 0;

      // first search in the output buffer
      bValid = bValid || pStart >= pOutputBufferStart && pEnd <= pOutputBufferEnd;

      // possibly a forwarded data block, try the input buffers
      if (!bValid) {
        vector<AliHLTComponentBlockData>::const_iterator ci = inputBlocks.begin();
        for (; ci != inputBlocks.end(); ci++) {
          AliHLTUInt8_t* pInputBufferStart = reinterpret_cast<AliHLTUInt8_t*>(ci->fPtr);
          AliHLTUInt8_t* pInputBufferEnd = pInputBufferStart + ci->fSize;
          if ((bValid = (pStart >= pInputBufferStart && pEnd <= pInputBufferEnd))) {
            break;
          }
        }
      }

      if (bValid) {
        totalPayloadSize += pOutputBlock->fSize;
        validBlocks++;
        memcpy(pFiltered, pOutputBlock, sizeof(AliHLTComponentBlockData));
        pFiltered++;
      } else {
        cerr << "Inconsistent data reference in output block " << blockIndex << endl;
      }
    }
    evtData.fBlockCnt=validBlocks;

    // create the messages
    // TODO: for now there is an extra copy of the data, but it should be
    // handled in place
    vector<MessageFormat::BufferDesc_t> outputMessages =
      mFormatHandler.createMessages(pOutputBlocks, validBlocks, totalPayloadSize, evtData, cbAllocate);
    dataArray.insert(dataArray.end(), outputMessages.begin(), outputMessages.end());
  }

  // cleanup
  // NOTE: don't cleanup mOutputBuffer as the data is going to be used outside the class
  // until released.
  inputBlocks.clear();
  outputBlockCnt = 0;
  if (pOutputBlocks) delete[] pOutputBlocks;
  pOutputBlocks = nullptr;
  if (pEventDoneData) delete pEventDoneData;
  pEventDoneData = nullptr;

  return -iResult;
}
