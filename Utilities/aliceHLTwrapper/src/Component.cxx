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

//  @file   Component.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  A component running ALICE HLT code

#include "aliceHLTwrapper/Component.h"
#include "aliceHLTwrapper/AliHLTDataTypes.h"
#include "aliceHLTwrapper/SystemInterface.h"
#include "FairMQLogger.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <memory>

using namespace o2::alice_hlt;

using std::endl;
using std::string;
using std::unique_ptr;
using std::vector;
using std::stringstream;

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

constexpr const char* Component::OptionKeys[];

bpo::options_description Component::GetOptionsDescription()
{
  // clang-format off
  bpo::options_description od("HLT Component options");
  od.add_options()
    ((std::string(OptionKeys[OptionKeyLibrary]) + ",l").c_str(),
     bpo::value<string>()->required(),
     "component library")
    ((std::string(OptionKeys[OptionKeyComponent]) + ",c").c_str(),
     bpo::value<string>()->required(),
     "component id")
    ((std::string(OptionKeys[OptionKeyParameter]) + ",p").c_str(),
     bpo::value<string>()->default_value(""),
     "component command line parameter")
    ((std::string(OptionKeys[OptionKeyRun]) + ",r").c_str(),
     bpo::value<string>()->required(),
     "run number")
    ((std::string(OptionKeys[OptionKeyOCDB])).c_str(),
     bpo::value<string>(),
     "ocdb uri")
    ((std::string(OptionKeys[OptionKeyMsgsize]) + ",s").c_str(),
     bpo::value<string>()->default_value("0"),
     "maximum size of output buffer/msg")
    ((std::string(OptionKeys[OptionKeyOutputMode]) + ",m").c_str(),
     bpo::value<string>()->default_value("3"),
     "output mode");
  // clang-format on
  return od;
}

int Component::init(int argc, char** argv)
{
  /// initialize: scan arguments, setup system interface and create component

  // the hidden options are not exposed to the outside
  bpo::options_description od("component options");
  od.add_options()
    (OptionKeys[OptionKeyInstanceId],
     bpo::value<string>()->required(),
     "internal instance id");
  // now add all the visible options
  od.add(GetOptionsDescription());

  // HLT components are implemented in shared libraries, the library name
  // and component id are used to factorize a component
  // optionally, a list of configuration parameters can be specified as
  // one single string which is translated in an array of string in the
  // argc/argv format
  string componentParameter;
  string instanceId;

  // the configuration and calibration is fixed for every run and identified
  // by the run no
  int runNumber = 0;

  bpo::variables_map varmap;
  bpo::store(bpo::parse_command_line(argc, argv, od), varmap);

  for (int option = 0; option < OptionKeyLast; ++option) {
    if (varmap.count(OptionKeys[option]) == 0) continue;
    switch (option) {
    case OptionKeyLibrary: break;
    case OptionKeyComponent: break;
    case OptionKeyParameter:
      componentParameter = varmap[OptionKeys[option]].as<string>();
      break;
    case OptionKeyRun:
      stringstream(varmap[OptionKeys[option]].as<string>()) >> runNumber;
      break;
    case OptionKeyOCDB:
      if (getenv("ALIHLT_HCDBDIR") != nullptr) {
        LOG(WARN) << "overriding value of ALICEHLT_HCDBDIR by --ocdb command option";
      }
      setenv("ALIHLT_HCDBDIR", varmap[OptionKeys[option]].as<string>().c_str(), 1);
      break;
    case OptionKeyMsgsize: {
      unsigned size = 0;
      stringstream(varmap[OptionKeys[option]].as<string>()) >> size;
      mOutputBuffer.resize(size);
    } break;
    case OptionKeyOutputMode: {
      unsigned mode;
      stringstream(varmap[OptionKeys[option]].as<string>()) >> mode;
      mFormatHandler.setOutputMode(mode);
    } break;
    case OptionKeyInstanceId: break;
      instanceId = varmap[OptionKeys[option]].as<string>();
      break;
    }
  }

  // TODO: this can be a loop over all required options
  // probably we won't come here anyhow as the parser will detect
  // the absence of a required parameter, check and handle the parser
  // exception
  if (varmap.count(OptionKeys[OptionKeyLibrary]) == 0 ||
      varmap.count(OptionKeys[OptionKeyComponent]) == 0 ||
      runNumber < 0) {
    LOG(ERROR) << "missing argument, required options: library, component,run";
    return -EINVAL;
  }

  // check the OCDB URI
  // the HLT code relies on ALIHLT_HCDBDIR environment variable to be set
  if (!getenv("ALIHLT_HCDBDIR")) {
    LOG(ERROR) << "FATAL: OCDB URI is needed, use option --ocdb or environment variable ALIHLT_HCDBDIR";
// temporary fix to regain compilation on MacOS (which on some platforms does not define ENOKEY)
#ifndef ENOKEY
#define ENOKEY 126
#endif
    return -ENOKEY;
  }

  int iResult = 0;
  // TODO: make the SystemInterface a singleton
  unique_ptr<o2::alice_hlt::SystemInterface> iface(new SystemInterface);
  if (iface.get() == nullptr || ((iResult = iface->initSystem(runNumber))) < 0) {
    // LOG(ERROR) << "failed to set up SystemInterface " << iface.get() << " (" << iResult << ")";
    return -ENOSYS;
  }

  // basic initialization succeeded, make the instances persistent
  mpSystem = iface.release();

  // load the component library
  if ((iResult = mpSystem->loadLibrary(varmap[OptionKeys[OptionKeyLibrary]].as<string>().c_str())) != 0)
    return iResult > 0 ? -iResult : iResult;

  // chop the parameter string in order to provide parameters in the argc/argv format
  vector<const char*> parameters;
  unsigned parameterLength = componentParameter.length();
  unique_ptr<char> parameterBuffer(new char[parameterLength + 1]);
  if (parameterLength > 0 && parameterBuffer.get() != nullptr) {
    strcpy(parameterBuffer.get(), componentParameter.c_str());
    char* iterator = parameterBuffer.get();
    parameters.emplace_back(iterator);
    for (; *iterator != 0; iterator++) {
      if (*iterator != ' ') continue;
      *iterator = 0; // separate strings
      if (*(iterator + 1) != ' ' && *(iterator + 1) != 0) parameters.emplace_back(iterator + 1);
    }
  }

  // create component
  string description;
  description+=" chainid=" + instanceId;
  if ((iResult=mpSystem->createComponent(varmap[OptionKeys[OptionKeyComponent]].as<string>().c_str(), nullptr, parameters.size(), &parameters[0], &mProcessor, description.c_str()))<0) {
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

  uint32_t outputBlockCnt = 0;
  AliHLTComponentBlockData* pOutputBlocks = nullptr;
  AliHLTComponentEventDoneData* pEventDoneData = nullptr;

  // prepare input structure for the ALICE HLT component
  mFormatHandler.clear();
  mFormatHandler.addMessages(dataArray);
  vector<BlockDescriptor>& inputBlocks = mFormatHandler.getBlockDescriptors();
  unsigned nofInputBlocks = inputBlocks.size();
  if (dataArray.size() > 0 && nofInputBlocks == 0 && mFormatHandler.getEvtDataList().size() == 0) {
    LOG(ERROR) << "warning: none of " << dataArray.size() << " input buffer(s) recognized as valid input";
  }
  dataArray.clear();

  if (mFormatHandler.getEvtDataList().size()>0) {
    // copy the oldest event header
    memcpy(&evtData, &mFormatHandler.getEvtDataList().front(), sizeof(AliHLTComponentEventData));
  }

  // determine the total input size, needed later on for the calculation of the output buffer size
  int totalInputSize = 0;
  for (auto & ci : inputBlocks) {
    totalInputSize += ci.fSize;
  }

  // add event type data block
  // this data block describes the type of the event, set it
  // to 'data' by using specification gkAliEventTypeData
  const AliHLTComponentDataType kDataTypeEvent = AliHLTComponentDataTypeInitializer("EVENTTYP", "PRIV");
  inputBlocks.emplace_back(nullptr, 0, kDataTypeEvent, gkAliEventTypeData);

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
        LOG(ERROR) << "FATAL: fatal error: component writing beyond buffer capacity";
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
    uint8_t* pOutputBufferStart = &mOutputBuffer[0];
    uint8_t* pOutputBufferEnd = pOutputBufferStart + mOutputBuffer.size();
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
      if (pOutputBlock->fDataType == kDataTypeEvent) continue;

      // block descriptors without any attached payload are propagated
      bool bValid = pOutputBlock->fSize == 0;

      // calculate the data reference
      uint8_t* pStart =
        pOutputBlock->fPtr != nullptr ? reinterpret_cast<uint8_t*>(pOutputBlock->fPtr) : &mOutputBuffer[0];
      pStart += pOutputBlock->fOffset;
      uint8_t* pEnd = pStart + pOutputBlock->fSize;
      pOutputBlock->fPtr = pStart;
      pOutputBlock->fOffset = 0;

      // first search in the output buffer
      bValid = bValid || (pStart >= pOutputBufferStart && pEnd <= pOutputBufferEnd);

      // possibly a forwarded data block, try the input buffers
      if (!bValid) {
        for (auto & ci : inputBlocks) {
          uint8_t* pInputBufferStart = reinterpret_cast<uint8_t*>(ci.fPtr);
          uint8_t* pInputBufferEnd = pInputBufferStart + ci.fSize;
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
        LOG(ERROR) << "Inconsistent data reference in output block " << blockIndex;
      }
    }
    evtData.fBlockCnt=validBlocks;

    // create the messages
    // TODO: for now there is an extra copy of the data, but it should be
    // handled in place
    vector<MessageFormat::BufferDesc_t> outputMessages =
      mFormatHandler.createMessages(pOutputBlocks, validBlocks, totalPayloadSize, &evtData, cbAllocate);
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
