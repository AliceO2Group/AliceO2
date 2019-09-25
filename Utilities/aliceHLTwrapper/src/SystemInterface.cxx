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

//  @file   SystemInterface.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  FairRoot/ALFA interface to ALICE HLT code

#include "aliceHLTwrapper/SystemInterface.h"
#include "aliceHLTwrapper/AliHLTDataTypes.h"
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <dlfcn.h>
using namespace o2::alice_hlt;

using std::cerr;
using std::cout;
using std::endl;
using std::string;

SystemInterface::SystemInterface()
  : mpAliHLTExtFctInitSystem(nullptr), mpAliHLTExtFctDeinitSystem(nullptr), mpAliHLTExtFctLoadLibrary(nullptr), mpAliHLTExtFctUnloadLibrary(nullptr), mpAliHLTExtFctCreateComponent(nullptr), mpAliHLTExtFctDestroyComponent(nullptr), mpAliHLTExtFctProcessEvent(nullptr), mpAliHLTExtFctGetOutputDataType(nullptr), mpAliHLTExtFctGetOutputSize(nullptr), mEnvironment()
{
  memset(&mEnvironment, 0, sizeof(mEnvironment));
  mEnvironment.fStructSize = sizeof(mEnvironment);
  mEnvironment.fAllocMemoryFunc = SystemInterface::alloc;
}

SystemInterface::~SystemInterface() = default;
/* THINK ABOUT
     make SystemInterface a singleton and release the interface here if still
     active
   */

const char* gInterfaceCallSignatures[] = {
  // int AliHLTAnalysisInitSystem( unsigned long version, AliHLTAnalysisEnvironment* externalEnv, unsigned long runNo, const char* runType)
  "int AliHLTAnalysisInitSystem(unsigned long,AliHLTAnalysisEnvironment*,unsigned long,const char*)",

  // int AliHLTAnalysisDeinitSystem()
  "int AliHLTAnalysisDeinitSystem()",

  // int AliHLTAnalysisLoadLibrary( const char* libraryPath)
  "int AliHLTAnalysisLoadLibrary(const char*)",

  // int AliHLTAnalysisUnloadLibrary( const char* /*libraryPath*/)
  "int AliHLTAnalysisUnloadLibrary(const char*)",

  // int AliHLTAnalysisCreateComponent( const char* componentType, void* environParam, int argc, const char** argv, AliHLTComponentHandle* handle, const char* description)
  "int AliHLTAnalysisCreateComponent(const char*,void*,int,const char**,AliHLTComponentHandle*,const char*)",

  // int AliHLTAnalysisDestroyComponent( AliHLTComponentHandle handle)
  "int AliHLTAnalysisDestroyComponent(AliHLTComponentHandle)",

  // int AliHLTAnalysisProcessEvent( AliHLTComponentHandle handle, const AliHLTComponentEventData* evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData* trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t* size, AliHLTUInt32_t* outputBlockCnt, AliHLTComponentBlockData** outputBlocks, AliHLTComponentEventDoneData** edd)
  "int AliHLTAnalysisProcessEvent(AliHLTComponentHandle,const AliHLTComponentEventData*,const AliHLTComponentBlockData*,AliHLTComponentTriggerData*,AliHLTUInt8_t*,AliHLTUInt32_t*,AliHLTUInt32_t*,AliHLTComponentBlockData**,AliHLTComponentEventDoneData**)",

  // int AliHLTAnalysisGetOutputDataType( AliHLTComponentHandle handle, AliHLTComponentDataType* dataType)
  "int AliHLTAnalysisGetOutputDataType(AliHLTComponentHandle,AliHLTComponentDataType*)",

  // int AliHLTAnalysisGetOutputSize( AliHLTComponentHandle handle, unsigned long* constEventBase, unsigned long* constBlockBase, double* inputBlockMultiplier)
  "int AliHLTAnalysisGetOutputSize(AliHLTComponentHandle,unsigned long*,unsigned long*,double*)",

  nullptr};

int SystemInterface::initSystem(unsigned long runNo)
{
  /// init the system: load interface libraries and read function pointers
  int iResult = 0;

  string libraryPath = ALIHLTANALYSIS_INTERFACE_LIBRARY;

  void* libHandle = dlopen(libraryPath.c_str(), RTLD_NOW);
  if (!libHandle) {
    cerr << "error: can not load library " << libraryPath.c_str() << endl;
#ifdef __APPLE__
    int returnvalue = -EFTYPE;
#else
    int returnvalue = -ELIBACC;
#endif

    return returnvalue;
  }

  AliHLTAnalysisFctGetInterfaceCall fctGetSystemCall =
    (AliHLTAnalysisFctGetInterfaceCall)dlsym(libHandle, ALIHLTANALYSIS_FCT_GETINTERFACECALL);
  if (!fctGetSystemCall) {
    cerr << "error: can not find function '" << ALIHLTANALYSIS_FCT_GETINTERFACECALL << "' in " << libraryPath.c_str()
         << endl;
    return -ENOSYS;
  }

  const char** arrayCalls = gInterfaceCallSignatures;
  for (int i = 0; arrayCalls[i] != nullptr; i++) {
    AliHLTExtFctInitSystem call = (AliHLTExtFctInitSystem)(*fctGetSystemCall)(arrayCalls[i]);
    if (call == nullptr) {
      cerr << "error: can not find function signature '" << arrayCalls[i] << "' in " << libraryPath.c_str() << endl;
    } else {
      cout << "function '" << arrayCalls[i] << "' loaded from " << libraryPath.c_str() << endl;
      switch (i) {
        case 0:
          mpAliHLTExtFctInitSystem = (AliHLTExtFctInitSystem)call;
          break;
        case 1:
          mpAliHLTExtFctDeinitSystem = (AliHLTExtFctDeinitSystem)call;
          break;
        case 2:
          mpAliHLTExtFctLoadLibrary = (AliHLTExtFctLoadLibrary)call;
          break;
        case 3:
          mpAliHLTExtFctUnloadLibrary = (AliHLTExtFctUnloadLibrary)call;
          break;
        case 4:
          mpAliHLTExtFctCreateComponent = (AliHLTExtFctCreateComponent)call;
          break;
        case 5:
          mpAliHLTExtFctDestroyComponent = (AliHLTExtFctDestroyComponent)call;
          break;
        case 6:
          mpAliHLTExtFctProcessEvent = (AliHLTExtFctProcessEvent)call;
          break;
        case 7:
          mpAliHLTExtFctGetOutputDataType = (AliHLTExtFctGetOutputDataType)call;
          break;
        case 8:
          mpAliHLTExtFctGetOutputSize = (AliHLTExtFctGetOutputSize)call;
          break;
        default:
          cerr << "error: number of function signatures does not match expected number of functions" << endl;
      }
    }
  }

  if (mpAliHLTExtFctInitSystem) {
    if ((iResult = (*mpAliHLTExtFctInitSystem)(ALIHLT_DATA_TYPES_VERSION, &mEnvironment, runNo, nullptr)) != 0) {
      cerr << "error: AliHLTAnalysisInitSystem failed with error " << iResult << endl;
      return -ENOSYS;
    }
  }

  return 0;
}

int SystemInterface::releaseSystem()
{
  /// release the system interface, clean all internal structures

  /* THINK ABOUT
     bookkeeping of loaded libraries and unloading them before releasing the system?
   */
  int iResult = 0;
  if (mpAliHLTExtFctDeinitSystem)
    iResult = (*mpAliHLTExtFctDeinitSystem)();
  clear();
  return iResult;
}

int SystemInterface::loadLibrary(const char* libname)
{
  if (!mpAliHLTExtFctLoadLibrary)
    return -ENOSYS;
  return (*mpAliHLTExtFctLoadLibrary)(libname);
}

int SystemInterface::unloadLibrary(const char* libname)
{
  if (!mpAliHLTExtFctUnloadLibrary)
    return -ENOSYS;
  return (*mpAliHLTExtFctUnloadLibrary)(libname);
}

int SystemInterface::createComponent(const char* componentId,
                                     void* environParam,
                                     int argc,
                                     const char** argv,
                                     AliHLTComponentHandle* handle,
                                     const char* description)
{
  if (!mpAliHLTExtFctCreateComponent)
    return -ENOSYS;
  return (*mpAliHLTExtFctCreateComponent)(componentId, environParam, argc, argv, handle, description);
}

int SystemInterface::destroyComponent(AliHLTComponentHandle handle)
{
  if (!mpAliHLTExtFctDestroyComponent)
    return -ENOSYS;
  return (*mpAliHLTExtFctDestroyComponent)(handle);
}

int SystemInterface::processEvent(AliHLTComponentHandle handle,
                                  const AliHLTComponentEventData* evtData, const AliHLTComponentBlockData* blocks,
                                  AliHLTComponentTriggerData* trigData,
                                  AliHLTUInt8_t* outputPtr, AliHLTUInt32_t* size,
                                  AliHLTUInt32_t* outputBlockCnt, AliHLTComponentBlockData** outputBlocks,
                                  AliHLTComponentEventDoneData** edd)
{
  if (!mpAliHLTExtFctProcessEvent)
    return -ENOSYS;
  return (*mpAliHLTExtFctProcessEvent)(handle, evtData, blocks, trigData,
                                       outputPtr, size, outputBlockCnt, outputBlocks, edd);
}

int SystemInterface::getOutputDataType(AliHLTComponentHandle handle, AliHLTComponentDataType* dataType)
{
  if (!mpAliHLTExtFctGetOutputDataType)
    return -ENOSYS;
  return (*mpAliHLTExtFctGetOutputDataType)(handle, dataType);
}

int SystemInterface::getOutputSize(AliHLTComponentHandle handle, unsigned long* constEventBase,
                                   unsigned long* constBlockBase, double* inputBlockMultiplier)
{
  if (!mpAliHLTExtFctGetOutputSize)
    return -ENOSYS;
  return (*mpAliHLTExtFctGetOutputSize)(handle, constEventBase, constEventBase, inputBlockMultiplier);
}

void SystemInterface::clear(const char* /*option*/)
{
  /// clear the object and reset pointer references
  mpAliHLTExtFctInitSystem = nullptr;
  mpAliHLTExtFctDeinitSystem = nullptr;
  mpAliHLTExtFctLoadLibrary = nullptr;
  mpAliHLTExtFctUnloadLibrary = nullptr;
  mpAliHLTExtFctCreateComponent = nullptr;
  mpAliHLTExtFctDestroyComponent = nullptr;
  mpAliHLTExtFctProcessEvent = nullptr;
  mpAliHLTExtFctGetOutputDataType = nullptr;
  mpAliHLTExtFctGetOutputSize = nullptr;
}

void SystemInterface::print(const char* /*option*/) const
{
  /// print info
}

void* SystemInterface::alloc(void* /*param*/, unsigned long size)
{
  // allocate memory
  return malloc(size);
}

void SystemInterface::dealloc(void* buffer, unsigned long /*size*/)
{
  // deallocate memory
  if (buffer == nullptr)
    return;
  free(buffer);
}
