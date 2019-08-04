// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef SYSTEMINTERFACE_H
#define SYSTEMINTERFACE_H
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

//  @file   SystemInterface.h
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  FairRoot/ALFA interface to ALICE HLT code

#include "AliHLTDataTypes.h"
namespace o2
{
namespace alice_hlt
{

/// @class SystemInterface
/// Tool class for the ALICE HLT external interface defined in
/// AliHLTDataTypes.h
///
/// The class loads the interface library and loads the function
/// pointers. The individual functions of the external interface
/// can be used by calling the corresponding functions of this class.
class SystemInterface
{
 public:
  /// default constructor
  SystemInterface();
  /// destructor
  ~SystemInterface();

  /** initilize the system
   *  load external library and set up the HLT system
   */
  int initSystem(unsigned long runNo);

  /** cleanup and release system
   */
  int releaseSystem();

  /** load HLT plugin library
   */
  int loadLibrary(const char* libname);

  /** unload HLT plugin library
   */
  int unloadLibrary(const char* libname);

  /** create/factorize component
   *  @param componentType
   *  @param environParam
   *  @param argc
   *  @param argv
   *  @param handle
   *  @param description
   *  @return 0 on success and valid handle
   */
  int createComponent(const char* componentId,
                      void* environParam,
                      int argc,
                      const char** argv,
                      AliHLTComponentHandle* handle,
                      const char* description);

  /** create/factorize component
   *  @param handle
   *  @return 0 on success
   */
  int destroyComponent(AliHLTComponentHandle handle);

  /** process event
   */
  int processEvent(AliHLTComponentHandle handle,
                   const AliHLTComponentEventData* evtData, const AliHLTComponentBlockData* blocks,
                   AliHLTComponentTriggerData* trigData,
                   AliHLTUInt8_t* outputPtr, AliHLTUInt32_t* size,
                   AliHLTUInt32_t* outputBlockCnt, AliHLTComponentBlockData** outputBlocks,
                   AliHLTComponentEventDoneData** edd);

  /** get the output data type
   */
  int getOutputDataType(AliHLTComponentHandle handle, AliHLTComponentDataType* dataType);

  /** get output data size
   *  return an estimation of the size of the produced data relative to the number of
   *  input blocks and input size
   */
  int getOutputSize(AliHLTComponentHandle handle, unsigned long* constEventBase,
                    unsigned long* constBlockBase, double* inputBlockMultiplier);

  /// clear the object and reset pointer references
  virtual void clear(const char* /*option*/ = "");

  /// print info
  virtual void print(const char* option = "") const;

  /// allocate memory
  static void* alloc(void* param, unsigned long size);

  /// deallocate memory
  static void dealloc(void* buffer, unsigned long size);

 protected:
 private:
  AliHLTExtFctInitSystem mpAliHLTExtFctInitSystem;
  AliHLTExtFctDeinitSystem mpAliHLTExtFctDeinitSystem;
  AliHLTExtFctLoadLibrary mpAliHLTExtFctLoadLibrary;
  AliHLTExtFctUnloadLibrary mpAliHLTExtFctUnloadLibrary;
  AliHLTExtFctCreateComponent mpAliHLTExtFctCreateComponent;
  AliHLTExtFctDestroyComponent mpAliHLTExtFctDestroyComponent;
  AliHLTExtFctProcessEvent mpAliHLTExtFctProcessEvent;
  AliHLTExtFctGetOutputDataType mpAliHLTExtFctGetOutputDataType;
  AliHLTExtFctGetOutputSize mpAliHLTExtFctGetOutputSize;

  AliHLTAnalysisEnvironment mEnvironment;
};

} // namespace alice_hlt
} // namespace o2
#endif // SYSTEMINTERFACE_H
