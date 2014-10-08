//-*- Mode: C++ -*-

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
namespace ALICE
{
  namespace HLT
  {

    class SystemInterface {
    public:
      /// default constructor
      SystemInterface();
      /// destructor
      ~SystemInterface();

      /** initilize the system
       *  load external library and set up the HLT system
       */
      int InitSystem(unsigned long runNo);

      /** cleanup and release system
       */
      int ReleaseSystem();

      /** load HLT plugin library
       */
      int LoadLibrary(const char* libname);

      /** unload HLT plugin library
       */
      int UnloadLibrary(const char* libname);

      /** create/factorize component
       *  @param componentType
       *  @param environParam
       *  @param argc
       *  @param argv
       *  @param handle
       *  @param description
       *  @return 0 on success and valid handle
       */
      int CreateComponent(const char* componentId,
			  void* environParam,
			  int argc,
			  const char** argv,
			  AliHLTComponentHandle* handle,
			  const char* description
			  );

      /** create/factorize component
       *  @param handle
       *  @return 0 on success
       */
      int DestroyComponent(AliHLTComponentHandle handle);

      /** process event
       */
      int ProcessEvent( AliHLTComponentHandle handle,
			const AliHLTComponentEventData* evtData, const AliHLTComponentBlockData* blocks,
			AliHLTComponentTriggerData* trigData,
			AliHLTUInt8_t* outputPtr, AliHLTUInt32_t* size,
			AliHLTUInt32_t* outputBlockCnt,	AliHLTComponentBlockData** outputBlocks,
			AliHLTComponentEventDoneData** edd );

      /** get the output data type
       */
      int GetOutputDataType(AliHLTComponentHandle handle, AliHLTComponentDataType* dataType);

      /** get output data size
       *  return an estimation of the size of the produced data relative to the number of
       *  input blocks and input size
       */
      int GetOutputSize(AliHLTComponentHandle handle, unsigned long* constEventBase,
			unsigned long* constBlockBase, double* inputBlockMultiplier);

      /// clear the object and reset pointer references
      virtual void Clear(const char* /*option*/ ="");

      /// print info
      virtual void Print(const char* option="") const;

      /// allocate memory
      static void* Alloc( void* param, unsigned long size );

      /// deallocate memory
      static void Dealloc( void* buffer, unsigned long size );

    protected:

    private:
      AliHLTExtFctInitSystem        mpAliHLTExtFctInitSystem;
      AliHLTExtFctDeinitSystem      mpAliHLTExtFctDeinitSystem;
      AliHLTExtFctLoadLibrary       mpAliHLTExtFctLoadLibrary;
      AliHLTExtFctUnloadLibrary     mpAliHLTExtFctUnloadLibrary;
      AliHLTExtFctCreateComponent   mpAliHLTExtFctCreateComponent;
      AliHLTExtFctDestroyComponent  mpAliHLTExtFctDestroyComponent;
      AliHLTExtFctProcessEvent      mpAliHLTExtFctProcessEvent;
      AliHLTExtFctGetOutputDataType mpAliHLTExtFctGetOutputDataType;
      AliHLTExtFctGetOutputSize     mpAliHLTExtFctGetOutputSize;

      AliHLTAnalysisEnvironment     mEnvironment;
    };

  }    // namespace hlt
}      // namespace alice
#endif // SYSTEMINTERFACE_H
