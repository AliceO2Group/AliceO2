//-*- Mode: C++ -*-

#ifndef WRAPPERDEVICE_H
#define WRAPPERDEVICE_H
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

//  @file   WrapperDevice.h
//  @author Matthias Richter
//  @since  2014-05-08
//  @brief  FairRoot/ALFA device running ALICE HLT code

#include "AliHLTDataTypes.h"
#include "FairMQDevice.h"
#include <vector>

class AliHLTHOMERReader;
class AliHLTHOMERWriter;

namespace ALICE
{
  namespace HLT
  {
    class HOMERFactory;
    class SystemInterface;

    class WrapperDevice : public FairMQDevice {
    public:
      /// default constructor
      WrapperDevice(const char* library="", const char* id="", const char* parameter="", unsigned runNumber=0);
      /// destructor
      ~WrapperDevice();

      /////////////////////////////////////////////////////////////////
      // the FairMQDevice interface

      /// inherited from FairMQDevice
      virtual void Init();
      /// inherited from FairMQDevice
      virtual void Run();
      /// inherited from FairMQDevice
      virtual void Pause();
      /// inherited from FairMQDevice
      virtual void Shutdown();
      /// inherited from FairMQDevice
      virtual void InitOutput();
      /// inherited from FairMQDevice
      virtual void InitInput();
    protected:

    private:
      // copy constructor prohibited
      WrapperDevice(const WrapperDevice&);
      // assignment operator prohibited
      WrapperDevice& operator=(const WrapperDevice&);

      // send data blocks in a group of messages
      int SendMultiMessages(AliHLTComponentBlockData* pOutputBlocks, AliHLTUInt32_t outputBlockCnt);

      // send data blocks in HOMER format in one message
      int SendHOMERMessage(AliHLTComponentBlockData* pOutputBlocks, AliHLTUInt32_t outputBlockCnt);

      // read a single block from message payload consisting of AliHLTComponentBlockData followed by
      // the block data
      int ReadSingleBlock(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks);

      // read message payload in HOMER format
      int ReadHOMERMessage(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& inputBlocks);

      string          mComponentLibrary;
      string          mComponentId;
      string          mComponentParameter;
      unsigned        mRunNumber;
      vector<AliHLTUInt8_t>            mOutputBuffer;
      vector<AliHLTComponentBlockData> mOutputBlocks;

      SystemInterface*   mpSystem;
      HOMERFactory*      mpFactory;
      AliHLTHOMERReader* mpReader;
      AliHLTHOMERWriter* mpWriter;
      AliHLTComponentHandle mProcessor;
    };

  }    // namespace hlt
}      // namespace alice
#endif // WRAPPERDEVICE_H
