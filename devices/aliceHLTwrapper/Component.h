//-*- Mode: C++ -*-

#ifndef COMPONENT_H
#define COMPONENT_H
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

//  @file   Component.h
//  @author Matthias Richter
//  @since  2014-05-08
//  @brief  A component running ALICE HLT code

#include "AliHLTDataTypes.h"
#include "MessageFormat.h"
#include <vector>

class AliHLTHOMERReader;
class AliHLTHOMERWriter;

namespace ALICE
{
  namespace HLT
  {
    class HOMERFactory;
    class SystemInterface;

    class Component {
    public:
      /// default constructor
      Component();
      /// destructor
      ~Component();

      int Init(int argc, char** argv);

      enum {
	// all blocks in HOMER format
	kOutputModeHOMER=0,
	// each block individually as part of a multi-part output
	kOutputModeMultiPart,
	// all blocks as sequence of header and payload
	kOutputModeSequence,
	kOutputModeLast
      };

      int Process(vector<AliceO2::AliceHLT::MessageFormat::BufferDesc_t>& dataArray);

      AliHLTUInt64_t ByteSwap64(AliHLTUInt64_t src);
      AliHLTUInt32_t ByteSwap32(AliHLTUInt32_t src);

    protected:

    private:
      // copy constructor prohibited
      Component(const Component&);
      // assignment operator prohibited
      Component& operator=(const Component&);

      // create HOMER format from the output blocks
      AliHLTHOMERWriter* CreateHOMERFormat(AliHLTComponentBlockData* pOutputBlocks, AliHLTUInt32_t outputBlockCnt);

      vector<AliHLTUInt8_t>            mOutputBuffer;

      SystemInterface*   mpSystem;
      HOMERFactory*      mpFactory;
      AliHLTHOMERWriter* mpWriter;
      AliHLTComponentHandle mProcessor;
      int mOutputMode;
      int mEventCount;
    };

  }    // namespace hlt
}      // namespace alice
#endif // COMPONENT_H
