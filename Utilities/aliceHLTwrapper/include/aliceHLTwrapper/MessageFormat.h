//-*- Mode: C++ -*-

#ifndef MESSAGEFORMAT_H
#define MESSAGEFORMAT_H
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

//  @file   MessageFormat.h
//  @author Matthias Richter
//  @since  2014-12-11
//  @brief  Helper class for message format of ALICE HLT data blocks

#include "AliHLTDataTypes.h"
#include "HOMERFactory.h"
#include <vector>
#include <boost/signals2.hpp>

class AliHLTHOMERReader;
class AliHLTHOMERWriter;

namespace o2 {
namespace AliceHLT {
/// @class MessageFormat
/// Helper class to format ALICE HLT data blocks for transport in
/// messaging system.
///
/// Data blocks in the ALICE HLT are described by a block descriptor
/// (data type, size, specification) and the actual data somewhere in
/// memory. For transporting them in a message, block descriptors and
/// payloads are written as a sequence, every block descriptor directly
/// followed by its payload
class MessageFormat {
public:
  /// default constructor
  MessageFormat();
  /// destructor
  ~MessageFormat();

  struct BufferDesc_t {
    unsigned char* mP;
    unsigned mSize;

    BufferDesc_t(unsigned char* p, unsigned size)
    {
      mP = p;
      mSize = size;
    }
  };

  enum {
    // all blocks in HOMER format
    kOutputModeHOMER = 0,
    // each block individually as part of a multi-part output
    kOutputModeMultiPart,
    // all blocks as sequence of header and payload
    kOutputModeSequence,
    kOutputModeLast
  };

  // cleanup internal buffers
  void clear();

  // set output mode
  void setOutputMode(unsigned mode) {mOutputMode=mode;}

  // add message
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer
  int addMessage(AliHLTUInt8_t* buffer, unsigned size);

  // add list of messages
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer
  int addMessages(const std::vector<BufferDesc_t>& list);

  // add a block descriptor and its payload to the message
  // planned for future extension
  //int AddOutput(AliHLTComponentBlockData* db);

  std::vector<AliHLTComponentBlockData>& getBlockDescriptors()
  {
    return mBlockDescriptors;
  }

  const std::vector<AliHLTComponentBlockData>& getBlockDescriptors() const
  {
    return mBlockDescriptors;
  }

  // create message payloads in the internal buffer and return list
  // of decriptors
  std::vector<BufferDesc_t> createMessages(const AliHLTComponentBlockData* blocks, unsigned count,
                                           unsigned totalPayloadSize, const AliHLTComponentEventData& evtData,
                                           boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate=nullptr);

  // read a sequence of blocks consisting of AliHLTComponentBlockData followed by payload
  // from a buffer
  int readBlockSequence(AliHLTUInt8_t* buffer, unsigned size, std::vector<AliHLTComponentBlockData>& descriptorList) const;

  // read message payload in HOMER format
  int readHOMERFormat(AliHLTUInt8_t* buffer, unsigned size, std::vector<AliHLTComponentBlockData>& descriptorList) const;

  // create HOMER format from the output blocks
  AliHLTHOMERWriter* createHOMERFormat(const AliHLTComponentBlockData* pOutputBlocks,
				       AliHLTUInt32_t outputBlockCnt) const;

  // insert event header to list, sort by time, oldest first
  int insertEvtData(const AliHLTComponentEventData& evtData);

  // get event header list
  const std::vector<AliHLTComponentEventData>& getEvtDataList() const {
    return mListEvtData;
  }

  AliHLTUInt64_t byteSwap64(AliHLTUInt64_t src) const;
  AliHLTUInt32_t byteSwap32(AliHLTUInt32_t src) const;

protected:

private:
  // copy constructor prohibited
  MessageFormat(const MessageFormat&);
  // assignment operator prohibited
  MessageFormat& operator=(const MessageFormat&);

  std::vector<AliHLTComponentBlockData> mBlockDescriptors;
  /// internal buffer to assemble message data
  std::vector<AliHLTUInt8_t>            mDataBuffer;
  /// list of message payload descriptors
  std::vector<BufferDesc_t>             mMessages;
  /// HOMER factory for creation and deletion of HOMER readers and writers
  ALICE::HLT::HOMERFactory*        mpFactory;
  /// output mode: HOMER, multi-message, sequential
  int mOutputMode;
  /// list of event descriptors
  std::vector<AliHLTComponentEventData> mListEvtData;
};

} // namespace AliceHLT
} // namespace AliceO2
#endif // MESSAGEFORMAT_H
