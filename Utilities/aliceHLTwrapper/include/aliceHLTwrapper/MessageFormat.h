// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MESSAGEFORMAT_H
#define MESSAGEFORMAT_H
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

//  @file   MessageFormat.h
//  @author Matthias Richter
//  @since  2014-12-11
//  @brief  Helper class for message format of ALICE HLT data blocks

#include "AliHLTDataTypes.h"
#include "HOMERFactory.h"
#include <vector>
#include <cstdint>
#include <boost/signals2.hpp>
#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"

class AliHLTHOMERReader;
class AliHLTHOMERWriter;

namespace o2 {
namespace alice_hlt {
/// @struct BlockDescriptor
/// Helper struct to provide constructors to AliHLTComponentBlockData
///
struct BlockDescriptor : public AliHLTComponentBlockData {
  BlockDescriptor(AliHLTUInt32_t offset,
                  void* ptr,
                  AliHLTUInt32_t size,
                  AliHLTComponentDataType datatype,
                  AliHLTUInt32_t specification)
  {
    fStructSize = sizeof(AliHLTComponentBlockData);
    memset(&fShmKey, 0, sizeof(AliHLTComponentShmData));
    fOffset = offset;
    fPtr = ptr;
    fSize = size;
    fDataType = datatype;
    fSpecification = specification;
  }

  BlockDescriptor(const AliHLTComponentBlockData& src)
    : BlockDescriptor(src.fOffset, src.fPtr, src.fSize, src.fDataType, src.fSpecification) {}

  BlockDescriptor(void* ptr,
                  AliHLTUInt32_t size,
                  AliHLTComponentDataType datatype,
                  AliHLTUInt32_t specification)
    : BlockDescriptor(0, ptr, size, datatype, specification) {}

  BlockDescriptor()
    : BlockDescriptor(0, nullptr, 0, kAliHLTVoidDataType, kAliHLTVoidDataSpec) {}

  BlockDescriptor(void* ptr,
                  AliHLTUInt32_t size,
                  const o2::header::DataHeader& o2dh)
    : BlockDescriptor(0, ptr, size, AliHLTComponentDataTypeInitializer(o2dh.dataDescription.str, o2dh.dataOrigin.str), o2dh.subSpecification) {}
};

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

  using DataHeader = o2::header::DataHeader;
  using HeartbeatFrameEnvelope = o2::header::HeartbeatFrameEnvelope;
  using HeartbeatHeader = o2::header::HeartbeatHeader;
  using HeartbeatTrailer = o2::header::HeartbeatTrailer;

  struct BufferDesc_t {
    using PtrT = unsigned char*;
    PtrT mP;
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
    // O2 data format, header-payload pairs
    kOutputModeO2,
    kOutputModeLast
  };

  // cleanup internal buffers
  void clear();

  // set output mode
  void setOutputMode(unsigned mode) {mOutputMode=mode;}

  // add message
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer
  int addMessage(uint8_t* buffer, unsigned size);

  // add list of messages
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer
  int addMessages(const std::vector<BufferDesc_t>& list);

  // add a block descriptor and its payload to the message
  // planned for future extension
  //int AddOutput(AliHLTComponentBlockData* db);

  std::vector<BlockDescriptor>& getBlockDescriptors()
  {
    return mBlockDescriptors;
  }

  const std::vector<BlockDescriptor>& getBlockDescriptors() const
  {
    return mBlockDescriptors;
  }

  // create message payloads in the internal buffer and return list
  // of decriptors
  std::vector<BufferDesc_t> createMessages(const AliHLTComponentBlockData* blocks, unsigned count,
                                           unsigned totalPayloadSize, const AliHLTComponentEventData* evtData = nullptr,
                                           boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate=nullptr);

  // read a sequence of blocks consisting of AliHLTComponentBlockData followed by payload
  // from a buffer
  int readBlockSequence(uint8_t* buffer, unsigned size, std::vector<BlockDescriptor>& descriptorList) const;

  // read message payload in HOMER format
  int readHOMERFormat(uint8_t* buffer, unsigned size, std::vector<BlockDescriptor>& descriptorList) const;

  // read messages in O2 format
  int readO2Format(const std::vector<BufferDesc_t>& list, std::vector<BlockDescriptor>& descriptorList, HeartbeatHeader& hbh, HeartbeatTrailer& hbt) const;

  // create HOMER format from the output blocks
  AliHLTHOMERWriter* createHOMERFormat(const AliHLTComponentBlockData* pOutputBlocks,
                                       uint32_t outputBlockCnt) const;

  // insert event header to list, sort by time, oldest first
  int insertEvtData(const AliHLTComponentEventData& evtData);

  // get event header list
  const std::vector<AliHLTComponentEventData>& getEvtDataList() const {
    return mListEvtData;
  }

  uint64_t byteSwap64(uint64_t src) const;
  uint32_t byteSwap32(uint32_t src) const;

protected:

private:
  // copy constructor prohibited
  MessageFormat(const MessageFormat&);
  // assignment operator prohibited
  MessageFormat& operator=(const MessageFormat&);

  // single point to provide a target pointer, either by using a
  // provided callback function or in the internal buffer, which has
  // to be allocated completely in advance in order to ensure validity
  // of the pointers
  uint8_t* MakeTarget(unsigned size, unsigned position, boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate);

  std::vector<BlockDescriptor> mBlockDescriptors;
  /// internal buffer to assemble message data
  std::vector<uint8_t>            mDataBuffer;
  /// list of message payload descriptors
  std::vector<BufferDesc_t>             mMessages;
  /// HOMER factory for creation and deletion of HOMER readers and writers
  o2::alice_hlt::HOMERFactory*        mpFactory;
  /// output mode: HOMER, multi-message, sequential
  int mOutputMode;
  /// list of event descriptors
  std::vector<AliHLTComponentEventData> mListEvtData;

  /// the current  heartbeat header
  HeartbeatHeader mHeartbeatHeader;
  /// the current  heartbeat trailer
  HeartbeatTrailer mHeartbeatTrailer;
};

} // namespace alice_hlt
} // namespace o2
#endif // MESSAGEFORMAT_H
