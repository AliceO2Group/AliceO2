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
#include "Algorithm/HeaderStack.h"
#include "Algorithm/Parser.h"
#include "Algorithm/TableView.h"

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

  // the pointer in this structure should be const, but the old definition in the
  // HLT interface does not allow this
  BlockDescriptor(const void* ptr,
                  AliHLTUInt32_t size,
                  const o2::Header::DataHeader& o2dh)
    : BlockDescriptor(0, const_cast<void*>(ptr), size, AliHLTComponentDataTypeInitializer(o2dh.dataDescription.str, o2dh.dataOrigin.str), o2dh.subSpecification) {}
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

  using DataHeader = o2::Header::DataHeader;
  using HeartbeatFrameEnvelope = o2::Header::HeartbeatFrameEnvelope;
  using HeartbeatHeader = o2::Header::HeartbeatHeader;
  using HeartbeatTrailer = o2::Header::HeartbeatTrailer;

  using ParserType = o2::algorithm::ReverseParser<HeartbeatHeader,
                                                  HeartbeatTrailer
                                                  >;
  using HeartbeatFrameSequence = o2::algorithm::TableView<DataHeader,
                                                          HeartbeatHeader,
                                                          ParserType
                                                          >;

  struct BufferDesc_t {
    using Type = unsigned char;
    using PtrT = Type*;
    using SizeT = size_t;
    PtrT mP;
    SizeT mSize;

    PtrT pointer() const {return mP;}
    SizeT size() const {return mSize;}

    BufferDesc_t(PtrT p, size_t size)
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

  // create message payloads in the internal buffer and return list
  // of decriptors
  std::vector<BufferDesc_t> createMessages(const AliHLTComponentBlockData* blocks,
                                           unsigned count,
                                           unsigned totalPayloadSize,
                                           const AliHLTComponentEventData* evtData = nullptr,
                                           boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate=nullptr);

  // read a sequence of blocks consisting of AliHLTComponentBlockData
  // followed by payload from a buffer
  // place descriptors in the target list
  int readBlockSequence(uint8_t* buffer, unsigned size,
                        std::vector<BlockDescriptor>& descriptorList) const;

  // read message payload in HOMER format
  // place descriptors in the target list
  int readHOMERFormat(uint8_t* buffer, unsigned size,
                      std::vector<BlockDescriptor>& descriptorList) const;

  /**
   * parse an input list and try to interpret in O2 data format
   * O2 format consist of header-payload message pairs. The header message
   * always starts with the DataHeader, optionally there can be more
   * headers in the header stack.
   *
   * An insert function needs to be provided, which has signature
   * (const DataHeader&, ptr, size), e.g. through a lambda
   *   auto insertFct = [&] (const auto & dataheader,
   *                         BufferDesc_t::PtrT ptr,
   *                         BufferDesc_t::SizeT size) {
   *     // do something with dataheader and buffer
   *   }
   *
   * Optionally, also the header stack can be parsed by specifying further
   * arguments. For every header supposed to be parsed, a pair of a dummy object
   * and callback has to be specified, e.g.
   *   // handler callback for MyHeaderStruct
   *   auto onMyHeaderStruct = [&] (const auto & mystruct) {
   *     // do something with mystruct
   *   }; // end handler callback
   *
   *   parseO2Format(list, insertFct, MyHeaderStruct(), onMyHeaderStruct);
   *
   * TODO: this function can probably be factored out to module 'Algorithm'
   */
  template<
    typename InputListT
    , typename InsertFctT // (const DataHeader&, ptr, size)
    , typename... HeaderStackTypes // pairs of HeaderType and CallbackType
    >
  int parseO2Format(const InputListT& list,
                    InsertFctT insert,
                    HeaderStackTypes&&... stackArgs
                    );

  // create HOMER format from the output blocks
  AliHLTHOMERWriter* createHOMERFormat(const AliHLTComponentBlockData* pOutputBlocks,
                                       uint32_t outputBlockCnt) const;

  // insert event header to list, sort by time, oldest first
  int insertEvtData(const AliHLTComponentEventData& evtData);

  // get event header list
  const std::vector<AliHLTComponentEventData>& getEvtDataList() const {
    return mListEvtData;
  }

  using BlockDescriptorList = std::vector<BlockDescriptor>;
  using IteratorBase = std::iterator<std::forward_iterator_tag, BlockDescriptorList>;

  // a forward iterator to access the data sets column vice
  // the columns are defined be heartbeats
  // for sure a better name can be found in the course of further development
  class ColumnIterator : public IteratorBase {
  public:
    using ParentType = MessageFormat;
    using self_type = ColumnIterator;
    using reference = typename IteratorBase::reference;
    using pointer = typename IteratorBase::pointer;

    ColumnIterator() = delete;
    ColumnIterator(ParentType & parent)
      : mPosition(-1)
      , mParent(parent)
      , mElement()
      , mColumnIterator(mParent.mSeqHandler.end())
    {}
    ColumnIterator(ParentType & parent, unsigned position = 0)
      : mPosition(position)
      , mParent(parent)
      , mElement()
      , mColumnIterator(mParent.mSeqHandler.begin())
    {
      auto end = mParent.mSeqHandler.end();
      auto i = mPosition;
      for (; i > 0 && mColumnIterator != end; --i, ++mColumnIterator);
    }
    ~ColumnIterator() = default;

    // prefix increment
    self_type& operator++() {
      mElement.clear();
      if (mPosition >= mParent.mSeqHandler.getNColumns()) {
        mPosition = -1;
      } else {
        ++mPosition;
        if (mParent.mSeqHandler.getNColumns() != 0) ++mColumnIterator;
      }
      return *this;
    }
    // postfix increment
    self_type operator++(int /*unused*/) {
      self_type copy(*this); operator++(); return copy;
    }
    // return reference
    reference operator*() {
      if (mElement.size() == 0) {
        InitElement();
      }
      return mElement;
    }
    // comparison
    bool operator==(const self_type& rh) {
      return mPosition == rh.mPosition;
    }
    // comparison
    bool operator!=(const self_type& rh) {
      return mPosition != rh.mPosition;
    }

  private:
    void InitElement() {
      // init the element, aka the block decriptor list, from the parent
      if (mPosition < 0) return;
      if (mPosition == 0 && mParent.mSeqHandler.getNColumns() == 0) {
        // no heartbeat sequence, there is only one valid iterator position
        mElement.insert(mElement.begin(),
                        mParent.mBlockDescriptors.begin(),
                        mParent.mBlockDescriptors.end()
                        );
        return;
      }
      if (mPosition < mParent.mSeqHandler.getNColumns()) {
        // current position is within the column range
        // assemble the column pointed to by the column iterator
        // it provides a begin method to iterate row by row
        unsigned rowIndex = 0;
        for (const auto & i : mColumnIterator) {
          mElement.emplace_back(i.buffer, i.size,
                                mParent.mSeqHandler.getRowData(rowIndex++));
        }
        return;
      }
    }

    int mPosition;
    ParentType& mParent;
    HeartbeatFrameSequence::ColumnIterator mColumnIterator;
    BlockDescriptorList mElement;
  };

  ColumnIterator begin() {
    return ColumnIterator(*this, 0);
  }

  ColumnIterator end() {
    return ColumnIterator(*this, -1);
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
  BufferDesc_t::PtrT MakeTarget(unsigned size, unsigned position, boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate);

  std::vector<BlockDescriptor> mBlockDescriptors;
  /// internal buffer to assemble message data
  std::vector<BufferDesc_t::Type> mDataBuffer;
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
  /// handler for sequences of heartbeat frames
  HeartbeatFrameSequence mSeqHandler;

};

template<
  typename InputListT
  , typename InsertFctT // (const auto&, ptr, size)
  , typename... HeaderStackTypes // pairs of HeaderType and CallbackType
  >
int MessageFormat::parseO2Format(const InputListT& list,
                                InsertFctT insert,
                                HeaderStackTypes&&... stackArgs
                                )
{
  const o2::Header::DataHeader* dh = nullptr;
  for (auto part : list) {
    if (!dh) {
      // new header - payload pair, read DataHeader
      dh = o2::Header::get<o2::Header::DataHeader>(part.pointer(), part.size());
      if (!dh) {
        //std::cerr << "can not find DataHeader" << std::endl;
        return -ENOMSG;
      }
      o2::algorithm::dispatchHeaderStackCallback(part.pointer(),
                                                 part.size(),
                                                 stackArgs...
                                                 );
    } else {
      insert(*dh, part.pointer(), part.size());
      dh = nullptr;
    }
  }
  if (dh) {
    // std::cerr << "missing payload to the last header" << std::endl;
    return -ENOMSG;
  }
  return list.size()/2;
}

} // namespace alice_hlt
} // namespace o2
#endif // MESSAGEFORMAT_H
