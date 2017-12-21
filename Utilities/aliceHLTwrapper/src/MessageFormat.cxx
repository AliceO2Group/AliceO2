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

//  @file   MessageFormat.cxx
//  @author Matthias Richter
//  @since  2014-12-11
//  @brief  Helper class for message format of ALICE HLT data blocks

#include "aliceHLTwrapper/MessageFormat.h"
#include "aliceHLTwrapper/HOMERFactory.h"
#include "aliceHLTwrapper/AliHLTHOMERData.h"
#include "aliceHLTwrapper/AliHLTHOMERWriter.h"
#include "aliceHLTwrapper/AliHLTHOMERReader.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <cassert>
#include <sstream>

using namespace o2::alice_hlt;
using std::cerr;
using std::endl;
using std::unique_ptr;
using std::vector;

// TODO: central logging to be implemented

MessageFormat::MessageFormat()
  : mBlockDescriptors()
  , mDataBuffer()
  , mMessages()
  , mpFactory(nullptr)
  , mOutputMode(kOutputModeO2)
  , mListEvtData()
  , mHeartbeatHeader()
  , mHeartbeatTrailer()
{
  // invalidate the heartbeat header and trailer
  mHeartbeatHeader.headerWord = 0;
  mHeartbeatTrailer.trailerWord = 0;
}

MessageFormat::~MessageFormat()
{
  if (mpFactory) delete mpFactory;
  mpFactory = nullptr;
}

void MessageFormat::clear()
{
  mBlockDescriptors.clear();
  mDataBuffer.clear();
  mMessages.clear();
  mListEvtData.clear();

  // invalidate the heartbeat header and trailer
  mHeartbeatHeader.headerWord = 0;
  mHeartbeatTrailer.trailerWord = 0;
}

int MessageFormat::addMessage(uint8_t* buffer, unsigned size)
{
  // add message
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer

  unsigned count = mBlockDescriptors.size();
  // the buffer might start with an event descriptor of type AliHLTComponentEventData
  // the first read attempt is assuming the descriptor and checking consistency
  // - buffer size at least size of AliHLTComponentEventData struct
  // - fStructSize member matches
  // - number of blocks matches fBlockCnt member
  unsigned position=0;
  AliHLTComponentEventData* evtData=reinterpret_cast<AliHLTComponentEventData*>(buffer+position);
  if (position+sizeof(AliHLTComponentEventData)<=size &&
      evtData->fStructSize==sizeof(AliHLTComponentEventData)) {
    position += sizeof(AliHLTComponentEventData);
  } else {
    // one of the criteria does not match -> no event descriptor
    evtData=nullptr;
  }
  do {
    if (evtData && evtData->fBlockCnt==0 && size<sizeof(AliHLTComponentBlockData)) {
      // special case: no block data, only event header
      break;
    }
    if (readBlockSequence(buffer+position, size-position, mBlockDescriptors) < 0 ||
        (evtData!=nullptr && ((mBlockDescriptors.size()-count) != evtData->fBlockCnt))) {
      // not in the format of a single block, check if its a HOMER block
      if (readHOMERFormat(buffer+position, size-position, mBlockDescriptors) < 0 ||
         (evtData!=nullptr && ((mBlockDescriptors.size()-count) != evtData->fBlockCnt))) {
        // not in HOMER format either
        if (position>0) {
          // try once more without the assumption of event data header
          position=0;
          evtData=nullptr;
          continue;
        }
        return -ENODATA;
      }
    }
  } while (false);

  int result=0;
  if (evtData && (result=insertEvtData(*evtData))<0) {
    // error in the event data header, probably headers of different events
    mBlockDescriptors.resize(count);
    return result;
  }

  return mBlockDescriptors.size() - count;
}

int MessageFormat::addMessages(const vector<BufferDesc_t>& list)
{
  // add list of messages
  int totalCount = 0;
  int i = 0;
  bool tryO2format = true;
  for (auto & data : list) {
    if (tryO2format) {
      if (o2::header::get<o2::header::DataHeader>(data.mP, data.mSize)) {
        return readO2Format(list, mBlockDescriptors, mHeartbeatHeader, mHeartbeatTrailer);
      }
      tryO2format = false;
    }
    if (data.mSize > 0) {
      unsigned nofEventHeaders=mListEvtData.size();
      int result = addMessage(data.mP, data.mSize);
      if (result >= 0)
        totalCount += result;
      else {
        cerr << "warning: no valid data blocks in message " << i << endl;
      }
    } else {
      cerr << "warning: ignoring message " << i << " with payload of size 0" << endl;
    }
  }
  return 0;
}

int MessageFormat::readBlockSequence(uint8_t* buffer, unsigned size,
                                     vector<BlockDescriptor>& descriptorList) const
{
  // read a sequence of blocks consisting of AliHLTComponentBlockData followed by payload
  // from a buffer
  if (buffer == nullptr) return 0;
  unsigned position = 0;
  vector<BlockDescriptor> input;
  while (position + sizeof(AliHLTComponentBlockData) < size) {
    AliHLTComponentBlockData* p = reinterpret_cast<AliHLTComponentBlockData*>(buffer + position);
    if (p->fStructSize == 0 ||                         // no valid header
        p->fStructSize + position > size ||            // no space for the header
        p->fStructSize + p->fSize + position > size) { // no space for the payload
      // the buffer is only a valid sequence of data blocks if payload
      // of the last block exacly matches the buffer boundary
      // otherwize all blocks added until now are ignored
      return -ENODATA;
    }
    // insert a new block
    input.emplace_back(*p);
    position += p->fStructSize;
    if (p->fSize > 0) {
      input.back().fPtr = buffer + position;
      position += p->fSize;
    } else {
      // Note: also a valid block, payload is optional
      input.back().fPtr = nullptr;
    }
    // offset always 0 for iput blocks
    input.back().fOffset = 0;
  }

  descriptorList.insert(descriptorList.end(), input.begin(), input.end());
  return input.size();
}

int MessageFormat::readHOMERFormat(uint8_t* buffer, unsigned size,
                                   vector<BlockDescriptor>& descriptorList) const
{
  // read message payload in HOMER format
  if (mpFactory == nullptr) const_cast<MessageFormat*>(this)->mpFactory = new o2::alice_hlt::HOMERFactory;
  if (buffer == nullptr || mpFactory == nullptr) return -EINVAL;
  unique_ptr<AliHLTHOMERReader> reader(mpFactory->OpenReaderBuffer(buffer, size));
  if (reader.get() == nullptr) return -ENOMEM;

  unsigned nofBlocks = 0;
  if (reader->ReadNextEvent() == 0) {
    nofBlocks = reader->GetBlockCnt();
    for (unsigned i = 0; i < nofBlocks; i++) {
      descriptorList.emplace_back(const_cast<void*>(reader->GetBlockData(i)), reader->GetBlockDataLength(i), kAliHLTVoidDataType, reader->GetBlockDataSpec(i));
      homer_uint64 id = byteSwap64(reader->GetBlockDataType(i));
      homer_uint32 origin = byteSwap32(reader->GetBlockDataOrigin(i));
      memcpy(&descriptorList.back().fDataType.fID, &id,
             sizeof(id) > kAliHLTComponentDataTypefIDsize ? kAliHLTComponentDataTypefIDsize : sizeof(id));
      memcpy(&descriptorList.back().fDataType.fOrigin, &origin, 
             sizeof(origin) > kAliHLTComponentDataTypefOriginSize ? kAliHLTComponentDataTypefOriginSize : sizeof(origin));
    }
  }

  return nofBlocks;
}

int MessageFormat::readO2Format(const vector<BufferDesc_t>& list, std::vector<BlockDescriptor>& descriptorList, HeartbeatHeader& hbh, HeartbeatTrailer& hbt) const
{
  int partNumber = 0;
  const o2::header::DataHeader* dh = nullptr;
  for (auto part : list) {
    if (!dh) {
      // new header - payload pair, read DataHeader
      dh = o2::header::get<o2::header::DataHeader>(part.mP, part.mSize);
      if (!dh) {
        cerr << "can not find DataHeader" << endl;
        return -ENOMSG;
      }
      // extract the heartbeat information if available and keep it to
      // envelope the data blocks in the output message
      if (dh->dataDescription == o2::header::gDataDescriptionHeartbeatFrame) {
        const HeartbeatFrameEnvelope* hbf = o2::header::get<HeartbeatFrameEnvelope>(part.mP, part.mSize);
        if (hbf) {
          hbh = hbf->header;
          hbt = hbf->trailer;
        } else {
          hbh.headerWord = hbt.trailerWord = 0;
        }
      }
    } else {
      if (dh->dataDescription == o2::header::gDataDescriptionHeartbeatFrame) {
        // this is pure O2 information, do not forward to HLT component
        dh = nullptr;
        continue;
      }
      auto ptr = part.mP;
      auto size = part.mSize;
      if (hbh) {
        // check if the data is enveloped in a heartbeat frame
        if (size > sizeof(HeartbeatHeader) + sizeof(HeartbeatTrailer)) {
          auto* datahbh = reinterpret_cast<const HeartbeatHeader*>(ptr);
          auto* datahbt = reinterpret_cast<const HeartbeatTrailer*>(ptr + size - sizeof(HeartbeatTrailer));
          if ((*datahbh) && (*datahbt)) {
            // valid header and trailer found, strip the block
            ptr += sizeof(HeartbeatHeader);
            size -= sizeof(HeartbeatHeader) + sizeof(HeartbeatTrailer);
          }
        }
      }
      descriptorList.emplace_back(ptr, size, *dh);
      dh = nullptr;
    }
  }
  if (dh) {
    cerr << "missing payload to the last header" << endl;
    return -ENOMSG;
  }
  return list.size()/2;
}

vector<MessageFormat::BufferDesc_t> MessageFormat::createMessages(const AliHLTComponentBlockData* blocks,
                                                                  unsigned count, unsigned totalPayloadSize,
                                                                  const AliHLTComponentEventData* evtData,
                                                                  boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate)
{
  // O2 output mode does not support event info struct
  // for the moment simply ignore it, not sure if this is the best
  // way to go, but this function is anyhow subject to change
  if (mOutputMode == kOutputModeO2 && evtData != nullptr) {
    evtData = nullptr;
  }
  //assert(mOutputMode != kOutputModeO2 || evtData == nullptr);

  const AliHLTComponentBlockData* pOutputBlocks = blocks;
  uint32_t outputBlockCnt = count;
  mDataBuffer.clear();
  mMessages.clear();
  if (mOutputMode == kOutputModeHOMER) {
    AliHLTHOMERWriter* pWriter = createHOMERFormat(pOutputBlocks, outputBlockCnt);
    if (pWriter) {
      uint32_t position = mDataBuffer.size();
      uint32_t offset = 0;
      uint32_t payloadSize = pWriter->GetTotalMemorySize();
      auto msgSize=payloadSize + (evtData != nullptr?sizeof(AliHLTComponentEventData):0);
      if (cbAllocate==nullptr) {
        // make the target in the internal buffer
        mDataBuffer.resize(position + msgSize);
      }
      auto pTarget = MakeTarget(msgSize, position, cbAllocate);
      if (evtData) {
        memcpy(pTarget + offset, evtData, sizeof(AliHLTComponentEventData));
        offset+=sizeof(AliHLTComponentEventData);
      }
      pWriter->Copy(pTarget + offset, 0, 0, 0, 0);
      mpFactory->DeleteWriter(pWriter);
      offset+=payloadSize;
      mMessages.emplace_back(pTarget, offset);
    }
  } else if (mOutputMode == kOutputModeMultiPart ||
             mOutputMode == kOutputModeSequence ||
             mOutputMode == kOutputModeO2) {
    // the output blocks are assempled in the internal buffer, for each
    // block BlockData is added as header information, directly followed
    // by the block payload
    //
    // kOutputModeMultiPart:
    // multi part mode adds one buffer descriptor per output block
    // the devices decides what to do with the multiple descriptors, one
    // option is to send them in a multi-part message
    //
    // kOutputModeSequence:
    // sequence mode concatenates the output blocks in the internal
    // buffer. In contrast to multi part mode, only one buffer descriptor
    // for the complete sequence is handed over to device
    //
    // kOutputModeO2
    // the O2 data format consisting of header-payload pairs
    uint32_t position = mDataBuffer.size();
    uint32_t offset = 0;
    unsigned bi = 0;
    const auto* pOutputBlock = pOutputBlocks;
    auto maxBufferSize = totalPayloadSize;
    if (mOutputMode == kOutputModeO2) {
      maxBufferSize += count * sizeof(o2::header::DataHeader);
      if (mHeartbeatHeader) {
        // one extra header for the generated HB information
        maxBufferSize += sizeof(o2::header::DataHeader) + sizeof(HeartbeatFrameEnvelope);
        // the payload of the additional block
        maxBufferSize += sizeof(o2::header::HeartbeatStatistics);
        // one heartbeat header-trailer pair per data block
        maxBufferSize += count * (sizeof(mHeartbeatHeader) + sizeof(mHeartbeatTrailer));
      }
    } else {
      maxBufferSize += sizeof(AliHLTComponentEventData) + count * sizeof(AliHLTComponentBlockData);
    }
    if (mDataBuffer.size() < position + maxBufferSize) {
      // make the target in the internal buffer, for simplicity data is copied
      // to a new buffer, a memmove would be possible to make room for block
      // descriptors
      // resize to the full size before using individual chunks of
      // this buffer to ensure, that all pointers are valid at the end.
      mDataBuffer.resize(position + maxBufferSize);
    }
    auto pTarget=&mDataBuffer[position];
    pTarget = nullptr;
    if (mOutputMode == kOutputModeO2 && mHeartbeatHeader) {
      // add additional heartbeat envelope block at the beginning
      // data header
      o2::header::DataHeader dh;
      o2::header::HeartbeatStatistics hbfPayload;
      dh.dataDescription = o2::header::gDataDescriptionHeartbeatFrame;
      dh.dataOrigin = o2::header::gDataOriginAny;
      dh.payloadSize = sizeof(hbfPayload);
      dh.subSpecification = 0;
      // make the stack
      o2::header::Stack headerMessage(dh, HeartbeatFrameEnvelope(mHeartbeatHeader, mHeartbeatTrailer));

      auto msgSize = headerMessage.size();
      pTarget = MakeTarget(msgSize, position, cbAllocate);
      memcpy(pTarget, headerMessage.data(), msgSize);
      mMessages.emplace_back(pTarget, msgSize);
      if (cbAllocate == nullptr) position += msgSize;
      msgSize = dh.payloadSize;
      pTarget = MakeTarget(msgSize, position, cbAllocate);
      memcpy(pTarget, &hbfPayload, msgSize);
      mMessages.emplace_back(pTarget, msgSize);
      if (cbAllocate == nullptr) position += msgSize;
    }
    unsigned msgSize = 0;
    do {
      if (bi == 0 ||
          mOutputMode == kOutputModeMultiPart ||
          mOutputMode == kOutputModeO2) {
        // request a new message buffer when entering for the first time in concatanate mode
        // and for every block in multi part mode
        // the actual size depends on mode and block index
        // - event data structure is only written at beginning of first
        //   message, regardsless of mode
        // - concatanate mode requests one big buffer for sequential sequence of
        //   all blocks
        msgSize = (bi==0 && evtData != nullptr?sizeof(AliHLTComponentEventData):0); // first message has event data
        if (count>0 && mOutputMode==kOutputModeMultiPart) {
          msgSize+=sizeof(AliHLTComponentBlockData) + pOutputBlock->fSize;
        } else if (mOutputMode == kOutputModeSequence) {
          msgSize+=count*sizeof(AliHLTComponentBlockData) + totalPayloadSize;
        } else if (mOutputMode == kOutputModeO2 ) {
          msgSize = sizeof(o2::header::DataHeader);
        }
        pTarget = MakeTarget(msgSize, position, cbAllocate);
        offset = 0;
      }

      if (bi==0 && evtData != nullptr) {
        // event data only in the first message in order to avoid increase of required
        // buffer size due to duplicated event header
        memcpy(pTarget + offset, evtData, sizeof(AliHLTComponentEventData));
        if (mOutputMode == kOutputModeMultiPart && evtData->fBlockCnt>1) {
          // in multipart mode, there is only one block per part
          // consequently, the number of blocks indicated in the event data header
          // does not reflect the number of blocks in this data sample. But it is
          // set to 1 to make the single message consistent
          auto* pEvtData = reinterpret_cast<AliHLTComponentEventData*>(pTarget + offset);
          pEvtData->fBlockCnt=1;
        }
        offset+=sizeof(AliHLTComponentEventData);
      }

      if (bi<count) {
        // copy BlockData and payload
        uint8_t* pData = reinterpret_cast<uint8_t*>(pOutputBlock->fPtr);
        pData += pOutputBlock->fOffset;
        auto* bdTarget = reinterpret_cast<AliHLTComponentBlockData*>(pTarget + offset);
        if (mOutputMode != kOutputModeO2) {
          assert(msgSize >= offset + sizeof(AliHLTComponentBlockData) + pOutputBlock->fSize);
          memcpy(bdTarget, pOutputBlock, sizeof(AliHLTComponentBlockData));
          bdTarget->fOffset = 0;
          bdTarget->fPtr = nullptr;
          offset += sizeof(AliHLTComponentBlockData);
          memcpy(pTarget + offset, pData, pOutputBlock->fSize);
          offset += pOutputBlock->fSize;
        }
        if (mOutputMode == kOutputModeMultiPart) {
          // send one descriptor per block back to device
          mMessages.emplace_back(pTarget, offset);
          if (cbAllocate == nullptr) position+=offset;
          offset = 0;
        } else if (mOutputMode == kOutputModeO2) {
          o2::header::DataHeader dh;
          dh.dataDescription.runtimeInit(pOutputBlock->fDataType.fID, kAliHLTComponentDataTypefIDsize);
          dh.dataOrigin.runtimeInit(pOutputBlock->fDataType.fOrigin, kAliHLTComponentDataTypefOriginSize);
          dh.payloadSize = pOutputBlock->fSize;
          dh.subSpecification = pOutputBlock->fSpecification;
          memcpy(pTarget, &dh, sizeof(o2::header::DataHeader));
          offset += sizeof(o2::header::DataHeader);
          mMessages.emplace_back(pTarget, offset);
          if (cbAllocate == nullptr) position+=offset;
          offset = 0;
          if (mHeartbeatHeader.headerWord == 0) {
            // no heartbeat information availalbe, send the buffer
            // as it is
            mMessages.emplace_back(pData, pOutputBlock->fSize);
          } else {
            // make the heartbeat frame
            msgSize = pOutputBlock->fSize + sizeof(mHeartbeatHeader) + sizeof(mHeartbeatTrailer);
            pTarget = MakeTarget(msgSize, position, cbAllocate);
            memcpy(pTarget + offset, &mHeartbeatHeader, sizeof(mHeartbeatHeader));
            offset += sizeof(mHeartbeatHeader);
            memcpy(pTarget + offset, pData, pOutputBlock->fSize);
            offset += pOutputBlock->fSize;
            memcpy(pTarget + offset, &mHeartbeatTrailer, sizeof(mHeartbeatTrailer));
            offset += sizeof(mHeartbeatTrailer);
            mMessages.emplace_back(pTarget, offset);
            if (cbAllocate == nullptr) position += offset;
            offset = 0;
          }
        }
        pOutputBlock++;
      }
    }
    while (++bi<count);
    if (mOutputMode == kOutputModeSequence || count==0) {
      // send one single descriptor for all concatenated blocks
      mMessages.emplace_back(pTarget, offset);
    }
  } else {
    // invalid output mode
    std::stringstream errorMsg;
    errorMsg << "invalid output mode: " << mOutputMode;
    throw std::runtime_error(errorMsg.str());
  }
  return mMessages;
}

uint8_t* MessageFormat::MakeTarget(unsigned size, unsigned position, boost::signals2::signal<unsigned char* (unsigned int)> *cbAllocate)
{
  // TODO: obviously this can be done in a better way, it's a bit of a hack
  // taking account for the limited lifetime of the wrapper device code
  uint8_t* pTarget = nullptr;
  if (cbAllocate==nullptr) {
    if (mDataBuffer.size() < position + size) {
      throw std::runtime_error("complete buffer allocation must be done in the first cycle");
    }
    pTarget=&mDataBuffer[position];
  } else {
    // use callback to create target
    pTarget=*(*cbAllocate)(size);
    if (pTarget==nullptr) {
      throw std::bad_alloc();
    }
  }
  return pTarget;
}

AliHLTHOMERWriter* MessageFormat::createHOMERFormat(const AliHLTComponentBlockData* pOutputBlocks,
                                                    uint32_t outputBlockCnt) const
{
  // send data blocks in HOMER format in one message
  int iResult = 0;
  if (mpFactory == nullptr) const_cast<MessageFormat*>(this)->mpFactory = new o2::alice_hlt::HOMERFactory;
  if (!mpFactory) return nullptr;
  unique_ptr<AliHLTHOMERWriter> writer(mpFactory->OpenWriter());
  if (writer.get() == nullptr) return nullptr;

  homer_uint64 homerHeader[kCount_64b_Words];
  HOMERBlockDescriptor homerDescriptor(homerHeader);

  const AliHLTComponentBlockData* pOutputBlock = pOutputBlocks;
  for (unsigned blockIndex = 0; blockIndex < outputBlockCnt; blockIndex++, pOutputBlock++) {
    if (pOutputBlock->fPtr == nullptr && pOutputBlock->fSize > 0) {
      cerr << "warning: ignoring block " << blockIndex << " because of missing data pointer" << endl;
    }
    memset(homerHeader, 0, sizeof(homer_uint64) * kCount_64b_Words);
    homerDescriptor.Initialize();
    homer_uint64 id = 0;
    homer_uint64 origin = 0;
    memcpy(&id, pOutputBlock->fDataType.fID, sizeof(homer_uint64));
    memcpy(((uint8_t*)&origin) + sizeof(homer_uint32), pOutputBlock->fDataType.fOrigin, sizeof(homer_uint32));
    homerDescriptor.SetType(byteSwap64(id));
    homerDescriptor.SetSubType1(byteSwap64(origin));
    homerDescriptor.SetSubType2(pOutputBlock->fSpecification);
    homerDescriptor.SetBlockSize(pOutputBlock->fSize);
    writer->AddBlock(homerHeader, pOutputBlock->fPtr);
  }
  return writer.release();
}

int MessageFormat::insertEvtData(const AliHLTComponentEventData& evtData)
{
  // insert event header to list, sort by time, oldest first
  if (mListEvtData.size()==0) {
    mListEvtData.emplace_back(evtData);
  } else {
    auto it=mListEvtData.begin();
    for (; it!=mListEvtData.end(); it++) {
      if ((it->fEventCreation_us*1e3 + it->fEventCreation_us/1e3)>
          (evtData.fEventCreation_us*1e3 + evtData.fEventCreation_us/1e3)) {
        // found a younger element
        break;
      }
    }
    // TODO: simple logic at the moment, header is not inserted
    // if there is a mismatch, as the headers are inserted one by one, all
    // headers in the list have the same ID
    if (it != mListEvtData.end() &&
        evtData.fEventID!=it->fEventID) {
      cerr << "Error: mismatching event ID " << evtData.fEventID
           << ", expected " << it->fEventID
           << " for event with timestamp "
           << evtData.fEventCreation_us*1e3 + evtData.fEventCreation_us/1e3 << " ms"
           << endl;
      return -1;
    }
    // insert before the younger element
    mListEvtData.insert(it, evtData);
  }
  return 0;
}

uint64_t MessageFormat::byteSwap64(uint64_t src) const
{
  // swap a 64 bit number
  return ((src & 0xFFULL) << 56) | 
    ((src & 0xFF00ULL) << 40) | 
    ((src & 0xFF0000ULL) << 24) | 
    ((src & 0xFF000000ULL) << 8) | 
    ((src & 0xFF00000000ULL) >> 8) | 
    ((src & 0xFF0000000000ULL) >> 24) | 
    ((src & 0xFF000000000000ULL) >>  40) | 
    ((src & 0xFF00000000000000ULL) >> 56);
}

uint32_t MessageFormat::byteSwap32(uint32_t src) const
{
  // swap a 32 bit number
  return ((src & 0xFFULL) << 24) | 
    ((src & 0xFF00ULL) << 8) | 
    ((src & 0xFF0000ULL) >> 8) | 
    ((src & 0xFF000000ULL) >> 24);
}
