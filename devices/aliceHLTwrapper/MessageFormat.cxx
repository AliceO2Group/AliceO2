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

//  @file   MessageFormat.cxx
//  @author Matthias Richter
//  @since  2014-12-11
//  @brief  Helper class for message format of ALICE HLT data blocks

#include "MessageFormat.h"
#include "HOMERFactory.h"
#include "AliHLTHOMERData.h"
#include "AliHLTHOMERWriter.h"
#include "AliHLTHOMERReader.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>

using namespace AliceO2::AliceHLT;
using namespace ALICE::HLT;

// TODO: central logging to be implemented

MessageFormat::MessageFormat()
  : mBlockDescriptors()
  , mDataBuffer()
  , mMessages()
  , mpFactory(NULL)
{
}

MessageFormat::~MessageFormat()
{
  if (mpFactory)
    delete mpFactory;
  mpFactory=NULL;
}

int MessageFormat::AddMessage(AliHLTUInt8_t* buffer, unsigned size)
{
  // add message
  // this will extract the block descriptors from the message
  // the descriptors refer to data in the original message buffer

  unsigned count=mBlockDescriptors.size();
  if (ReadBlockSequence(buffer, size, mBlockDescriptors)<0) {
    // not in the format of a single block, check if its a HOMER block
    if (ReadHOMERFormat(buffer, size, mBlockDescriptors)<0) {
      // not in HOMER format either, ignore the input
      // TODO: decide if that should be an error
    }
  }

  return mBlockDescriptors.size()-count;
}

int MessageFormat::AddMessages(const vector<BufferDesc_t>& list)
{
  // add list of messages
  int totalCount=0;
  int i=0;
  for (vector<BufferDesc_t>::const_iterator data=list.begin();
       data!=list.end(); data++, i++) {
    if (data->mSize>0) {
      int result=AddMessage(data->mP, data->mSize);
      if (result>0)
	totalCount+=result;
      else if (result==0) {
	cerr << "warning: no valid data blocks in message " << i << endl;
      }
    } else {
      cerr << "warning: ignoring message " << i << " with payload of size 0" << endl;
    }
  }
}

int MessageFormat::ReadBlockSequence(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& descriptorList) const
{
  // read a sequence of blocks consisting of AliHLTComponentBlockData followed by payload
  // from a buffer
  if (buffer==NULL) return 0;
  unsigned position=0;
  vector<AliHLTComponentBlockData> input;
  while (position+sizeof(AliHLTComponentBlockData)<size) {
    AliHLTComponentBlockData* p=reinterpret_cast<AliHLTComponentBlockData*>(buffer+position);
    if (p->fStructSize==0 ||                     // no valid header
	p->fStructSize+position>size ||          // no space for the header
	p->fStructSize+p->fSize+position>size) { // no space for the payload
      // the buffer is only a valid sequence of data blocks if payload
      // of the last block exacly matches the buffer boundary
      // otherwize all blocks added until now are ignored
      return -ENODATA;
    }
    // insert a new block
    input.push_back(*p);
    position+=p->fStructSize;
    if (p->fSize>0) {
      input.back().fPtr=buffer+position;
      position+=p->fSize;
    } else {
      // Note: also a valid block, payload is optional
      input.back().fPtr=NULL;
    }
    // offset always 0 for iput blocks
    input.back().fOffset=0;
  }

  descriptorList.insert(descriptorList.end(), input.begin(), input.end());
  return input.size();
}

int MessageFormat::ReadHOMERFormat(AliHLTUInt8_t* buffer, unsigned size, vector<AliHLTComponentBlockData>& descriptorList) const
{
  // read message payload in HOMER format
  if (mpFactory==NULL) 
    const_cast<MessageFormat*>(this)->mpFactory=new ALICE::HLT::HOMERFactory;
  if (buffer==NULL || mpFactory==NULL) return -EINVAL;
  auto_ptr<AliHLTHOMERReader> reader(mpFactory->OpenReaderBuffer(buffer, size));
  if (reader.get()==NULL) return -ENOMEM;

  unsigned nofBlocks=0;
  if (reader->ReadNextEvent()==0) {
    nofBlocks=reader->GetBlockCnt();
    for (unsigned i=0; i<nofBlocks; i++) {
      AliHLTComponentBlockData block;
      memset(&block, 0, sizeof(AliHLTComponentBlockData));
      block.fStructSize=sizeof(AliHLTComponentBlockData);
      block.fDataType.fStructSize=sizeof(AliHLTComponentDataType);
      homer_uint64 id=ByteSwap64(reader->GetBlockDataType( i ));
      homer_uint32 origin=ByteSwap32(reader->GetBlockDataOrigin( i ));
      memcpy(&block.fDataType.fID, &id, sizeof(id)>kAliHLTComponentDataTypefIDsize?kAliHLTComponentDataTypefIDsize:sizeof(id));
      memcpy(&block.fDataType.fOrigin, &origin, sizeof(origin)>kAliHLTComponentDataTypefOriginSize?kAliHLTComponentDataTypefOriginSize:sizeof(origin));
      block.fSpecification=reader->GetBlockDataSpec( i );
      block.fPtr=const_cast<void*>(reader->GetBlockData( i ));
      block.fSize=reader->GetBlockDataLength( i );
      descriptorList.push_back(block);
    }
  }

  return nofBlocks;
}

AliHLTHOMERWriter* MessageFormat::CreateHOMERFormat(AliHLTComponentBlockData* pOutputBlocks, AliHLTUInt32_t outputBlockCnt) const
{
  // send data blocks in HOMER format in one message
  int iResult=0;
  if (mpFactory==NULL) 
    const_cast<MessageFormat*>(this)->mpFactory=new ALICE::HLT::HOMERFactory;
  if (!mpFactory) return NULL;
  auto_ptr<AliHLTHOMERWriter> writer(mpFactory->OpenWriter());
  if (writer.get()==NULL) return NULL;

  homer_uint64 homerHeader[kCount_64b_Words];
  HOMERBlockDescriptor homerDescriptor(homerHeader);

  AliHLTComponentBlockData* pOutputBlock=pOutputBlocks;
  for (unsigned blockIndex=0; blockIndex<outputBlockCnt; blockIndex++, pOutputBlock++) {
    if (pOutputBlock->fPtr==NULL && pOutputBlock->fSize>0) {
      cerr << "warning: ignoring block " << blockIndex << " because of missing data pointer" << endl;
    }
    memset( homerHeader, 0, sizeof(homer_uint64)*kCount_64b_Words );
    homerDescriptor.Initialize();
    homer_uint64 id=0;
    homer_uint64 origin=0;
    memcpy(&id, pOutputBlock->fDataType.fID, sizeof(homer_uint64));
    memcpy(((AliHLTUInt8_t*)&origin)+sizeof(homer_uint32), pOutputBlock->fDataType.fOrigin, sizeof(homer_uint32));
    homerDescriptor.SetType(ByteSwap64(id));
    homerDescriptor.SetSubType1(ByteSwap64(origin));
    homerDescriptor.SetSubType2(pOutputBlock->fSpecification);
    homerDescriptor.SetBlockSize(pOutputBlock->fSize);
    writer->AddBlock(homerHeader, pOutputBlock->fPtr);
  }
  return writer.release();
}

AliHLTUInt64_t MessageFormat::ByteSwap64(AliHLTUInt64_t src) const
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

AliHLTUInt32_t MessageFormat::ByteSwap32(AliHLTUInt32_t src) const
{
  // swap a 32 bit number
  return ((src & 0xFFULL) << 24) | 
    ((src & 0xFF00ULL) << 8) | 
    ((src & 0xFF0000ULL) >> 8) | 
    ((src & 0xFF000000ULL) >> 24);
}
