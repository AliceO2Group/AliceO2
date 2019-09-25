// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// XEMacs -*-C++-*-
#ifndef ALIHLTHOMERDATA_H
#define ALIHLTHOMERDATA_H

//  @file   AliHLTHOMERData.h
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  Original AliHLTHOMERData.h of AliRoot included for HOMER
//          support in the ALFA project

//************************************************************************
//*
//*
//* This file is property of and copyright by the Technical Computer
//* Science Group, Kirchhoff Institute for Physics, Ruprecht-Karls-
//* University, Heidelberg, Germany, 2001
//* This file has been written by Timm Morten Steinbeck,
//* timm@kip.uni-heidelberg.de
//*
//*
//* See the file license.txt for details regarding usage, modification,
//* distribution and warranty.
//* Important: This file is provided without any warranty, including
//* fitness for any particular purpose.
//*
//*
//* Newer versions of this file's package will be made available from
//* http://web.kip.uni-heidelberg.de/Hardwinf/L3/
//* or the corresponding page of the Heidelberg Alice Level 3 group.
//*
//************************************************************************/

/*
***************************************************************************
**
** $Author$ - Initial Version by Timm Morten Steinbeck
**
** $Id$ 
**
***************************************************************************
*/

#ifdef USE_ROOT
#include "Rtypes.h"
#endif
#include <climits>

// Determine the sizes of the different integer type
// homer_uint32, homer_uint64
#if !defined(USE_ROOT)
// First homer_uint32
#if USHRT_MAX == 4294967295
typedef unsigned short homer_uint32;
#else // USHRT_MAX==4294967295

#if UINT_MAX == 4294967295
typedef unsigned int homer_uint32;
#else // UINT_MAX==4294967295

#if ULONG_MAX == 4294967295l
typedef unsigned long homer_uint32;
#else // ULONG_MAX==4294967295l

#error Could not typedef homer_uint32

#endif // ULONG_MAX==4294967295l

#endif // UINT_MAX==4294967295

#endif // USHRT_MAX==4294967295

// Then homer_uint16
#if USHRT_MAX == 65535
typedef unsigned short homer_uint16;
#else // USHRT_MAX==65535

#if UINT_MAX == 65535
typedef unsigned int homer_uint16;
#else // UINT_MAX==65535

#if ULONG_MAX == 65535
typedef unsigned long homer_uint16;
#else // ULONG_MAX==65535

#error Could not typedef homer_uint16

#endif // ULONG_MAX==65535

#endif // UINT_MAX==65535

#endif // USHRT_MAX==65535

// Then homer_uint64
#if ULONG_MAX == 18446744073709551615UL
typedef unsigned long homer_uint64;
#else // ULONG_MAX==18446744073709551615UL

#if defined __GNUC__
typedef unsigned long long homer_uint64;
#else // defined __GNUC__

#if defined __SUNPRO_CC
typedef unsigned long long homer_uint64;
#else // defined __SUNPRO_CC

#error Could not typedef homer_uint64

#endif // defined __SUNPRO_CC
#endif // defined __GNUC__

#endif // ULONG_MAX==18446744073709551615UL

typedef unsigned char homer_uint8;

#else // !USE_ROOT

typedef UShort_t homer_uint16;
typedef UInt_t homer_uint32;
typedef ULong64_t homer_uint64;
typedef Byte_t homer_uint8;

#ifdef __CINT__
typedef int key_t;
#endif

#endif // USE_ROOT

//typedef homer_uint64 AliEventID_t;

#define kAttribute_8b_StartOffset 0
#define kByteOrderAttribute_8b_Offset kAttribute_8b_StartOffset + 0
#define kVersionAttribute_8b_Offset kAttribute_8b_StartOffset + 1
#define kAlignment_8b_StartOffset 24
#define kUInt64Alignment_8b_Offset kAlignment_8b_StartOffset + 0
#define kUInt32Alignment_8b_Offset kAlignment_8b_StartOffset + 1
#define kUInt16Alignment_8b_Offset kAlignment_8b_StartOffset + 2
#define kUInt8Alignment_8b_Offset kAlignment_8b_StartOffset + 3
#define kDoubleAlignment_8b_Offset kAlignment_8b_StartOffset + 4
#define kFloatAlignment_8b_Offset kAlignment_8b_StartOffset + 5

#define kID_64b_Offset 1
#define kLength_64b_Offset 2
#define kType_64b_Offset 4
#define kSubType1_64b_Offset 5
#define kSubType2_64b_Offset 6
#define kBirth_s_64b_Offset 7
#define kBirth_us_64b_Offset 8
#define kProducerNode_64b_Offset 9
#define kOffset_64b_Offset 10
#define kSize_64b_Offset 11
#define kStatusFlags_64b_Offset 12
#define kEnd_64b_Offset 13
#define kCount_64b_Words kEnd_64b_Offset

// Possible values for fAttributes[kByteOrderAttribute]
/* Keep this consistent with BCLNetworkData.hpp kLittleEndian/kBigEndian and AliHLTSubEventDataDescriptor.hpp */
const homer_uint8 kHOMERUnknownByteOrder = 0;
const homer_uint8 kHOMERLittleEndianByteOrder = 1;
const homer_uint8 kHOMERBigEndianByteOrder = 2;
#ifdef __i386__
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#else
#ifdef __arm__
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#else
#ifdef __x86_64__
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#else
#ifdef __ia64__
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#else
#if defined(__powerpc__)
const homer_uint8 kHOMERNativeByteOrder = kHOMERBigEndianByteOrder;
#else
#ifdef __CINT__
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#warning Assuming little endian format for __CINT__
#else
const homer_uint8 kHOMERNativeByteOrder = kHOMERLittleEndianByteOrder;
#warning Assuming little endian format for unknown architecture
//#error Byte format (little/big endian) currently not defined for platforms other than intel i386 compatible, x86-64 (AMD64) and arm...
#endif
#endif
#endif
#endif
#endif
#endif

//#define HOMER_BLOCK_DESCRIPTOR_TYPEID             (((homer_uint64)'HOBL')<<32 | 'KDES')
#define HOMER_BLOCK_DESCRIPTOR_TYPEID ((((homer_uint64)'H') << 56) | (((homer_uint64)'O') << 48) | (((homer_uint64)'B') << 40) | (((homer_uint64)'L') << 32) | (((homer_uint64)'K') << 24) | (((homer_uint64)'D') << 16) | (((homer_uint64)'E') << 8) | (((homer_uint64)'S') << 0))
//#define HOMER_BLOCK_DESCRIPTOR_TYPEID            ( (((homer_uint64)'H')<<56)|(((homer_uint64)'O')<<48) )
#define HOMER_HEADER_CURRENT_VERSION 2

class AliHLTHOMERBlockDescriptor
{
 public:
  static unsigned GetHOMERBlockDescriptorSize()
  {
    return sizeof(homer_uint64) * kCount_64b_Words;
  }

  AliHLTHOMERBlockDescriptor(void* header = nullptr)
    : fHeader(header)
  {
  }
  void UseHeader(void* header)
  {
    fHeader = header;
  }
  void Initialize()
  {
    if (fHeader) {
      for (unsigned ii = 0; ii < kCount_64b_Words; ii++)
        ((homer_uint64*)fHeader)[ii] = (homer_uint64)0;
      ((homer_uint64*)fHeader)[kID_64b_Offset] = HOMER_BLOCK_DESCRIPTOR_TYPEID;
      ((homer_uint64*)fHeader)[kLength_64b_Offset] = GetHOMERBlockDescriptorSize();
      ((homer_uint8*)fHeader)[kByteOrderAttribute_8b_Offset] = kHOMERNativeByteOrder;
      ((homer_uint8*)fHeader)[kVersionAttribute_8b_Offset] = HOMER_HEADER_CURRENT_VERSION;
    }
  }

  void SetByteOrder(homer_uint8 bo) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kByteOrderAttribute_8b_Offset] = bo;
  }
  homer_uint8 GetByteOrder() const
  {
    if (fHeader)
      return ((homer_uint8*)fHeader)[kByteOrderAttribute_8b_Offset];
    return 0xFF;
  }
  void SetVersion(homer_uint8 v) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kVersionAttribute_8b_Offset] = v;
  }
  void SetID(homer_uint64 id) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kID_64b_Offset] = id;
  }
  void SetHeaderLength(homer_uint64 l) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kLength_64b_Offset] = l;
  }
  homer_uint64 GetHeaderLength() const
  {
    if (fHeader)
      return ((homer_uint64*)fHeader)[kLength_64b_Offset];
    return 0;
  }
  void SetAlignment(homer_uint8 type, homer_uint8 align) const
  {
    if (fHeader && type < 6)
      ((homer_uint8*)fHeader)[kAlignment_8b_StartOffset + type] = align;
  }
  void SetUInt64Alignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kUInt64Alignment_8b_Offset] = align;
  }
  void SetUInt32Alignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kUInt32Alignment_8b_Offset] = align;
  }
  void SetUInt16Alignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kUInt16Alignment_8b_Offset] = align;
  }
  void SetUInt8Alignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kUInt8Alignment_8b_Offset] = align;
  }
  void SetDoubleAlignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kDoubleAlignment_8b_Offset] = align;
  }
  void SetFloatAlignment(homer_uint8 align) const
  {
    if (fHeader)
      ((homer_uint8*)fHeader)[kFloatAlignment_8b_Offset] = align;
  }
  void SetType(homer_uint64 t) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kType_64b_Offset] = t;
  }
  void SetSubType1(homer_uint64 st1) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kSubType1_64b_Offset] = st1;
  }
  void SetSubType2(homer_uint64 st2) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kSubType2_64b_Offset] = st2;
  }
  void SetBirth_s(homer_uint64 bs) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kBirth_s_64b_Offset] = bs;
  }
  void SetBirth_us(homer_uint64 bus) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kBirth_us_64b_Offset] = bus;
  }
  void SetProducerNode(homer_uint64 pn) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kProducerNode_64b_Offset] = pn;
  }
  void SetBlockOffset(homer_uint64 bo) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kOffset_64b_Offset] = bo;
  }
  homer_uint64 GetBlockOffset() const
  {
    if (fHeader)
      return ((homer_uint64*)fHeader)[kOffset_64b_Offset];
    return 0;
  }
  void SetBlockSize(homer_uint64 bs) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kSize_64b_Offset] = bs;
  }
  homer_uint64 GetBlockSize() const
  {
    if (fHeader)
      return ((homer_uint64*)fHeader)[kSize_64b_Offset];
    return 0;
  }
  void SetStatusFlags(homer_uint64 bs) const
  {
    if (fHeader)
      ((homer_uint64*)fHeader)[kStatusFlags_64b_Offset] = bs;
  }
  homer_uint64 GetStatusFlags() const
  {
    if (fHeader)
      return ((homer_uint64*)fHeader)[kStatusFlags_64b_Offset];
    return 0;
  }

  void* GetHeader() const
  {
    return fHeader;
  }

 protected:
  void* fHeader; //! transient

 private:
  /** copy constructor prohibited */
  AliHLTHOMERBlockDescriptor(const AliHLTHOMERBlockDescriptor&);
  /** assignment operator prohibited */
  AliHLTHOMERBlockDescriptor& operator=(const AliHLTHOMERBlockDescriptor&);
};

// the HOMERBlockDescriptor is used in the code
typedef class AliHLTHOMERBlockDescriptor HOMERBlockDescriptor;

/*
***************************************************************************
**
** $Author$ - Initial Version by Timm Morten Steinbeck
**
** $Id$ 
**
***************************************************************************
*/

#endif // ALIHLTHOMERDATA_H
