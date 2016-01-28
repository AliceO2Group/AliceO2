#ifndef MEMORY_FORMAT_H
#define MEMORY_FORMAT_H
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

//  @file   memory_format.h
//  @author Matthias Richter
//  @since  2016-01-28
//  @brief  Primitives for the ALICE O2 in-memory format

// use the standard definitions of int variables
#include <stdint.h>

namespace AliceO2 {
namespace Format {
  /**
   * Basic struct for the data block meta data
   *
   * In total 24 byte are used per data block to descibe meta data
   */
  struct BlockMetaData_t {
    /** Size and version of the struct */
    int16_t mStructSize;
    /** Subsystem or detector */
    char mDataOrigin[3+1]; // find if there is a consistent 3-letter naming scheme in ALICE, enforce 0 at the end
    /** Main specification, e.g. raw, clusters, tracks */
    char mDataDescriptor[8];
    /** A system or detector specific sub specification */
    int32_t mSubSpec;
  };

  /**
   * Block data header to be commonly used for all in-memory data blocks
   *
   * To be decided: use unions to fit different members for transfer and in-memory?
   *
   * The struct size could be handled by the transport layer in the multi-header
   * approach, having the size as member keeps the possibility of sub headers
   */
  struct BlockData_t
  {
    /** 4 bytes of a magic string */
    int32_t mMagicString;
    /** size and version of the struct */
    int16_t mStructSize;
    /** flags, one of the bits to indicate that a sub header is following */
    int16_t mFlags;
    /** size of the data block in memory */
    int64_t mPayloadSize; // keep it for storage on disk
    /** header serialization method for transfer */
    int64_t mHeaderSerializationMethod; // use string
    /** payload serialization method for transfer */
    int64_t mPayloadSerializationMethod; // use string
    /** data block meta data: type, origin, specification */
    BlockMetaData_t mMetaData;
  };

}; // namespace Format
}; // namespace AliceO2
#endif
