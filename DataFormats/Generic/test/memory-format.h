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

/// @file   memory_format.h
/// @author Matthias Richter
/// @since  2016-01-28
/// @brief  Helper structs for the ALICE O2 generic format API test
/// @note   DO NOT USE OUTSIDE UNIT TEST OF FORMAT API
///         This definitions have been a first draft during discussion of
///         in memory data formats, the final header file has been placed
///         elsewhere, but this file is temporarily kept for the unit test

// use the standard definitions of int variables
#include <cstdint>
#include "header_versions.h"

/**
The defined headers are a tradeoff to provide the necessary information in a lightweight way and to allow for evolution and compatibility.

General header format
- Starts with basic header information, never serialized, with unique version number
- Strict policy enforced: no changes to members (e.g. width) or sequence of members
- New members can be appended
- All basic header structs are defined with fixed endianess and padding
- Header-stack concept: optional headers can follow the basic header
*/
namespace o2 {
namespace Format {
  /**
   * Data header to be commonly used for all in-memory data blocks
   *
   * Unique header version; struct size included for consistency check
   * and to facilitate later implementation of conversion handlers.
   *
   * A magic string makes identification of header simpler, e.g. after
   * a data corruption; great help for low level debugging
   *
   * PayloadSize is a redundant information, to be used for integrity
   * check and mandatory for disk dumped data
   *
   * Payload serialization method defined in the header, allows to build
   * common functionality. Framework can choose the right tool for
   * de-serialization
   */
  struct  DataHeader_t {
    /** 4 bytes of a magic string */
    int32_t  mMagicString;
    /** size of the struct */
    int32_t  mStructSize;
    /** header version, bookkeeping in the software */
    int32_t  mHeaderVersion;
    /** Flags field, valid bits and their meaning defined by the header version */
    int32_t  mFlags;
    /** size of payload in memory */
    int64_t  mPayloadSize;
    /** payload serialization method for transfer */
    char mPayloadSerializationMethod[7+1];
    /** Payload meta data: Subsystem or detector */
    char mDataOrigin[3+1];
    /** Payload meta data: Data description, e.g. raw, clusters, tracks */
    char mDataDescriptor[15+1];
    /** Payload meta data: A system or detector specific sub specification */
    int64_t mSubSpec;
  };

  /**
   * Helper struct for the payload meta data
   *
   * This struct is an addition to DataHeader_t to allow implementation
   * of operators. All meta data members are directly included in DataHeader_t
   * for easier access and consistency.
   */
  struct PayloadMetaData_t {
    /** Subsystem or detector */
    char mDataOrigin[3+1];
    /** Data description, e.g. raw, clusters, tracks */
    char mDataDescriptor[15+1];
    /** A system or detector specific sub specification */
    int64_t mSubSpec;
  };

  /**
   * Header-stack:: optional headers can follow the basic header
   * A next header is indicated in the flag member of preceeding header
   * Optional headers consist of a fixed NextHeaderDescription and a variable
   * NextHeaderContent
   */
  struct NextHeaderDescription_t {
    /** size of this next header description */
    int32_t mStructSize;
    /** size of the next header payload */
    int32_t mNextHeaderContentSize;
    /** Common flags for all next-headers, includes next-header flag */
    int32_t mFlags;
    /** Descriptor */
    char mHeaderDescriptor[15+1];
    /** serialization method */
    char mSerializationMethod[7+1];
  };

}; // namespace Format
}; // namespace AliceO2
#endif
