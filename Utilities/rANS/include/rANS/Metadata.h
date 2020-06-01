// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Metadata.h
/// @brief  A version header and metadata stored with compressed data

#ifndef RANS_METADATA_H
#define RANS_METADATA_H

#include <Rtypes.h>

namespace o2
{
namespace rans
{

struct Header {
  uint8_t majorVersion;
  uint8_t minorVersion;

  void clear() { majorVersion = minorVersion = 0; }
  ClassDefNV(Header, 2);
};

struct Metadata {
  enum OptStore : uint8_t { // describe how the store the data described by this metadata
    EENCODE,                // entropy encoding applied
    ROOTCompression,        // original data repacked to array with slot-size = streamSize and saved with root compression
    NONE                    // original data repacked to array with slot-size = streamSize and saved w/o compression
  };
  size_t messageLength = 0;
  uint8_t coderType = 0;
  uint8_t streamSize = 0;
  uint8_t probabilityBits = 0;
  OptStore opt = EENCODE;
  int32_t min = 0;
  int32_t max = 0;
  int nDictWords = 0;
  int nDataWords = 0;

  void clear()
  {
    min = max = 0;
    messageLength = 0;
    coderType = 0;
    streamSize = 0;
    probabilityBits = 0;
    nDataWords = nDictWords = 0;
  }
  ClassDefNV(Metadata, 2);
};

} // namespace rans
} // namespace o2

#endif
