// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataFormat.h
/// \brief Definition of the TOF raw data format

#ifndef ALICEO2_TOF_DATAFORMAT_H
#define ALICEO2_TOF_DATAFORMAT_H

#include <stdint.h>

namespace o2
{
namespace tof
{
namespace compressed
{
/** generic word **/

struct Word_t {
  uint32_t undefined : 31;
  uint32_t wordType : 1;
};

/** data format **/

struct CrateHeader_t {
  uint32_t bunchID : 12;
  uint32_t eventCounter : 12;
  uint32_t drmID : 7;
  uint32_t mustBeOne : 1;
};

struct FrameHeader_t {
  uint32_t numberOfHits : 16;
  uint32_t frameID : 8;
  uint32_t trmID : 4;
  uint32_t deltaBC : 3;
  uint32_t mustBeZero : 1;
};

struct PackedHit_t {
  uint32_t tot : 11;
  uint32_t time : 13;
  uint32_t channel : 3;
  uint32_t tdcID : 4;
  uint32_t chain : 1;
};

struct CrateTrailer_t {
  uint32_t trmFault03 : 3;
  uint32_t trmFault04 : 3;
  uint32_t trmFault05 : 3;
  uint32_t trmFault06 : 3;
  uint32_t trmFault07 : 3;
  uint32_t trmFault08 : 3;
  uint32_t trmFault09 : 3;
  uint32_t trmFault10 : 3;
  uint32_t trmFault11 : 3;
  uint32_t trmFault12 : 3;
  uint32_t crateFault : 1;
  uint32_t mustBeOne : 1;
};

/** union **/

union Union_t {
  uint32_t data;
  Word_t word;
  CrateHeader_t crateHeader;
  FrameHeader_t frameHeader;
  PackedHit_t packedHit;
  CrateTrailer_t crateTrailer;
};
} // namespace compressed

namespace raw
{
// to be filled

} // namespace raw
} // namespace tof
} // namespace o2
#endif
