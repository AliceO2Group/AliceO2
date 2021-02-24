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
/// \brief Definition of the HMPID raw data format

#ifndef ALICEO2_HMP_DATAFORMAT_H
#define ALICEO2_HMP_DATAFORMAT_H

#include <stdint.h>

namespace o2
{
namespace hmpid
{
namespace raw
{

/** generic word *

  struct Word_t {
    uint32_t undefined : 31;
    uint32_t wordType : 1;
  };

  union Union_t {
    uint32_t data;
    Word_t word;
    CrateHeader_t crateHeader;
    FrameHeader_t frameHeader;
    PackedHit_t packedHit;
    CrateTrailer_t crateTrailer;
  };
*/
} // namespace raw
} // namespace hmpid
} // namespace o2
#endif
