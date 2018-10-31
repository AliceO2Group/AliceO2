// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPCSECTORHEADER_H
#define O2_TPCSECTORHEADER_H

#include "Headers/DataHeader.h"
#include "DataFormatsTPC/Constants.h"

namespace o2
{
namespace TPC
{

/// @struct TPCSectorHeader
/// TPC specific header to be transported on the header stack
struct TPCSectorHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "TPCSectH";
  static const uint32_t sVersion = 1;

  TPCSectorHeader(int s)
    : BaseHeader(sizeof(TPCSectorHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), sector(s)
  {
  }

  static constexpr int NSectors = o2::TPC::Constants::MAXSECTOR;
  int sector;
  union {
    uint64_t activeSectorsFlags = 0;
    struct {
      uint64_t activeSectors : NSectors;
      uint64_t unused : 12;
      uint64_t flags : 16;
    };
  };
};
} // namespace TPC
} // namespace o2

#endif // O2_TPCSECTORHEADER_H
