// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_CPVBLOCKHEADER
#define O2_CPVBLOCKHEADER

#include "Headers/DataHeader.h"

namespace o2
{

namespace cpv
{

/// @struct CPVBlockHeader
/// @brief Header for CPV flagging the following CPV payload
struct CPVBlockHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "CPVBlkH";
  static const uint32_t sVersion = 1;

  CPVBlockHeader(bool hasPayload) : BaseHeader(sizeof(CPVBlockHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), mHasPayload(hasPayload) {}

  bool mHasPayload; ///< Field specifying whether the data block has payload
};

} // namespace cpv
} // namespace o2

#endif // O2_CPVBLOCKHEADER
