// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_PHOSBLOCKHEADER
#define O2_PHOSBLOCKHEADER

#include "Headers/DataHeader.h"

namespace o2
{

namespace phos
{

/// @struct PHOSBlockHeader
/// @brief Header for PHOS flagging the following PHOS payload
struct PHOSBlockHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "PHSBlkH";
  static const uint32_t sVersion = 1;

  PHOSBlockHeader(bool hasPayload) : BaseHeader(sizeof(PHOSBlockHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), mHasPayload(hasPayload) {}

  bool mHasPayload; ///< Field specifying whether the data block has payload
};

} // namespace phos
} // namespace o2

#endif // O2_PHOSBLOCKHEADER
