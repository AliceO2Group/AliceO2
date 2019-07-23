// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_EMCALBLOCKHEADER
#define O2_EMCALBLOCKHEADER

#include "Headers/DataHeader.h"

namespace o2
{

namespace emcal
{

/// @struct EMCALBlockHeader
/// @brief Header for EMCAL flagging the following EMCAL payload
struct EMCALBlockHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "EMCBlkH";
  static const uint32_t sVersion = 1;

  EMCALBlockHeader(bool hasPayload) : BaseHeader(sizeof(EMCALBlockHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), mHasPayload(hasPayload) {}

  bool mHasPayload; ///< Field specifying whether the data block has payload
};

} // namespace emcal
} // namespace o2

#endif // O2_EMCALBLOCKHEADER
