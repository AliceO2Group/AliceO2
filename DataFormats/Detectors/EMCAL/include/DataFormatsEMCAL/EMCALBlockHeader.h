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

namespace o2 {

namespace emcal {

/// @struct EMCALBlockHeader
/// @brief Header for EMCAL flagging the following EMCAL payload
struct EMCALBlockHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "EMCBlkH";
  static const uint32_t sVersion = 1;

  /// @enum PayloadType_t
  /// @brief Type of payload in the data block
  enum class PayloadType_t {
    kNoPayload,       ///< No payload
    kDigits,          ///< Digits payload
    kClusters,        ///< Clusters payload
    kRaw              ///< Raw data payload
  };

  EMCALBlockHeader(PayloadType_t payloadType) : BaseHeader(sizeof(EMCALBlockHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), mPayloadType(payloadType) {}

  PayloadType_t         mPayloadType;                   ///< Type of the payload in the data block of the message
};

} // namespace emcal
} // namespace o2

#endif // O2_EMCALBLOCKHEADER
