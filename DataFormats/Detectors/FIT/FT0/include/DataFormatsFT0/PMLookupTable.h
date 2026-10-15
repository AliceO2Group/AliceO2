// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FT0_PMLOOKUPTABLE_H_
#define O2_FT0_PMLOOKUPTABLE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>

#include "FT0Base/Constants.h"

namespace o2::ft0
{

/// Cached mapping between FT0 channels and PM modules.
///
/// The mapping is constructed once from SingleLUT and can then be reused by
/// digitization, reconstruction, trigger and QA code without reparsing the LUT.
class PMLookupTable
{
 public:
  using ChannelID = uint16_t;
  using PMHash = uint8_t;
  using PMMap = std::map<PMHash, bool>; // PM hash -> true for A side, false for C side

  /// Return the process-wide immutable lookup table.
  static const PMLookupTable& Instance();

  /// Return the PM hash assigned to a detector channel.
  PMHash getPMHash(ChannelID channelID) const;

  /// Return true for an A-side PM and false for a C-side PM.
  bool isASide(PMHash pmHash) const;

  /// Return all PM hashes and their sides. TCM entries are not included.
  const PMMap& getPMs() const noexcept { return mPMHash2IsASide; }

 private:
  PMLookupTable();

  std::array<PMHash, Constants::sNCHANNELS_PM> mChannelID2PMHash{};
  std::array<bool, Constants::sNCHANNELS_PM> mChannelIsMapped{};
  PMMap mPMHash2IsASide;
};

} // namespace o2::ft0

#endif // O2_FT0_PMLOOKUPTABLE_H_
