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

/// \file   MIDFiltering/FetToDead.h
/// \brief  Class to convert the FEE test event into dead channels
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 May 2021
#ifndef O2_MID_FETTODEAD_H
#define O2_MID_FETTODEAD_H

#include <cstdint>
#include <gsl/span>
#include "DataFormatsMID/ColumnData.h"
#include "MIDFiltering/ChannelMasksHandler.h"

namespace o2
{
namespace mid
{

class FetToDead
{
 public:
  FetToDead();
  ~FetToDead() = default;

  FetToDead(const FetToDead&) = default;
  FetToDead(FetToDead&&) = default;

  std::vector<ColumnData> process(gsl::span<const ColumnData> fetData);

  /// Sets the masks
  void setMasks(std::vector<ColumnData> masks);

 private:
  bool invertPattern(const ColumnData& col, std::vector<ColumnData>& invertedData);

  ChannelMasksHandler mMasksHandler{};                      // Masks handler
  std::unordered_map<uint16_t, ColumnData> mInvertedActive; // Map of inverted active channels
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_FETTODEAD_H */
