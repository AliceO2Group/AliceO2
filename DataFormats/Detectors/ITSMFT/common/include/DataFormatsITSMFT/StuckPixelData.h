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

/// @file   StuckPixelData.h
/// @brief  CCDB-serializable container for stuck (repeating) pixel error records.
///
/// Design rationale
/// ----------------
/// TTree-based storage is intentionally avoided for CCDB objects because TTree
/// branches hold internal file-pointer state; serialising an in-memory TTree
/// via CcdbApi::createObjectImage() can silently drop the last unflushed basket.
/// A plain std::vector<StuckPixelEntry> has no such issue: ROOT's TClass
/// machinery serialises it correctly via the generated dictionary, exactly as
/// it does for TimeDeadMap.

#ifndef ITSMFT_STUCKPIXELDATA_H
#define ITSMFT_STUCKPIXELDATA_H

#include <vector>
#include <cstdint>
#include <Rtypes.h> // ClassDefNV

namespace o2
{
namespace itsmft
{

/// One stuck-pixel (RepeatingPixel error) record.
struct StuckPixelEntry {
  Long64_t orbit{0};  ///< first orbit of the TF in which the error was seen
  uint16_t chipID{0}; ///< global chip ID (ITS only)
  uint16_t row{0};    ///< pixel row
  uint16_t col{0};    ///< pixel column

  StuckPixelEntry() = default;
  StuckPixelEntry(Long64_t o, uint16_t c, uint16_t r, uint16_t col_)
    : orbit(o), chipID(c), row(r), col(col_) {}

  ClassDefNV(StuckPixelEntry, 1);
};

/// CCDB payload object: a run-level collection of stuck-pixel records.
class StuckPixelData
{
 public:
  StuckPixelData() = default;
  ~StuckPixelData() = default;

  void addEntry(Long64_t orbit, uint16_t chipID, uint16_t row, uint16_t col)
  {
    mEntries.emplace_back(orbit, chipID, row, col);
  }

  void clear() { mEntries.clear(); }

  const std::vector<StuckPixelEntry>& getEntries() const { return mEntries; }
  std::size_t size() const { return mEntries.size(); }
  bool empty() const { return mEntries.empty(); }

 private:
  std::vector<StuckPixelEntry> mEntries;

  ClassDefNV(StuckPixelData, 1);
};

} // namespace itsmft
} // namespace o2

#endif // ITSMFT_STUCKPIXELDATA_H