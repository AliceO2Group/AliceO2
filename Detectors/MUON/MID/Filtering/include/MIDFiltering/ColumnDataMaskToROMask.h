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

/// \file   MIDFiltering/ColumnDataMaskToROMask.h
/// \brief  Converts ColumnData masks into local board masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 October 2021
#ifndef O2_MID_COLUMNDATAMASKTOROMASK_H
#define O2_MID_COLUMNDATAMASKTOROMASK_H

#include <vector>
#include <gsl/span>
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateMapper.h"

namespace o2
{
namespace mid
{

class ColumnDataMaskToROMask
{
 public:
  ColumnDataMaskToROMask();
  ~ColumnDataMaskToROMask() = default;

  std::vector<ROBoard> convert(gsl::span<const ColumnData> colDataMasks);
  void write(gsl::span<const ColumnData> colDataMasks, const char* outFilename);
  uint32_t makeConfigWord(const ROBoard& mask) const;

 private:
  bool needsMask(const ROBoard& mask, bool hasDirectInputY) const;
  ColumnDataToLocalBoard mColToBoard;    /// ColumnData to local board
  CrateMapper mCrateMapper;              /// Crate mapper
  std::vector<ColumnData> mDefaultMasks; /// Default masks

  static constexpr uint32_t sTxDataMask = 0x10000;
  static constexpr uint32_t sMonmoff = 0x2;
  static constexpr uint32_t sXorY = 0x400;
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATAMASKTOROMASK_H */
