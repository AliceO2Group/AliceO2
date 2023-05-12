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

/// \file IRFrameSelector.h
/// \brief Class to check if give InteractionRecord or IRFrame is selected by the external IRFrame vector
/// \author ruben.shahoyan@cern.ch

#ifndef O2_UTILS_IRFRAMESELECTOR_H
#define O2_UTILS_IRFRAMESELECTOR_H

#include "CommonDataFormat/IRFrame.h"
#include <gsl/span>

namespace o2::utils
{
class IRFrameSelector
{
 public:
  long check(o2::dataformats::IRFrame fr, size_t bwd = 0, size_t fwd = 0);
  long check(const o2::InteractionRecord& ir, size_t bwd = 0, size_t fwd = 0) { return check(o2::dataformats::IRFrame{ir, ir}, bwd, fwd); }
  gsl::span<const o2::dataformats::IRFrame> getMatchingFrames(const o2::dataformats::IRFrame& fr);

  template <typename SPAN>
  void setSelectedIRFrames(const SPAN& sp, size_t bwd = 0, size_t fwd = 0, long shift = 0, bool removeOverlaps = true)
  {
    mFrames = gsl::span<const o2::dataformats::IRFrame>(sp.data(), sp.size());
    mIsSet = true;
    applyMargins(bwd, fwd, shift, removeOverlaps);
    mLastIRFrameChecked.getMin().clear(); // invalidate
    mLastBoundID = -1;
  }

  void clear();
  size_t loadIRFrames(const std::string& fname);
  void applyMargins(size_t bwd, size_t fwd, long shift, bool removeOverlaps = true);
  void print(bool lst = false) const;

  auto getIRFrames() const { return mFrames; }
  bool isSet() const { return mIsSet; }

 private:
  gsl::span<const o2::dataformats::IRFrame> mFrames{}; // externally provided span of IRFrames, must be sorted in IRFrame.getMin()
  o2::dataformats::IRFrame mLastIRFrameChecked{};      // last frame which was checked
  long mLastBoundID = -1;                              // id of the last checked entry >= mLastIRFrameChecked
  bool mIsSet = false;                                 // flag that something was set (even if empty)
  std::vector<o2::dataformats::IRFrame> mOwnList;      // list loaded from the file
  ClassDefNV(IRFrameSelector, 1);
};

} // namespace o2::utils

#endif
