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

#include "DigitTreeReader.h"
#include <limits>
#include <fmt/format.h>

namespace o2::mch::raw
{

void AssertBranch(ROOT::Internal::TTreeReaderValueBase& value)
{
  if (value.GetSetupStatus() < 0) {
    throw std::invalid_argument(fmt::format("Error {} setting up tree reader for branch {}",
                                            (int)value.GetSetupStatus(), value.GetBranchName()));
  }
}

DigitTreeReader::DigitTreeReader(TTree* tree) : mCurrentRof{std::numeric_limits<size_t>::max()}
{
  if (!tree) {
    throw std::invalid_argument("cannot work with a null tree pointer");
  }
  mTreeReader.SetTree(tree);
  mTreeReader.Restart();
  mTreeReader.Next();
  mCurrentRof = 0;
  AssertBranch(mDigits);
  AssertBranch(mRofs);
}

bool DigitTreeReader::nextDigits(o2::mch::ROFRecord& rof, std::vector<o2::mch::Digit>& digits)
{
  if (mCurrentRof >= mRofs->size()) {
    if (!mTreeReader.Next()) {
      return false;
    }
    mCurrentRof = 0;
  }

  if (mRofs->empty()) {
    return false;
  }
  rof = (*mRofs)[mCurrentRof];
  digits.clear();
  auto& tfDigits = *mDigits;
  digits.insert(digits.begin(), tfDigits.begin() + rof.getFirstIdx(), tfDigits.begin() + rof.getLastIdx() + 1);
  ++mCurrentRof;
  return true;
}
} // namespace o2::mch::raw
