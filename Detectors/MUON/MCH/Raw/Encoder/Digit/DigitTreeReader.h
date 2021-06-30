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

#ifndef O2_MCH_RAW_ENCODER_DIGIT_TREE_READER_H
#define O2_MCH_RAW_ENCODER_DIGIT_TREE_READER_H

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <TTreeReader.h>
#include <vector>

namespace o2::mch::raw
{
class DigitTreeReader
{
 public:
  DigitTreeReader(TTree* tree);

  bool nextDigits(o2::mch::ROFRecord& rof, std::vector<o2::mch::Digit>& digits);

 private:
  TTreeReader mTreeReader;
  TTreeReaderValue<std::vector<o2::mch::Digit>> mDigits = {mTreeReader, "MCHDigit"};
  TTreeReaderValue<std::vector<o2::mch::ROFRecord>> mRofs = {mTreeReader, "MCHROFRecords"};
  size_t mCurrentRof;
};
} // namespace o2::mch::raw

#endif
