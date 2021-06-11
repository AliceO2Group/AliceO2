// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_DIGIT_READER_H
#define O2_MCH_RAW_ENCODER_DIGIT_READER_H

#include <map>
#include <vector>
#include "DataFormatsMCH/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <TFile.h>

class TTree;
class TBranch;

namespace o2::mch::raw
{
class DigitReader
{

 public:
  DigitReader(const char* digitTreeFileName);

  bool nextDigits(InteractionRecord& ir, std::vector<o2::mch::Digit>& digits);

 private:
  void readNextEntry();

 private:
  TFile mFile;
  TTree* mTree{nullptr};
  TBranch* mDigitBranch{nullptr};
  std::vector<o2::mch::Digit>* mDigits{nullptr};
  size_t mEntry{0};
  std::map<o2::InteractionRecord, std::vector<o2::mch::Digit>> mDigitsPerIR;
  std::map<o2::InteractionRecord, std::vector<o2::mch::Digit>>::iterator mIterator{mDigitsPerIR.end()};
};
} // namespace o2::mch::raw

#endif
