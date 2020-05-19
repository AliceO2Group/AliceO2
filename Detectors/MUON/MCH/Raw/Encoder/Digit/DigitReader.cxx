// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitReader.h"
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <limits>
#include "DetectorsRaw/HBFUtils.h"

namespace o2::mch::raw
{

DigitReader::DigitReader(const char* digitTreeFileName)
  : mFile(digitTreeFileName)
{
  mFile.GetObject("o2sim", mTree);
  mDigitBranch = mTree->GetBranch("MCHDigit");
  mDigitBranch->SetAddress(&mDigits);
}

void DigitReader::readNextEntry()
{
  assert(mDigitsPerIR.empty());

  mDigitBranch->GetEntry(mEntry);
  mEntry++;

  // builds a list of digits per HBF
  const o2::raw::HBFUtils& hbfutils = o2::raw::HBFUtils::Instance();

  for (auto d : (*mDigits)) {
    o2::InteractionTimeRecord ir(d.getTime().sampaTime);
    // get the closest (previous) HBF from this IR
    // and assign the digits to that one
    auto hbf = hbfutils.getIRHBF(hbfutils.getHBF(ir));
    mDigitsPerIR[hbf].push_back(d);
  }
}

bool DigitReader::nextDigits(InteractionRecord& ir, std::vector<o2::mch::Digit>& digits)
{
  if (mIterator == mDigitsPerIR.end()) {
    if (mEntry >= mDigitBranch->GetEntries()) {
      return false;
    }
    readNextEntry();
    mIterator = mDigitsPerIR.begin();
  }
  ir = mIterator->first;
  digits = mIterator->second;
  mIterator++;
  return true;
}
} // namespace o2::mch::raw
