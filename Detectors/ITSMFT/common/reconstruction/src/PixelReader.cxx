// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PixelReader.cxx
/// \brief Implementation of the ITS pixel reader class

#include "ITSMFTReconstruction/PixelReader.h"

using namespace o2::ITSMFT;
using o2::ITSMFT::Digit;

//______________________________________________________________________________
Bool_t DigitPixelReader::getNextChipData(ChipPixelData& chipData)
{
  chipData.clear();
  if (!mLastDigit) {
    if (mIdx >= mDigitArray->size()) {
      return kFALSE;
    }
    chipData.setStartID(mIdx);
    mLastDigit = &((*mDigitArray)[mIdx++]);
  } else {
    chipData.setStartID(mIdx);
  }
  chipData.setChipID(mLastDigit->getChipIndex());
  chipData.setROFrame(mLastDigit->getROFrame());
  chipData.getData().emplace_back(mLastDigit);
  mLastDigit = nullptr;

  while (mIdx < mDigitArray->size()) {
    mLastDigit = &((*mDigitArray)[mIdx++]);
    if (chipData.getChipID() != mLastDigit->getChipIndex())
      break;
    if (chipData.getROFrame() != mLastDigit->getROFrame())
      break;
    chipData.getData().emplace_back(mLastDigit);
    mLastDigit = nullptr;
  }
  return kTRUE;
}

//______________________________________________________________________________
Bool_t RawPixelReader::getNextChipData(ChipPixelData& chipData) { return kTRUE; }
