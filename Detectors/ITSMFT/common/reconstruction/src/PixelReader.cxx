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
Bool_t DigitPixelReader::getNextChipData(PixelReader::ChipPixelData &chipData)
{
  chipData.clear();
  if (!mLastDigit) {
    if (mIdx >= mDigitArray->size()) {
      return kFALSE;
    }
    mLastDigit = &((*mDigitArray)[mIdx++]);
  }
  chipData.chipID  = mLastDigit->getChipIndex();
  chipData.roFrame = mLastDigit->getROFrame();
  chipData.timeStamp = mLastDigit->GetTimeStamp(); // time difference within the same TFrame does not matter, take 1st one
  chipData.pixels.emplace_back(mLastDigit);
  mLastDigit = nullptr;
  
  while (mIdx < mDigitArray->size()) {
    mLastDigit = &((*mDigitArray)[mIdx++]);
    if (chipData.chipID  != mLastDigit->getChipIndex()) break;
    if (chipData.roFrame != mLastDigit->getROFrame()) break;
    chipData.pixels.emplace_back(mLastDigit);
    mLastDigit = nullptr;
  }
  return kTRUE;
}
  
//______________________________________________________________________________
Bool_t RawPixelReader::getNextChipData(PixelReader::ChipPixelData &chipData)
{
  return kTRUE;
}
