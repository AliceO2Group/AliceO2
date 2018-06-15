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
ChipPixelData* DigitPixelReader::getNextChipData(std::vector<ChipPixelData>& chipDataVec)
{
  // decode data of single chip to corresponding slot of chipDataVec
  if (!mLastDigit) {
    if (mIdx >= mDigitArray->size()) {
      return nullptr;
    }
    mLastDigit = &((*mDigitArray)[mIdx++]);
  }
  auto chipID = mLastDigit->getChipIndex();
  if (chipID >= chipDataVec.size()) {
    chipDataVec.resize(chipID + 100);
  }
  return getNextChipData(chipDataVec[chipID]) ? &chipDataVec[chipID] : nullptr;
}

//______________________________________________________________________________
bool DigitPixelReader::getNextChipData(ChipPixelData& chipData)
{
  // decode data of single chip to chipData
  if (!mLastDigit) {
    if (mIdx >= mDigitArray->size()) {
      return false;
    }
    mLastDigit = &((*mDigitArray)[mIdx++]);
  }
  chipData.clear();
  chipData.setStartID(mIdx - 1);
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
  return true;
}

//______________________________________________________________________________
bool RawPixelReader::getNextChipData(ChipPixelData& chipData) { return true; }

//______________________________________________________________________________
ChipPixelData* RawPixelReader::getNextChipData(std::vector<ChipPixelData>& chipDataVec) { return nullptr; }
