// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitPixelReader.cxx
/// \brief Implementation of the Alpide pixel reader for MC digits processing

#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "CommonUtils/RootChain.h"
#include <FairLogger.h>
#include <cassert>

using namespace o2::ITSMFT;
using o2::ITSMFT::Digit;

//______________________________________________________________________________
DigitPixelReader::~DigitPixelReader()
{
  // in the self-managed mode we need to delete locally created containers
  clear();
}

//______________________________________________________________________________
ChipPixelData* DigitPixelReader::getNextChipData(std::vector<ChipPixelData>& chipDataVec)
{
  // decode data of single chip to corresponding slot of chipDataVec
  if (!mLastDigit) {
    if (mIdx >= mDigits->size()) {
      return nullptr;
    }
    mLastDigit = &((*mDigits)[mIdx++]);
  }
  auto chipID = mLastDigit->getChipIndex();
  return getNextChipData(chipDataVec[chipID]) ? &chipDataVec[chipID] : nullptr;
}

//______________________________________________________________________________
bool DigitPixelReader::getNextChipData(ChipPixelData& chipData)
{
  // decode data of single chip to chipData
  if (!mLastDigit) {
    if (mIdx >= mDigits->size()) {
      return false;
    }
    mLastDigit = &((*mDigits)[mIdx++]);
  }
  chipData.clear();
  chipData.setStartID(mIdx - 1);
  chipData.setChipID(mLastDigit->getChipIndex());
  chipData.setROFrame(mLastDigit->getROFrame());
  chipData.getData().emplace_back(mLastDigit);
  mLastDigit = nullptr;

  while (mIdx < mDigits->size()) {
    mLastDigit = &((*mDigits)[mIdx++]);
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
void DigitPixelReader::openInput(const std::string inpName, o2::detectors::DetID det)
{
  // open input file, load digits, MC labels
  assert(det.getID() == o2::detectors::DetID::ITS || det.getID() == o2::detectors::DetID::MFT);

  clear();

  if (!(mInputTree = o2::utils::RootChain::load("o2sim", inpName))) {
    LOG(FATAL) << "Failed to load Digits tree from " << inpName << FairLogger::endl;
  }
  std::string brName = det.getName();
  mInputTree->SetBranchAddress((brName + "Digit").data(), &mDigitsSelf);
  if (!mDigitsSelf) {
    LOG(FATAL) << "Failed to find ITSDigit branch in the " << mInputTree->GetName()
               << " from file " << inpName << FairLogger::endl;
  }
  setDigits(mDigitsSelf);

  mInputTree->SetBranchAddress((brName + "DigitMCTruth").data(), &mDigitsMCTruthSelf);
  setDigitsMCTruth(mDigitsMCTruthSelf);
}

//______________________________________________________________________________
bool DigitPixelReader::readNextEntry()
{
  // load next entry from the self-managed input
  auto nev = mInputTree->GetEntries();
  auto evID = mInputTree->GetReadEntry();
  if (evID < -1)
    evID = -1;
  if (++evID < nev) {
    init();
    mInputTree->GetEntry(evID);
    return true;
  } else {
    return false;
  }
}

//______________________________________________________________________________
void DigitPixelReader::clear()
{
  // clear data structures
  mInputTree.reset();
  delete mDigits;
  delete mDigitsMCTruth;
  mDigitsMCTruthSelf = nullptr;
  mDigitsMCTruth = nullptr;
  mDigitsSelf = nullptr;
  mDigits = nullptr;
}
