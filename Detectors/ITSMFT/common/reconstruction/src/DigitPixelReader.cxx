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
  if (!mLastDigit) {                 // new ROF record should be started
    if (mIdDig >= mDigits->size()) { // nothing left
      return nullptr;
    }
    mLastDigit = &((*mDigits)[mIdDig++]);
  }
  auto chipID = mLastDigit->getChipIndex();
  return getNextChipData(chipDataVec[chipID]) ? &chipDataVec[chipID] : nullptr;
}

//______________________________________________________________________________
bool DigitPixelReader::getNextChipData(ChipPixelData& chipData)
{
  // decode data of single chip to chipData
  if (!mLastDigit) {                 // new ROF record should be started
    if (mIdDig >= mDigits->size()) { // nothing left
      return false;
    }
    mLastDigit = &((*mDigits)[mIdDig++]);
  }
  // get corresponding ROF record
  int lim = -1;
  if (mIdROF >= 0) {
    const auto& rofRec = (*mROFRecVec)[mIdROF];
    lim = rofRec.getROFEntry().getIndex() + rofRec.getNROFEntries();
  }
  while (mIdDig > lim) {
    const auto& rofRec = (*mROFRecVec)[++mIdROF];
    lim = rofRec.getROFEntry().getIndex() + rofRec.getNROFEntries();
    mInteractionRecord = rofRec.getBCData(); // update interaction record
  }
  chipData.clear();
  chipData.setStartID(mIdDig - 1); // ??
  chipData.setChipID(mLastDigit->getChipIndex());
  chipData.setROFrame(mLastDigit->getROFrame());
  chipData.getData().emplace_back(mLastDigit);
  mLastDigit = nullptr;

  for (; mIdDig < lim;) {
    mLastDigit = &((*mDigits)[++mIdDig]);
    if (mLastDigit->getChipIndex() != chipData.getChipID()) { // new chip starts
      return true;
    }
    chipData.getData().emplace_back(mLastDigit);
    mLastDigit = nullptr; // reset pointer of already used digit
  }
  return true;
}

//______________________________________________________________________________
void DigitPixelReader::openInput(const std::string inpName, o2::detectors::DetID det)
{
  // open input file, load digits, MC labels
  assert(det.getID() == o2::detectors::DetID::ITS || det.getID() == o2::detectors::DetID::MFT);

  clear();
  std::string detName = det.getName();

  if (!(mInputTree = o2::utils::RootChain::load("o2sim", inpName))) {
    LOG(FATAL) << "Failed to load Digits tree from " << inpName << FairLogger::endl;
  }
  mInputTree->SetBranchAddress((detName + "Digit").c_str(), &mDigitsSelf);
  if (!mDigitsSelf) {
    LOG(FATAL) << "Failed to find " << (detName + "Digit").c_str() << " branch in the " << mInputTree->GetName()
               << " from file " << inpName << FairLogger::endl;
  }
  setDigits(mDigitsSelf);

  if (!(mInputTreeROF = o2::utils::RootChain::load((detName + "DigitROF").c_str(), inpName))) {
    LOG(FATAL) << "Failed to load ROF records tree from " << inpName << FairLogger::endl;
  }
  mInputTreeROF->SetBranchAddress((detName + "DigitROF").c_str(), &mROFRecVecSelf);
  if (!mROFRecVecSelf) {
    LOG(FATAL) << "Failed to find " << (detName + "DigitROF").c_str() << " branch in the " << mInputTree->GetName()
               << " from file " << inpName << FairLogger::endl;
  }
  setROFRecords(mROFRecVecSelf);

  mInputTree->SetBranchAddress((detName + "DigitMCTruth").data(), &mDigitsMCTruthSelf);
  setDigitsMCTruth(mDigitsMCTruthSelf);
}

//______________________________________________________________________________
bool DigitPixelReader::readNextEntry()
{
  // load next entry from the self-managed input
  auto nev = mInputTree->GetEntries();
  if (mInputTreeROF->GetEntries() != nev || nev != 1) {
    LOG(FATAL) << "In the self-managed mode the Digits and ROFRecords trees must have 1 entry only";
  }
  auto evID = mInputTree->GetReadEntry();
  if (evID < -1)
    evID = -1;
  if (++evID < nev) {
    init();
    mInputTree->GetEntry(evID);
    mInputTreeROF->GetEntry(evID);
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
  delete mDigitsSelf;
  delete mDigitsMCTruthSelf;
  mDigitsMCTruthSelf = nullptr;
  mDigitsSelf = nullptr;
}
