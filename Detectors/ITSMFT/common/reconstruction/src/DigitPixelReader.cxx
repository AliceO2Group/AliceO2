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

/// \file DigitPixelReader.cxx
/// \brief Implementation of the Alpide pixel reader for MC digits processing

#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "CommonUtils/RootChain.h"
#include <fairlogger/Logger.h>
#include <cassert>

using namespace o2::itsmft;
using o2::itsmft::Digit;

//______________________________________________________________________________
DigitPixelReader::~DigitPixelReader()
{
  // in the self-managed mode we need to delete locally created containers
  clear();
}

//_____________________________________
int DigitPixelReader::decodeNextTrigger()
{
  // prerare data of the next trigger, return number of digits
  while (++mIdROF < mROFRecVec.size()) {
    if (mROFRecVec[mIdROF].getNEntries() > 0) {
      mIdDig = 0; // jump to the 1st digit of the trigger
      mInteractionRecord = mROFRecVec[mIdROF].getBCData();
      return mROFRecVec[mIdROF].getNEntries();
    }
  }
  return 0;
}

//______________________________________________________________________________
ChipPixelData* DigitPixelReader::getNextChipData(std::vector<ChipPixelData>& chipDataVec)
{
  // decode data of single chip to corresponding slot of chipDataVec
  if (mIdROF >= mROFRecVec.size()) {
    return nullptr; // TF is done
  }
  if (mIdROF < 0 || mIdDig >= mROFRecVec[mIdROF].getNEntries()) {
    if (!mDecodeNextAuto || !decodeNextTrigger()) { // no automatic decoding is asked or we are at the end of the TF
      return nullptr;
    }
  }
  auto chipID = mDigits[mROFRecVec[mIdROF].getFirstEntry() + mIdDig].getChipIndex();
  return getNextChipData(chipDataVec[chipID]) ? &chipDataVec[chipID] : nullptr;
}

//______________________________________________________________________________
bool DigitPixelReader::getNextChipData(ChipPixelData& chipData)
{
  // decode data of single chip to chipData
  if (mIdROF >= mROFRecVec.size()) {
    return false; // TF is done
  }
  if (mIdROF < 0 || mIdDig >= mROFRecVec[mIdROF].getNEntries()) {
    if (!mDecodeNextAuto || !decodeNextTrigger()) { // no automatic decoding is asked or we are at the end of the TF
      return false;
    }
  }
  chipData.clear();
  int did = mROFRecVec[mIdROF].getFirstEntry() + mIdDig;
  chipData.setStartID(did); // for the MC references
  const auto* digit = &mDigits[did];
  chipData.setChipID(digit->getChipIndex());
  chipData.setROFrame(mROFRecVec[mIdROF].getROFrame());
  chipData.setInteractionRecord(mInteractionRecord);
  chipData.setTrigger(mTrigger);
  chipData.getData().emplace_back(digit);
  int lim = mROFRecVec[mIdROF].getFirstEntry() + mROFRecVec[mIdROF].getNEntries();
  while ((++did < lim) && (digit = &mDigits[did])->getChipIndex() == chipData.getChipID()) {
    chipData.getData().emplace_back(digit);
  }
  mIdDig = did - mROFRecVec[mIdROF].getFirstEntry();
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
    LOG(fatal) << "Failed to load Digits tree from " << inpName;
  }
  mInputTree->SetBranchAddress((detName + "Digit").c_str(), &mDigitsSelf);
  if (!mDigitsSelf) {
    LOG(fatal) << "Failed to find " << (detName + "Digit").c_str() << " branch in the " << mInputTree->GetName()
               << " from file " << inpName;
  }

  mInputTree->SetBranchAddress((detName + "DigitROF").c_str(), &mROFRecVecSelf);
  if (!mROFRecVecSelf) {
    LOG(fatal) << "Failed to find " << (detName + "DigitROF").c_str() << " branch in the " << mInputTree->GetName()
               << " from file " << inpName;
  }

  mInputTree->SetBranchAddress((detName + "DigitMC2ROF").c_str(), &mMC2ROFRecVecSelf);
  if (!mMC2ROFRecVecSelf) {
    LOG(fatal) << "Failed to find " << (detName + "DigitMC2ROF").c_str() << " branch in the " << mInputTree->GetName()
               << " from file " << inpName;
  }

  mInputTree->SetBranchAddress((detName + "DigitMCTruth").data(), &mDigitsMCTruthSelf);
  //setDigitsMCTruth(mDigitsMCTruthSelf); // it will be assigned again at the reading, this is just to signal that the MCtruth is there
}

//______________________________________________________________________________
bool DigitPixelReader::readNextEntry()
{
  // load next entry from the self-managed input
  auto nev = mInputTree->GetEntries();
  auto evID = mInputTree->GetReadEntry();
  if (evID < -1) {
    evID = -1;
  }
  if (++evID < nev) {
    init();
    mInputTree->GetEntry(evID);
    setDigits(gsl::span(mDigitsSelf->data(), mDigitsSelf->size()));
    setROFRecords(gsl::span(mROFRecVecSelf->data(), mROFRecVecSelf->size()));
    setMC2ROFRecords(gsl::span(mMC2ROFRecVecSelf->data(), mMC2ROFRecVecSelf->size()));
    //setDigitsMCTruth(mDigitsMCTruthSelf);
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
  //
  mDigits = gsl::span<const o2::itsmft::Digit>();
  mROFRecVec = gsl::span<const o2::itsmft::ROFRecord>();
  mMC2ROFRecVec = gsl::span<const o2::itsmft::MC2ROFRecord>();
}
