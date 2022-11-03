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
#include <algorithm>
#include <numeric>

#include <iostream>

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
      if (mSquashOverflowsDepth) {
        auto nUsed = std::accumulate(mSquashedDigitsMask.begin() + mROFRecVec[mIdROF].getFirstEntry(),
                                     mSquashedDigitsMask.begin() + mROFRecVec[mIdROF].getFirstEntry() + mROFRecVec[mIdROF].getNEntries() - 1, 0);
        return mROFRecVec[mIdROF].getNEntries() - nUsed;
      } else {
        return mROFRecVec[mIdROF].getNEntries();
      }
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
  chipData.setStartID(did); // for the MC references, not used if squashing
  const auto* digit = &mDigits[did];
  chipData.setChipID(digit->getChipIndex());
  chipData.setROFrame(mROFRecVec[mIdROF].getROFrame());
  chipData.setInteractionRecord(mInteractionRecord);
  chipData.setTrigger(mTrigger);
  if (mSquashOverflowsDepth) {
    if (!mSquashedDigitsMask[did]) {
      chipData.getData().emplace_back(digit);
      mSquashedDigitsMask[did] = true;
      if (mDigitsMCTruth) {
        chipData.getPixIds().push_back(did); // this we do only with squashing + MC
      }
    }
  } else {
    chipData.getData().emplace_back(digit);
  }
  int lim = mROFRecVec[mIdROF].getFirstEntry() + mROFRecVec[mIdROF].getNEntries();
  while ((++did < lim) && (digit = &mDigits[did])->getChipIndex() == chipData.getChipID()) {
    if (mSquashOverflowsDepth) {
      if (!mSquashedDigitsMask[did]) {
        chipData.getData().emplace_back(digit);
        mSquashedDigitsMask[did] = true;
        if (mDigitsMCTruth) {
          chipData.getPixIds().push_back(did); // this we do only with squashing + MC
        }
      }
    } else {
      chipData.getData().emplace_back(digit);
    }
  }
  mIdDig = did - mROFRecVec[mIdROF].getFirstEntry();

  // Merge overflow digits from next N ROFs, being N the depth of the search
  if (!mIdROF || mIdROFLast != mIdROF) {
    mIdROFLast = mIdROF;
    for (uint16_t iROF{1}; iROF <= mSquashOverflowsDepth && (mIdROF + iROF) < mROFRecVec.size(); ++iROF) {
      mBookmarkNextROFs[iROF - 1] = mROFRecVec[mIdROF + iROF].getFirstEntry(); // reset starting bookmark
    }
  }

  // Loop over next ROFs
  for (uint16_t iROF{1}; iROF <= mSquashOverflowsDepth && (mIdROF + iROF) < mROFRecVec.size(); ++iROF) {
    int idNextROF{mIdROF + iROF};
    if (std::abs(mROFRecVec[idNextROF].getBCData().differenceInBC(mROFRecVec[idNextROF - 1].getBCData())) > mMaxBCSeparationToSquash) {
      break; // ROFs are too distant in BCs
    }

    int nDigitsNext{0};
    for (int i{mBookmarkNextROFs[iROF - 1]}; i < mROFRecVec[idNextROF].getFirstEntry() + mROFRecVec[idNextROF].getNEntries(); ++i) {
      if (mDigits[i].getChipIndex() < chipData.getChipID()) {
        ++mBookmarkNextROFs[iROF - 1];
      } else {
        if (mDigits[i].getChipIndex() == chipData.getChipID()) {
          ++nDigitsNext;
        } else {
          break;
        }
      }
    }
    if (!nDigitsNext) { // No data related to this chip in next rof, we stop squashing, assuming continue persistence
      break;
    }
    size_t initialSize{chipData.getData().size()};
    // Compile mask for repeated digits (digits with same row,col)
    for (size_t iPixel{0}, iDigitNext{0}; iPixel < initialSize; ++iPixel) {
      auto& pixel = chipData.getData()[iPixel];
      // seek to iDigitNext which is inferior than itC - mMaxSquashDist
      auto mincol = pixel.getCol() > mMaxSquashDist ? pixel.getCol() - mMaxSquashDist : 0;
      auto minrow = pixel.getRowDirect() > mMaxSquashDist ? pixel.getRowDirect() - mMaxSquashDist : 0;
      if (iDigitNext == nDigitsNext) { // in case iDigitNext loop below reached the end
        iDigitNext--;
      }
      while ((mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext].getColumn() > mincol || mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext].getRow() > minrow) && iDigitNext > 0) {
        iDigitNext--;
      }
      for (; iDigitNext < nDigitsNext; iDigitNext++) {
        if (mSquashedDigitsMask[mBookmarkNextROFs[iROF - 1] + iDigitNext]) {
          continue;
        }
        const auto* digitNext = &mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext];
        auto drow = static_cast<int>(digitNext->getRow()) - static_cast<int>(pixel.getRowDirect());
        auto dcol = static_cast<int>(digitNext->getColumn()) - static_cast<int>(pixel.getCol());
        if (!dcol && !drow) {
          mSquashedDigitsMask[mBookmarkNextROFs[iROF - 1] + iDigitNext] = true;
          break;
        }
      }
    }

    // loop over chip pixels
    for (int iPixel{0}, iDigitNext{0}; iPixel < initialSize; ++iPixel) {
      auto& pixel = chipData.getData()[iPixel];
      // seek to iDigitNext which is inferior than itC - mMaxSquashDist
      auto mincol = pixel.getCol() > mMaxSquashDist ? pixel.getCol() - mMaxSquashDist : 0;
      auto minrow = pixel.getRowDirect() > mMaxSquashDist ? pixel.getRowDirect() - mMaxSquashDist : 0;
      if (iDigitNext == nDigitsNext) { // in case iDigitNext loop below reached the end
        iDigitNext--;
      }
      while ((mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext].getColumn() > mincol || mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext].getRow() > minrow) && iDigitNext > 0) {
        iDigitNext--;
      }
      for (; iDigitNext < nDigitsNext; iDigitNext++) {
        if (mSquashedDigitsMask[mBookmarkNextROFs[iROF - 1] + iDigitNext]) {
          continue;
        }
        const auto* digitNext = &mDigits[mBookmarkNextROFs[iROF - 1] + iDigitNext];
        auto drow = static_cast<int>(digitNext->getRow()) - static_cast<int>(pixel.getRowDirect());
        auto dcol = static_cast<int>(digitNext->getColumn()) - static_cast<int>(pixel.getCol());
        if (!dcol && !drow) {
          // same pixel fired in two ROFs
          mSquashedDigitsMask[mBookmarkNextROFs[iROF - 1] + iDigitNext] = true;
          continue;
        }
        if (dcol > mMaxSquashDist || (dcol == mMaxSquashDist && drow > mMaxSquashDist)) {
          break; // all greater iDigitNexts will not match to this pixel too
        }
        if (dcol < -mMaxSquashDist || (drow > mMaxSquashDist || drow < -mMaxSquashDist)) {
          continue;
        } else {
          chipData.getData().emplace_back(digitNext); // push squashed pixel
          mSquashedDigitsMask[mBookmarkNextROFs[iROF - 1] + iDigitNext] = true;
          if (mDigitsMCTruth) {
            chipData.getPixIds().push_back(mBookmarkNextROFs[iROF - 1] + iDigitNext);
          }
        }
      }
    }
    mBookmarkNextROFs[iROF - 1] += nDigitsNext;
  }
  if (mSquashOverflowsDepth) {
    if (mDigitsMCTruth) {
      auto& pixels = chipData.getData();
      chipData.getPixelsOrder().resize(pixels.size());
      std::iota(chipData.getPixelsOrder().begin(), chipData.getPixelsOrder().end(), 0);
      std::sort(chipData.getPixelsOrder().begin(), chipData.getPixelsOrder().end(), [&pixels](int a, int b) -> bool { return pixels[a].getCol() == pixels[b].getCol() ? pixels[a].getRow() < pixels[b].getRow() : pixels[a].getCol() < pixels[b].getCol(); });
    }
    std::sort(chipData.getData().begin(), chipData.getData().end(), [](auto& d1, auto& d2) -> bool { return d1.getCol() == d2.getCol() ? d1.getRow() < d2.getRow() : d1.getCol() < d2.getCol(); });
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
  // setDigitsMCTruth(mDigitsMCTruthSelf); // it will be assigned again at the reading, this is just to signal that the MCtruth is there
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
    // setDigitsMCTruth(mDigitsMCTruthSelf);
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
