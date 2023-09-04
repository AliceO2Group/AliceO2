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

/// \file   DigitDump.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <cstddef>
#include "TTree.h"
#include "TString.h"

#include "Framework/Logger.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCCalibration/DigitDumpParam.h"

using namespace o2::tpc;

//______________________________________________________________________________
DigitDump::~DigitDump()
{
  if (mFile) {
    mFile->Write();
  }
}

//______________________________________________________________________________
void DigitDump::init()
{
  const auto& param = DigitDumpParam::Instance();

  if (param.FirstTimeBin >= 0) {
    mFirstTimeBin = param.FirstTimeBin;
    LOGP(info, "Setting FirstTimeBin = {} from TPCDigitDump.FirstTimeBin", mFirstTimeBin);
  }
  if (param.LastTimeBin >= 0) {
    mLastTimeBin = param.LastTimeBin;
    LOGP(info, "Setting LastTimeBin = {} from TPCDigitDump.LastTimeBin", mLastTimeBin);
  }
  mADCMin = param.ADCMin;
  mADCMax = param.ADCMax;
  mNoiseThreshold = param.NoiseThreshold;
  mPedestalAndNoiseFile = param.PedestalAndNoiseFile;
}

//______________________________________________________________________________
Int_t DigitDump::updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                           const Int_t timeBin, const Float_t signal)
{
  if ((timeBin < mFirstTimeBin) || (timeBin > mLastTimeBin)) {
    return 0;
  }

  if (!mInitialized) {
    initInputOutput();
  }

  Mapper& mapper = Mapper::instance();
  const PadRegionInfo& regionInfo = mMapper.getPadRegionInfo(cru.region());
  const int globalRow = row + regionInfo.getGlobalRowOffset();
  const int sectorRow = globalRow - (cru.rocType() == RocType::OROC) * mapper.getNumberOfRowsROC(0);

  // noise and pedestal values
  float pedestal = 0;
  float noise = 0;
  if (mPedestal) {
    pedestal = mPedestal->getValue(cru.roc(), sectorRow, pad);
  }
  if (mNoise) {
    noise = mNoise->getValue(cru.roc(), sectorRow, pad);
  }

  // check adc thresholds (zero suppression)
  const Float_t signalCorr = signal - pedestal;

  if ((signalCorr < mADCMin) || (signalCorr > mADCMax)) {
    return 0;
  }

  if ((mNoiseThreshold > 0) && (signalCorr < noise * mNoiseThreshold)) {
    return 0;
  }

  // check for masked pads
  if (mPadMask.size() && std::find(mPadMask.begin(), mPadMask.end(), std::array<int, 3>({int(cru.roc()), sectorRow, pad})) != mPadMask.end()) {
    return 1;
  }
  // printf("updateCRU: %d, %d (%d, %d), %d, %d, %f, %f\n", int(cru), row, globalRow, sectorRow, pad, timeBin, signal, pedestal);

  // fill digits
  addDigit(cru, signalCorr, globalRow, pad, timeBin);

  // fill time bin occupancy
  ++mTimeBinOccupancy[timeBin - mFirstTimeBin];

  return 0;
}

//______________________________________________________________________________
void DigitDump::sortDigits()
{
  // sort digits
  for (auto& digits : mDigits) {
    std::sort(digits.begin(), digits.end(), [](const auto& a, const auto& b) {
      if (a.getTimeStamp() < b.getTimeStamp()) {
        return true;
      }
      if (a.getTimeStamp() == b.getTimeStamp()) {
        if (a.getRow() < b.getRow()) {
          return true;
        } else if (a.getRow() == b.getRow()) {
          return a.getPad() < b.getPad();
        }
      }
      return false;
    });
  }
}

//______________________________________________________________________________
void DigitDump::endEvent()
{
  // sort digits
  sortDigits();

  mTree->Fill();

  clearDigits();
}

//______________________________________________________________________________
void DigitDump::loadNoiseAndPedestal()
{
  if (!mPedestalAndNoiseFile.size()) {
    LOG(warning) << "No pedestal and noise file name set";
    return;
  }

  std::unique_ptr<TFile> f(TFile::Open(mPedestalAndNoiseFile.data()));
  if (!f || !f->IsOpen() || f->IsZombie()) {
    LOG(fatal) << "Could not open pedestal file: " << mPedestalAndNoiseFile;
  }

  CalPad* pedestal{nullptr};
  CalPad* noise{nullptr};

  f->GetObject("Pedestals", pedestal);
  f->GetObject("Noise", noise);

  mPedestal = std::move(std::unique_ptr<const CalPad>(pedestal));
  mNoise = std::move(std::unique_ptr<const CalPad>(noise));
}

//______________________________________________________________________________
void DigitDump::setupOutputTree()
{
  mFile = std::make_unique<TFile>(mDigitFile.data(), "recreate");
  mTree = new TTree("o2sim", "o2sim");

  for (int iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
    mTree->Branch(Form("TPCDigit_%d", iSec), &mDigits[iSec]);
  }
}

//______________________________________________________________________________
void DigitDump::initInputOutput()
{
  loadNoiseAndPedestal();
  if (!mInMemoryOnly) {
    setupOutputTree();
  }
  mTimeBinOccupancy.resize(mLastTimeBin - mFirstTimeBin + 1);
  mInitialized = true;
}

//______________________________________________________________________________
void DigitDump::checkDuplicates(bool removeDuplicates)
{
  sortDigits();

  auto isEqual = [](const Digit& a, const Digit& b) {
    if ((a.getTimeStamp() == b.getTimeStamp()) && (a.getRow() == b.getRow()) && (a.getPad() == b.getPad())) {
      LOGP(debug, "digit found twice at sector {:2}, cru {:3}, row {:3}, pad {:3}, time {:6}, ADC {:.2} (other: {:.2})", b.getCRU() / 10, b.getCRU(), b.getRow(), b.getPad(), b.getTimeStamp(), b.getChargeFloat(), a.getChargeFloat());
      return true;
    }
    return false;
  };

  for (size_t iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
    auto& digits = mDigits[iSec];
    if (!digits.size()) {
      continue;
    }

    size_t nDuplicates = 0;
    if (removeDuplicates) {
      const auto last = std::unique(digits.begin(), digits.end(), isEqual);
      nDuplicates = std::distance(last, digits.end());
      digits.erase(last, digits.end());
    } else {
      auto first = digits.begin();
      const auto last = digits.end();
      while (++first != last) {
        nDuplicates += isEqual(*(first - 1), *first);
      }
    }
    if (nDuplicates) {
      static std::array<size_t, Sector::MAXSECTOR> nWarning{};
      static std::array<size_t, Sector::MAXSECTOR> suppression{};
      if (nWarning[iSec] < 5 || nWarning[iSec] == suppression[iSec]) {
        LOGP(alarm, "{} {} duplicate digits in sector {}, warned {} times in this sector", removeDuplicates ? "removed" : "found", nDuplicates, iSec, nWarning[iSec]);
        if (nWarning[iSec] == 4) {
          suppression[iSec] = 10;
        }
        suppression[iSec] *= 10;
      }
      ++nWarning[iSec];
    }
  }
}

//______________________________________________________________________________
void DigitDump::removeCEdigits(uint32_t removeNtimeBinsBefore, uint32_t removeNtimeBinsAfter, std::array<std::vector<Digit>, Sector::MAXSECTOR>* removedDigits)
{
  if (!mInitialized || !mTimeBinOccupancy.size()) {
    LOGP(info, "Cannot calculate CE position, mInitialized = {}, mTimeBinOccupancy.size() = {}", mInitialized, mTimeBinOccupancy.size());
    return;
  }
  // ===| check if proper CE signal was found |===
  const auto sectorsWithDigits = std::count_if(mDigits.begin(), mDigits.end(), [](const auto& v) { return v.size(); });
  const auto maxElem = std::max_element(mTimeBinOccupancy.begin(), mTimeBinOccupancy.end());
  const auto maxVal = *maxElem;

  if (!sectorsWithDigits || maxVal < 10) {
    LOGP(info, "No sectors with digits: {} or too few pads with max number of digits: {} < 10, CE detection stopped", sectorsWithDigits, maxVal);
    return;
  }

  // at least 20% of all pad should have fired in sectors wich have digits
  const size_t threshold = Mapper::getPadsInSector() * sectorsWithDigits / 5;
  if (maxVal < threshold) {
    LOGP(warning, "No CE signal found. Number of fired pads is too small {} < {} (with {} sectors having digits)", maxVal, threshold, sectorsWithDigits);
    return;
  }

  // identify the first time bin to remove
  const auto posMaxElem = std::distance(mTimeBinOccupancy.begin(), maxElem);
  const auto cePos = posMaxElem + mFirstTimeBin;

  if (cePos < posMaxElem) {
    LOGP(warning, "Number of time bins to be removed {} is bigger than the CE peak position {}", removeNtimeBinsBefore, posMaxElem);
    return;
  }

  const auto firstTimeBin = cePos - removeNtimeBinsBefore;
  const auto lastTimeBin = cePos + removeNtimeBinsAfter;
  LOGP(info, "CE position found at time bin {}, removing the range {} - {}", cePos, firstTimeBin, lastTimeBin);

  for (size_t iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
    auto& digits = mDigits[iSec];
    if (!digits.size()) {
      continue;
    }

    // LOGP(info, "processing sector iSec");
    const auto itFirstTB = std::lower_bound(digits.begin(), digits.end(),
                                            firstTimeBin,
                                            [](const auto& digit, const auto val) {
                                              return digit.getTimeStamp() < val;
                                            });

    // LOGP(info, "first time bin to remove is {} at position {} / {}", *itFirstTB, std::distance(digits.begin(), itFirstTB), digits.size());
    if (itFirstTB == digits.end()) {
      continue;
    }

    const auto itLastTB = std::upper_bound(digits.begin(), digits.end(),
                                           lastTimeBin,
                                           [](const auto val, const auto& digit) {
                                             return val < digit.getTimeStamp();
                                           });

    // LOGP(info, "last time bin to remove is {} at position {} / {}", *(itLastTB - 1), std::distance(digits.begin(), itLastTB), digits.size());
    if (removedDigits) {
      // LOGP(info, "copy removed digits");
      auto& cpDigits = (*removedDigits)[iSec];
      cpDigits.clear();
      std::copy(itFirstTB, itLastTB, std::back_inserter(cpDigits));
    }

    // LOGP(info, "erasing {} digits", std::distance(itFirstTB, itLastTB));
    digits.erase(itFirstTB, itLastTB);
  }
}
