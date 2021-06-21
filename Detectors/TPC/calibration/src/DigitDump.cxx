// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DigitDump.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TTree.h"
#include "TString.h"

#include <fairlogger/Logger.h>

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

  mFirstTimeBin = param.FirstTimeBin;
  mLastTimeBin = param.LastTimeBin;
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
  //printf("updateCRU: %d, %d (%d, %d), %d, %d, %f, %f\n", int(cru), row, globalRow, sectorRow, pad, timeBin, signal, pedestal);

  // fill digits
  addDigit(cru, signalCorr, globalRow, pad, timeBin);

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
    LOG(WARNING) << "No pedestal and noise file name set";
    return;
  }

  std::unique_ptr<TFile> f(TFile::Open(mPedestalAndNoiseFile.data()));
  if (!f || !f->IsOpen() || f->IsZombie()) {
    LOG(FATAL) << "Could not open pedestal file: " << mPedestalAndNoiseFile;
  }

  CalPad* pedestal{nullptr};
  CalPad* noise{nullptr};

  f->GetObject("Pedestals", pedestal);
  f->GetObject("Noise", noise);

  mPedestal = std::move(std::unique_ptr<CalPad>(pedestal));
  mNoise = std::move(std::unique_ptr<CalPad>(noise));
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
  mInitialized = true;
}

//______________________________________________________________________________
void DigitDump::checkDuplicates(bool removeDuplicates)
{
  auto isEqual = [](const Digit& a, const Digit& b) {
    return (a.getTimeStamp() == b.getTimeStamp()) && (a.getRow() == b.getRow()) && (a.getPad() == b.getPad());
  };

  sortDigits();
  for (size_t iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
    auto& digits = mDigits[iSec];
    const auto nDigits = digits.size();
    if (nDigits < 2) {
      continue;
    }

    for (auto iDigit = nDigits - 2; iDigit--;) {
      auto& dig = digits[iDigit];
      auto& digPrev = digits[iDigit + 1];

      if (isEqual(dig, digPrev)) {
        if (removeDuplicates) {
          digits.erase(digits.begin() + iDigit + 1);
          LOGP(warning, "dig found twice at sector {:2}, cru {:3}, row {:3}, pad {:3}, time {:6}, ADC {:.2} (previous: {:.2}), removing it", iSec, dig.getCRU(), dig.getRow(), dig.getPad(), dig.getTimeStamp(), dig.getChargeFloat(), digPrev.getChargeFloat());
        } else {
          LOGP(warning, "dig found twice at sector {:2}, cru {:3}, row {:3}, pad {:3}, time {:6}, ADC {:.2} (previous: {:.2})", iSec, dig.getCRU(), dig.getRow(), dig.getPad(), dig.getTimeStamp(), dig.getChargeFloat(), digPrev.getChargeFloat());
        }
      }
    }
  }
}
