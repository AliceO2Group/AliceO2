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

/// \file IonTailCorrection.cxx
/// \brief Implementation of the ion tail correction from TPC digits
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
/// \author Marian Ivanov, m.ivanov@gsi.de

#include <cmath>
#include <vector>
#include <algorithm>

#include "TPCReconstruction/IonTailCorrection.h"
#include "TPCReconstruction/IonTailCorrectionSettings.h"

using namespace o2::tpc;

IonTailCorrection::IonTailCorrection()
{
  const auto& settings = IonTailCorrectionSettings::Instance();
  mKTime = settings.kTime;
  mITMultFactor = settings.ITMultFactor;
}

void IonTailCorrection::filterDigitsDirect(std::vector<Digit>& digits)
{
  // make sure we process the digits pad-by-pad in increasing time order
  sortDigitsOneSectorPerPad(digits);
  int lastRow = -1;
  int lastPad = -1;
  int lastTime = digits.front().getTimeStamp() - 1;

  float cumul = 0;

  // TODO: for now hard-coded, make pad-by-pad, if available
  float kAmp = std::abs(mITMultFactor) * 0.1276; // TODO: replace with pad-by-pad value
  if (mITMultFactor < 0) {
    kAmp = kAmp / (1 + kAmp);
  }
  const float kTime = mKTime; // TODO: replace with pad-by-pad value
  const float tailSlopeUnit = std::exp(-kTime);

  for (auto& digit : digits) {
    const auto row = digit.getRow();
    const auto pad = digit.getPad();
    auto time = digit.getTimeStamp();
    auto charge = digit.getChargeFloat();

    // reset charge cumulation if pad has changed
    if (row != lastRow || pad != lastPad) {
      cumul = 0;
      lastTime = time - 1;
    }
    auto origCharge = charge;
    charge = origCharge + mSign * kAmp * (1 - tailSlopeUnit) * cumul;

    // cumulate charge also over missing time stamps
    while (lastTime < time) {
      ++lastTime;

      using Streamer = o2::utils::DebugStreamer;
      if (Streamer::checkStream(o2::utils::StreamFlags::streamITCorr)) {
        mStreamer.setStreamer("debug_ITCorr", "UPDATE");
        int cru = digit.getCRU();
        auto rowTmp = row;
        auto padTmp = pad;
        auto kAmpTmp = kAmp;
        auto kTimeTmp = kTime;
        auto tailSlopeUnitTmp = tailSlopeUnit;
        auto chargeTmp = origCharge + mSign * kAmp * (1 - tailSlopeUnit) * cumul;

        mStreamer.getStreamer()
          << mStreamer.getUniqueTreeName("itCorr").data()
          << "cru=" << cru
          << "row=" << rowTmp
          << "pad=" << padTmp
          << "time=" << lastTime

          << "kAmp=" << kAmpTmp
          << "kTime=" << kTimeTmp
          << "tailSlopeUnit=" << tailSlopeUnitTmp
          << "itMultFactor=" << mITMultFactor

          << "origCharge=" << origCharge
          << "charge=" << chargeTmp
          << "\n";
      }

      cumul += origCharge;
      cumul *= tailSlopeUnit;

      if (cumul < 0.1) {
        cumul = 0;
        break;
      }
      origCharge = 0;
    }

    digit.setCharge(charge);

    lastRow = row;
    lastPad = pad;
    lastTime = time;
  }

  sortDigitsOneSectorPerTimeBin(digits);
}

void IonTailCorrection::sortDigitsOneSectorPerPad(std::vector<Digit>& digits)
{
  // sort digits
  std::sort(digits.begin(), digits.end(), [](const auto& a, const auto& b) {
    if (a.getRow() < b.getRow()) {
      return true;
    }
    if (a.getRow() == b.getRow()) {
      if (a.getPad() < b.getPad()) {
        return true;
      } else if (a.getPad() == b.getPad()) {
        return a.getTimeStamp() < b.getTimeStamp();
      }
    }
    return false;
  });
}

void IonTailCorrection::sortDigitsOneSectorPerTimeBin(std::vector<Digit>& digits)
{
  // sort digits
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
