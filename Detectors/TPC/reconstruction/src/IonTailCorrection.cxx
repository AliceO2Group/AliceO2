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
#include <filesystem>

#include "TPCBase/Utils.h"
#include "TPCBase/CRU.h"
#include "TPCReconstruction/IonTailCorrection.h"
#include "TPCBase/IonTailSettings.h"

using namespace o2::tpc;

IonTailCorrection::IonTailCorrection()
{
  const auto& settings = IonTailSettings::Instance();
  mKTime = settings.kTime;
  mITMultFactor = settings.ITMultFactor;
  LOGP(info, "IT settings: kTime = {}, ITMultFactor = {}, padITCorrFile = {}",
       settings.kTime, settings.ITMultFactor, settings.padITCorrFile);

  if (!settings.padITCorrFile.empty()) {
    loadITPadValuesFromFile(settings.padITCorrFile);
  }
}

void IonTailCorrection::filterDigitsDirect(std::vector<Digit>& digits)
{
  using Streamer = o2::utils::DebugStreamer;

  if (digits.size() == 0) {
    return;
  }

  // make sure we process the digits pad-by-pad in increasing time order
  sortDigitsOneSectorPerPad(digits);

  int lastRow = -1;
  int lastPad = -1;
  int lastTime = digits.front().getTimeStamp();
  float cumul = 0;

  // default IT parameters are replaced below,
  // in case pad-by-pad values were set up
  float kAmp = std::abs(mITMultFactor) * 0.1276;
  if (mITMultFactor < 0) {
    kAmp = kAmp / (1 + kAmp);
  }

  float kTime = mKTime;
  float tailSlopeUnit = std::exp(-kTime);

  for (auto& digit : digits) {
    const auto sector = CRU(digit.getCRU()).sector();
    const auto row = digit.getRow();
    const auto pad = digit.getPad();
    const auto time = digit.getTimeStamp();

    if (mFraction) {
      kAmp = mFraction->getValue(sector, row, pad) * std::abs(mITMultFactor);
      if (mITMultFactor < 0) {
        kAmp = kAmp / (1 + kAmp);
      }
    }
    if (mExpLambda) {
      tailSlopeUnit = mExpLambda->getValue(sector, row, pad);
      kTime = -std::log(tailSlopeUnit);
    }

    // reset charge cumulation if pad has changed
    if (row != lastRow || pad != lastPad) {
      cumul = 0;
      lastTime = time;
    }

    // cumulate charge also over missing time stamps
    while (lastTime + 1 < time) {
      auto origCuml = cumul;
      cumul *= tailSlopeUnit;

      if (Streamer::checkStream(o2::utils::StreamFlags::streamITCorr)) {
        streamData(digit.getCRU(), row, pad, time, lastTime, kAmp, kTime, tailSlopeUnit, origCuml, cumul, 0, 0);
      }

      ++lastTime;

      if (cumul < 0.1) {
        cumul = 0;
        break;
      }
    }

    const auto origCharge = digit.getChargeFloat();
    const auto charge = origCharge + mSign * kAmp * (1 - tailSlopeUnit) * cumul;
    digit.setCharge(charge);

    const auto origCuml = cumul;
    cumul += origCharge;
    cumul *= tailSlopeUnit;

    if (Streamer::checkStream(o2::utils::StreamFlags::streamITCorr)) {
      streamData(digit.getCRU(), row, pad, time, lastTime, kAmp, kTime, tailSlopeUnit, origCuml, cumul, origCharge, charge);
    }

    lastRow = row;
    lastPad = pad;
    lastTime = time;
  }

  // sort back
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

void IonTailCorrection::loadITPadValuesFromFile(std::string_view itParamFile)
{
  if (!std::filesystem::exists(itParamFile)) {
    LOGP(fatal, "Could not find IF param file {}", itParamFile);
  }

  auto calDets = utils::readCalPads(itParamFile, "fraction,expLambda");
  if (!calDets[0]) {
    LOGP(fatal, "Could not read IT fraction object from file {}", itParamFile);
  }
  if (!calDets[1]) {
    LOGP(fatal, "Could not read IT expLambda object from file {}", itParamFile);
  }

  mFraction.reset(calDets[0]);
  mExpLambda.reset(calDets[1]);

  LOGP(info, "Loaded ion tail correction values from file {}", itParamFile);
}

void IonTailCorrection::streamData(int cru, int row, int pad, int time, int lastTime, float kAmp, float kTime, float tailSlopeUnit, float origCumul, float cumul, float origCharge, float charge)
{
  mStreamer.setStreamer("debug_ITCorr", "UPDATE");

  mStreamer.getStreamer()
    << "itCorr"
    << "cru=" << cru
    << "row=" << row
    << "pad=" << pad
    << "time=" << lastTime

    << "kAmp=" << kAmp
    << "kTime=" << kTime
    << "tailSlopeUnit=" << tailSlopeUnit
    << "itMultFactor=" << mITMultFactor

    << "origCharge=" << origCharge
    << "charge=" << charge
    << "\n";
}
