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

#include "MCHSimulation/DEDigitizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "DetectorsRaw/HBFUtils.h"
#include "MCHSimulation/DigitizerParam.h"

/// Convert collision time to ROF time (ROF duration = 4 BC)
std::pair<o2::InteractionRecord, uint8_t> time2ROFtime(const o2::InteractionRecord& time)
{
  auto bc = static_cast<uint8_t>(time.bc % 4);
  return std::make_pair(o2::InteractionRecord(time.bc - bc, time.orbit), bc);
}

namespace o2::mch
{

DEDigitizer::DEDigitizer(int deId, math_utils::Transform3D transformation, std::mt19937& random)
  : mDeId{deId},
    mResponse{deId < 300 ? Station::Type1 : Station::Type2345},
    mTransformation{transformation},
    mSegmentation{mch::mapping::segmentation(deId)},
    mRandom{random},
    mMinChargeDist{DigitizerParam::Instance().minChargeMean, DigitizerParam::Instance().minChargeSigma},
    mTimeDist{0., DigitizerParam::Instance().timeSigma},
    mNoiseDist{0., DigitizerParam::Instance().noiseSigma},
    mNoiseOnlyDist{DigitizerParam::Instance().noiseOnlyMean, DigitizerParam::Instance().noiseOnlySigma},
    mNofNoisyPadsDist{DigitizerParam::Instance().noiseOnlyProba * mSegmentation.nofPads()},
    mPadIdDist{0, mSegmentation.nofPads() - 1},
    mBCDist{0, 3},
    mSignals(mSegmentation.nofPads())
{
}

void DEDigitizer::processHit(const Hit& hit, const InteractionRecord& collisionTime, int evID, int srcID)
{
  MCCompLabel label(hit.GetTrackID(), evID, srcID);

  // convert energy to charge
  auto charge = mResponse.etocharge(hit.GetEnergyLoss());
  auto chargeCorr = mResponse.chargeCorr();
  auto chargeBending = chargeCorr * charge;
  auto chargeNonBending = charge / chargeCorr;

  // local position of the charge distribution
  auto exitPoint = hit.exitPoint();
  auto entrancePoint = hit.entrancePoint();
  math_utils::Point3D<float> lexit{};
  math_utils::Point3D<float> lentrance{};
  mTransformation.MasterToLocal(exitPoint, lexit);
  mTransformation.MasterToLocal(entrancePoint, lentrance);

  auto hitlengthZ = lentrance.Z() - lexit.Z();
  auto pitch = mResponse.getPitch();

  math_utils::Point3D<float> lpos{};

  if (abs(hitlengthZ) > pitch * 1.99) {
    lpos.SetXYZ((lexit.X() + lentrance.X()) / 2., (lexit.Y() + lentrance.Y()) / 2., (lexit.Z() + lentrance.Z()) / 2.);
  } else {
    lpos.SetXYZ(lexit.X(), // take Bragg peak coordinates assuming electron drift parallel to z
                lexit.Y(), // take Bragg peak coordinates assuming electron drift parallel to z
                0.0);      // take wire position global coordinate negative
  }
  auto localX = mResponse.getAnod(lpos.X());
  auto localY = lpos.Y();

  // borders of charge integration area
  auto dxy = mResponse.getSigmaIntegration() * mResponse.getChargeSpread();
  auto xMin = localX - dxy;
  auto xMax = localX + dxy;
  auto yMin = localY - dxy;
  auto yMax = localY + dxy;

  // loop over all pads within the defined bounding box to compute the charge on each of them
  mSegmentation.forEachPadInArea(xMin, yMin, xMax, yMax, [&](int padid) {
    auto dx = mSegmentation.padSizeX(padid) * 0.5;
    auto dy = mSegmentation.padSizeY(padid) * 0.5;
    auto xPad = mSegmentation.padPositionX(padid) - localX;
    auto yPad = mSegmentation.padPositionY(padid) - localY;
    auto q = mResponse.chargePadfraction(xPad - dx, xPad + dx, yPad - dy, yPad + dy);
    if (mResponse.isAboveThreshold(q)) {
      q *= mSegmentation.isBendingPad(padid) ? chargeBending : chargeNonBending;
      if (q > 0.f) {
        addSignal(padid, collisionTime, q, label);
      }
    }
  });
}

void DEDigitizer::addNoise(const InteractionRecord& firstIR, const InteractionRecord& lastIR)
{
  if (mNofNoisyPadsDist.mean() > 0.) {
    auto firstROF = time2ROFtime(firstIR);
    auto lastROF = time2ROFtime(lastIR);
    for (auto ir = firstROF.first; ir <= lastROF.first; ir += 4) {
      int nofNoisyPads = mNofNoisyPadsDist(mRandom);
      for (auto i = 0; i < nofNoisyPads; ++i) {
        addNoise(mPadIdDist(mRandom), ir);
      }
    }
  }
}

size_t DEDigitizer::digitize(std::map<InteractionRecord, DigitsAndLabels>& irDigitsAndLabels)
{
  size_t nPileup = 0;
  for (int padid = 0; padid < mSignals.size(); ++padid) {
    auto& signals = mSignals[padid];

    // add time dispersion to physical signal (noise-only signal is already randomly distributed)
    if (mTimeDist.stddev() > 0.f) {
      for (auto& signal : signals) {
        if (!signal.labels.front().isNoise()) {
          addTimeDispersion(signal);
        }
      }
    }

    // sort signals in time (needed to handle pileup)
    if (DigitizerParam::Instance().handlePileup) {
      std::sort(signals.begin(), signals.end(), [](const Signal& s1, const Signal& s2) {
        return s1.rofIR < s2.rofIR || (s1.rofIR == s2.rofIR && s1.bcInROF < s2.bcInROF);
      });
    }

    DigitsAndLabels* previousDigitsAndLabels = nullptr;
    auto previousDigitBCStart = std::numeric_limits<int64_t>::min();
    auto previousDigitBCEnd = std::numeric_limits<int64_t>::min();
    float previousRawCharge = 0.f;
    for (auto& signal : signals) {

      auto rawCharge = signal.charge;
      auto nSamples = mResponse.nSamples(rawCharge);

      // add noise to physical signal and reject it if it is below threshold
      // (not applied to noise-only signals, which are noise above threshold by definition)
      if (!signal.labels.back().isNoise()) {
        addNoise(signal, nSamples);
        if (!isAboveThreshold(signal.charge)) {
          continue;
        }
      }

      // create a digit or add the signal to the previous one in case of overlap and if requested.
      // a correct handling of pileup would require a complete simulation of the electronic signal
      auto digitBCStart = signal.rofIR.toLong();
      auto digitBCEnd = digitBCStart + 4 * nSamples - 1;
      if (DigitizerParam::Instance().handlePileup && digitBCStart <= previousDigitBCEnd + 8) {
        rawCharge += previousRawCharge;
        auto minNSamples = (std::max(previousDigitBCEnd, digitBCEnd) + 1 - previousDigitBCStart) / 4;
        nSamples = std::max(static_cast<uint32_t>(minNSamples), mResponse.nSamples(rawCharge));
        appendLastDigit(previousDigitsAndLabels, signal, nSamples);
        previousDigitBCEnd = previousDigitBCStart + 4 * nSamples - 1;
        previousRawCharge = rawCharge;
        ++nPileup;
      } else {
        previousDigitsAndLabels = addNewDigit(irDigitsAndLabels, padid, signal, nSamples);
        previousDigitBCStart = digitBCStart;
        previousDigitBCEnd = digitBCEnd;
        previousRawCharge = rawCharge;
      }
    }
  }

  return nPileup;
}

void DEDigitizer::clear()
{
  for (auto& signals : mSignals) {
    signals.clear();
  }
}

void DEDigitizer::addSignal(int padid, const InteractionRecord& collisionTime, float charge, const MCCompLabel& label)
{
  // convert collision time to ROF time
  auto rofTime = time2ROFtime(collisionTime);

  // search if we already have a signal for that pad in that ROF
  auto& signals = mSignals[padid];
  auto itSignal = std::find_if(signals.begin(), signals.end(),
                               [&rofTime](const Signal& s) { return s.rofIR == rofTime.first; });

  if (itSignal != signals.end()) {
    // merge with the existing signal
    itSignal->bcInROF = std::min(itSignal->bcInROF, rofTime.second);
    itSignal->charge += charge;
    itSignal->labels.push_back(label);
  } else {
    // otherwise create a new signal
    signals.emplace_back(rofTime.first, rofTime.second, charge, label);
  }
}

void DEDigitizer::addNoise(int padid, const InteractionRecord& rofIR)
{
  // search if we already have a signal for that pad in that ROF
  auto& signals = mSignals[padid];
  auto itSignal = std::find_if(signals.begin(), signals.end(), [&rofIR](const Signal& s) { return s.rofIR == rofIR; });

  // add noise-only signal only if no signal found
  if (itSignal == signals.end()) {
    auto bc = static_cast<uint8_t>(mBCDist(mRandom));
    auto charge = (mNoiseOnlyDist.stddev() > 0.f) ? mNoiseOnlyDist(mRandom) : mNoiseOnlyDist.mean();
    if (charge > 0.f) {
      signals.emplace_back(rofIR, bc, charge, MCCompLabel(true));
    }
  }
}

void DEDigitizer::addNoise(Signal& signal, uint32_t nSamples)
{
  if (mNoiseDist.stddev() > 0.f) {
    signal.charge += mNoiseDist(mRandom) * std::sqrt(nSamples);
  }
}

void DEDigitizer::addTimeDispersion(Signal& signal)
{
  auto time = signal.rofIR.toLong() + signal.bcInROF + std::llround(mTimeDist(mRandom));
  // the time must be positive
  if (time < 0) {
    time = 0;
  }
  auto [ir, bc] = time2ROFtime(InteractionRecord::long2IR(time));
  signal.rofIR = ir;
  signal.bcInROF = bc;
}

bool DEDigitizer::isAboveThreshold(float charge)
{
  if (charge > 0.f) {
    if (mMinChargeDist.stddev() > 0.f) {
      return charge > mMinChargeDist(mRandom);
    }
    return charge > mMinChargeDist.mean();
  }
  return false;
}

DEDigitizer::DigitsAndLabels* DEDigitizer::addNewDigit(std::map<InteractionRecord, DigitsAndLabels>& irDigitsAndLabels,
                                                       int padid, const Signal& signal, uint32_t nSamples) const
{
  uint32_t adc = std::round(signal.charge);
  auto time = signal.rofIR.differenceInBC({0, raw::HBFUtils::Instance().orbitFirst});
  nSamples = std::min(nSamples, 0x3FFU); // the number of samples must fit within 10 bits
  bool saturated = false;
  // the charge sum must fit within 20 bits
  // FIXME: we should better handle charge saturation here
  if (adc > 0xFFFFFU) {
    adc = 0xFFFFFU;
    saturated = true;
  }

  auto& digitsAndLabels = irDigitsAndLabels[signal.rofIR];
  digitsAndLabels.first.emplace_back(mDeId, padid, adc, time, nSamples, saturated);
  digitsAndLabels.second.addElements(digitsAndLabels.first.size() - 1, signal.labels);

  return &digitsAndLabels;
}

void DEDigitizer::appendLastDigit(DigitsAndLabels* digitsAndLabels, const Signal& signal, uint32_t nSamples) const
{
  auto& lastDigit = digitsAndLabels->first.back();
  uint32_t adc = lastDigit.getADC() + std::round(signal.charge);

  lastDigit.setADC(std::min(adc, 0xFFFFFU));           // the charge sum must fit within 20 bits
  lastDigit.setNofSamples(std::min(nSamples, 0x3FFU)); // the number of samples must fit within 10 bits
  lastDigit.setSaturated(adc > 0xFFFFFU);              // FIXME: we should better handle charge saturation here
  digitsAndLabels->second.addElements(digitsAndLabels->first.size() - 1, signal.labels);
}

} // namespace o2::mch
