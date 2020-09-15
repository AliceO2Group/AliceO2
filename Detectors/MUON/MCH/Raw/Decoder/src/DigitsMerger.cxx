// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DataDecoder.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include "DigitsMerger.h"

#include <iostream>

static bool mPrint = false;

namespace o2
{
namespace mch
{
namespace raw
{

MergerDigit& MergerDigit::operator+=(const MergerDigit& right)
{
  digit.setADC(right.digit.getADC() + digit.getADC());
  stopTime = right.stopTime;
  return (*this);
}

void FeeIdMerger::setOrbit(uint32_t orbit, bool stop)
{
  // perform the merging and send digits of previous orbit if either the stop RDH is received
  // or a new orbit is started
  if ((orbit == buffers[currentBufId].orbit) && (!stop)) {
    return;
  }

  // do the merging
  mergeDigits();

  // send the merged digits
  int nSent = 0;
  for (auto& d : buffers[previousBufId].digits) {
    if (!d.merged && (d.digit.getPadID() >= 0)) {
      sendDigit(d.digit);
      nSent += 1;
    }
  }
  if (mPrint) {
    std::cout << "[FeeIdMerger] sent " << nSent << " digits for orbit " << buffers[previousBufId].orbit << "  current orbit is " << orbit << std::endl;
  }

  currentBufId = 1 - currentBufId;
  previousBufId = 1 - previousBufId;
  buffers[currentBufId].digits.clear();
  buffers[currentBufId].orbit = orbit;
}

// helper function to check if two digits correspond to the same pad;
static bool areSamePad(const MergerDigit& d1, const MergerDigit& d2)
{
  if (d1.solarId != d2.solarId)
    return false;
  if (d1.dsAddr != d2.dsAddr)
    return false;
  if (d1.chAddr != d2.chAddr)
    return false;
  if (d1.digit.getDetID() != d2.digit.getDetID())
    return false;
  if (d1.digit.getPadID() != d2.digit.getPadID())
    return false;

  return true;
}

void FeeIdMerger::mergeDigits()
{
  const uint32_t bxCounterRollover = 0x100000;

  auto updateDigits = [](MergerDigit& d1, MergerDigit& d2) -> bool {
    // skip digits that are already merged
    if (d1.merged || d2.merged) {
      return false;
    }

    // skip all digits not matching the detId/padId
    if (!areSamePad(d1, d2)) {
      return false;
    }

    // compute time difference
    Digit::Time startTime = d1.digit.getTime();
    uint32_t bxStart = startTime.bunchCrossing;
    Digit::Time stopTime = d2.stopTime;
    uint32_t bxStop = stopTime.bunchCrossing;
    // correct for value rollover
    if (bxStart < bxStop) {
      bxStart += bxCounterRollover;
    }

    uint32_t stopTimeFull = bxStop + (stopTime.sampaTime << 2);
    uint32_t startTimeFull = bxStart + (startTime.sampaTime << 2);
    uint32_t timeDiff = startTimeFull - stopTimeFull;

    if (mPrint) {
      std::cout << "updateDigits: " << d1.digit.getDetID() << "  " << d1.digit.getPadID() << "  " << bxStop << "  " << stopTime.sampaTime
                << "  " << bxStart << "  " << startTime.sampaTime << "  " << timeDiff << std::endl;
    }

    // skip if the time difference is not equal to 1 ADC clock cycle
    if (timeDiff != 1) {
      return false;
    }

    // merge digits
    d2 += d1;
    d1.merged = true;

    return true;
  };

  auto& currentBuffer = getCurrentBuffer();
  auto currentBufId = getCurrentBufId();
  auto& previousBuffer = getPreviousBuffer();
  auto previousBufId = getPreviousBufId();

  if (mPrint) {
    std::cout << "Merging digits in " << previousBufId << " (orbit=" << previousBuffer.orbit << ")\n";
  }

  for (size_t i = 0; i < previousBuffer.digits.size(); i++) {
    MergerDigit& d1 = previousBuffer.digits[i];

    // skip digits that do not start at the beginning of the time window
    Digit::Time startTime = d1.digit.getTime();
    if (startTime.sampaTime != 0) {
      continue;
    }

    for (size_t j = 0; j < previousBuffer.digits.size(); j++) {
      if (i == j) {
        continue;
      }
      MergerDigit& d2 = previousBuffer.digits[j];
      if (updateDigits(d1, d2)) {
        break;
      }
    }
  }

  // only merge digits from consecutive orbits
  uint32_t orbit_p = previousBuffer.orbit;
  uint32_t orbit_c = currentBuffer.orbit;
  if (mPrint) {
    std::cout << "orbit_c: " << orbit_c << "  orbit_p: " << orbit_p << std::endl;
  }
  if ((orbit_c >= orbit_p) && ((orbit_c - orbit_p) > 1)) {
    return;
  }

  if (mPrint) {
    std::cout << "Merging digits from " << currentBufId << " (orbit=" << currentBuffer.orbit << ") into "
              << previousBufId << " (orbit=" << previousBuffer.orbit << ")\n";
  }

  for (size_t i = 0; i < currentBuffer.digits.size(); i++) {
    MergerDigit& d1 = currentBuffer.digits[i];

    // skip digits that do not start at the beginning of the time window
    Digit::Time startTime = d1.digit.getTime();
    if (startTime.sampaTime != 0) {
      continue;
    }

    for (size_t j = 0; j < previousBuffer.digits.size(); j++) {
      MergerDigit& d2 = previousBuffer.digits[j];
      if (updateDigits(d1, d2)) {
        break;
      }
    }
  }
}

void Merger::setOrbit(int feeId, uint32_t orbit, bool stop)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  mergers[feeId].setOrbit(orbit, stop);
}

void Merger::setDigitHandler(std::function<void(const Digit&)> h)
{
  for (int feeId = 0; feeId <= MCH_MERGER_FEEID_MAX; feeId++) {
    mergers[feeId].setDigitHandler(h);
  }
}

void Merger::addDigit(int feeId, int solarId, int dsAddr, int chAddr,
                      int deId, int padId, int adc, Digit::Time time, uint16_t nSamples)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  Digit::Time stopTime;
  stopTime.sampaTime = time.sampaTime + nSamples - 1;
  stopTime.bunchCrossing = time.bunchCrossing;
  stopTime.orbit = time.orbit;

  mergers[feeId].getCurrentBuffer().digits.emplace_back(MergerDigit{o2::mch::Digit(deId, padId, adc, time, nSamples),
                                                                    stopTime, false, solarId, dsAddr, chAddr});
}

void Merger::mergeDigits(int feeId)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  mergers[feeId].mergeDigits();
}

} // namespace raw
} // namespace mch
} // end namespace o2
