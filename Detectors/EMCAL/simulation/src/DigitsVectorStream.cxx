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

#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <iostream>
#include <gsl/span>
#include <fairlogger/Logger.h>
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsVectorStream.h"
#include "CommonConstants/Triggers.h"
#include "SimConfig/DigiParams.h"

using namespace o2::emcal;

void DigitsVectorStream::init()
{
  mSimParam = &(o2::emcal::SimParam::Instance());
  auto randomSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  if (o2::conf::DigiParams::Instance().seed != 0) {
    randomSeed = o2::conf::DigiParams::Instance().seed;
  }
  mRandomGenerator = new TRandom3(randomSeed);

  mRemoveDigitsBelowThreshold = mSimParam->doRemoveDigitsBelowThreshold();
  mSimulateNoiseDigits = mSimParam->doSimulateNoiseDigits();
}

//_______________________________________________________________________
void DigitsVectorStream::addNoiseDigits(LabeledDigit& d1)
{
  double amplitude = d1.getAmplitude();
  double sigmaHG = mSimParam->getPinNoise();
  double sigmaLG = mSimParam->getPinNoiseLG();

  uint16_t noiseHG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaHG) / constants::EMCAL_ADCENERGY));                                 // ADC
  uint16_t noiseLG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaLG) / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR))); // ADC

  MCLabel label(true, 1.0);
  LabeledDigit d(d1.getTower(), noiseLG, noiseHG, d1.getTimeStamp(), label);

  d1 += d;
}

//_______________________________________________________________________
void DigitsVectorStream::fill(std::deque<o2::emcal::DigitTimebin>& digitlist, o2::InteractionRecord record)
{
  std::map<unsigned int, std::list<LabeledDigit>> outputList;

  for (auto& digitsTimeBin : digitlist) {

    for (auto& [tower, digitsList] : *digitsTimeBin.mDigitMap) {

      if (digitsList.size() == 0) {
        continue;
      }
      digitsList.sort();

      int digIndex = 0;
      for (auto& ld : digitsList) {

        // Loop over all digits in the time sample and sum the digits that belongs to the same tower and falls in one time bin
        int digIndex1 = 0;
        for (auto ld1 = digitsList.begin(); ld1 != digitsList.end(); ++ld1) {

          if (digIndex == digIndex1) {
            digIndex1++;
            continue;
          }

          if (ld.canAdd(*ld1)) {
            ld += *ld1;
            digitsList.erase(ld1--);
          }
          digIndex1++;
        }

        if (mSimulateNoiseDigits) {
          addNoiseDigits(ld);
        }

        if (mRemoveDigitsBelowThreshold && (ld.getAmplitude() < (mSimParam->getDigitThreshold() * constants::EMCAL_ADCENERGY))) {
          continue;
        }
        if (ld.getAmplitude() < 0) {
          continue;
        }
        if ((ld.getTimeStamp() >= mSimParam->getLiveTime()) || (ld.getTimeStamp() < 0)) {
          continue;
        }

        outputList[tower].push_back(ld);
        digIndex++;
      }
    }
  }

  unsigned int numberOfNewDigits = 0;
  for (const auto& [tower, outdiglist] : outputList) {
    for (const auto& d : outdiglist) {
      // outdiglist.sort();

      Digit digit = d.getDigit();
      std::vector<MCLabel> labels = d.getLabels();
      mDigits.push_back(digit);
      numberOfNewDigits++;

      Int_t LabelIndex = mLabels.getIndexedSize();
      for (const auto& label : labels) {
        mLabels.addElementRandomAccess(LabelIndex, label);
      }
    }
  }

  mTriggerRecords.emplace_back(record, o2::trigger::PhT, mStartIndex, numberOfNewDigits);
  mStartIndex = mDigits.size();

  LOG(info) << "Trigger Orbit " << record.orbit << ", BC " << record.bc << ": have " << numberOfNewDigits << " digits (" << mStartIndex << " total)";
}

//_______________________________________________________________________
void DigitsVectorStream::clear()
{
  mDigits.clear();
  mLabels.clear();
  mTriggerRecords.clear();
  mStartIndex = 0;
}