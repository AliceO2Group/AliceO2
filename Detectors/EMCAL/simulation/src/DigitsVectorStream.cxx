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
#include "FairLogger.h"
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsVectorStream.h"
#include "CommonConstants/Triggers.h"

using namespace o2::emcal;

void DigitsVectorStream::init()
{
  mSimParam = &(o2::emcal::SimParam::Instance());

  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  mRemoveDigitsBelowThreshold = mSimParam->doRemoveDigitsBelowThreshold();
  mSimulateNoiseDigits = mSimParam->doSimulateNoiseDigits();
}

//_______________________________________________________________________
void DigitsVectorStream::addNoiseDigits(LabeledDigit& d1)
{
  double amplitude = d1.getAmplitude();
  double sigma = mSimParam->getPinNoise();
  if (amplitude > constants::EMCAL_HGLGTRANSITION * constants::EMCAL_ADCENERGY) {
    sigma = mSimParam->getPinNoiseLG();
  }

  double noise = std::abs(mRandomGenerator->Gaus(0, sigma));
  MCLabel label(true, 1.0);
  LabeledDigit d(d1.getTower(), noise, d1.getTimeStamp(), label);
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

      for (auto& ld : digitsList) {

        // Loop over all digits in the time sample and sum the digits that belongs to the same tower and falls in one time bin
        for (auto ld1 = digitsList.begin(); ld1 != digitsList.end(); ++ld1) {

          if (ld == *ld1) {
            continue;
          }

          std::vector<decltype(digitsList.begin())> toDelete;

          if (ld.canAdd(*ld1)) {
            ld += *ld1;
            toDelete.push_back(ld1);
          }
          for (auto del : toDelete) {
            digitsList.erase(del);
          }
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

  LOG(info) << "Have " << mStartIndex << " digits ";
}

//_______________________________________________________________________
void DigitsVectorStream::clear()
{
  mDigits.clear();
  mLabels.clear();
  mTriggerRecords.clear();
  mStartIndex = 0;
}