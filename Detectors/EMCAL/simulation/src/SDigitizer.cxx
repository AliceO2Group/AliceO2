// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/SDigitizer.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <climits>
#include <list>
#include <chrono>
#include <numeric>
#include "FairLogger.h" // for LOG

ClassImp(o2::emcal::SDigitizer);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
std::vector<o2::emcal::LabeledDigit> SDigitizer::process(const std::vector<Hit>& hits)
{

  std::map<int, std::map<int, std::vector<o2::emcal::Hit>>> hitsPerTowerPerParticleID;

  // will be used to sort digits and labels by tower
  std::unordered_map<Int_t, std::vector<LabeledDigit>> digitsPerTower;

  for (auto hit : hits) {
    hitsPerTowerPerParticleID[hit.GetDetectorID()][hit.GetTrackID()].push_back(hit);
  }

  std::vector<o2::emcal::Hit> SHits;
  for (auto [towerID, hitsParticle] : hitsPerTowerPerParticleID) {
    for (auto [partID, Hits] : hitsParticle) {
      o2::emcal::Hit SHit = std::accumulate(std::next(Hits.begin()), Hits.end(), Hits.front());
      SHits.push_back(SHit);
    }
  }

  for (auto hit : SHits) {
    try {

      Int_t tower = hit.GetDetectorID();

      if (tower < 0 || tower > mGeometry->GetNCells()) {
        LOG(WARNING) << "tower index out of range: " << tower;
        continue;
      }

      Double_t energy = hit.GetEnergyLoss();

      //@TODO check if the summed digit time is set correctly
      Digit digit(tower, energy, hit.GetTime());

      MCLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, false, 1.0);
      if (digit.getAmplitude() == 0) {
        label.setAmplitudeFraction(0);
      }

      // Check whether the digit is high gain or low gain
      if (digit.getAmplitude() > constants::EMCAL_HGLGTRANSITION * constants::EMCAL_ADCENERGY) {
        digit.setLowGain();
      } else {
        digit.setHighGain();
      }

      LabeledDigit d(digit, label);
      digitsPerTower[tower].push_back(d);

    } catch (InvalidPositionException& e) {
      LOG(ERROR) << "Error in creating the digit: " << e.what();
    }
  }

  std::vector<LabeledDigit> digitsVector;

  // Assigning a channel type LG or HG
  for (auto t : digitsPerTower) {
    std::vector<LabeledDigit> digitsList = t.second;

    bool channelLowGain = false;

    // If the channel type is LG only keep low gain digits and discard the rest
    for (auto digit : digitsList) {
      if (digit.getLowGain()) {
        channelLowGain = true;
        break;
      }
    }

    for (auto digit : digitsList) {
      if (digit.getAmplitude() < 0) {
        continue;
      }

      if (digit.getLowGain() == channelLowGain) {
        digitsVector.push_back(digit);
      }
    }
  }

  digitsPerTower.clear();

  return digitsVector;
}

//_______________________________________________________________________
void SDigitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID();
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void SDigitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID();
  }
  mCurrEvID = v;
}
