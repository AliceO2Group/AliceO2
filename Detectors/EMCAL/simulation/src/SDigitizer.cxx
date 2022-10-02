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
#include <fairlogger/Logger.h> // for LOG

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
        LOG(warning) << "tower index out of range: " << tower;
        continue;
      }

      Double_t energy = hit.GetEnergyLoss();

      //@TODO check if the summed digit time is set correctly
      Digit digit(tower, energy, hit.GetTime());

      MCLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, false, 1.0);
      if (digit.getAmplitude() < __DBL_EPSILON__) {
        label.setAmplitudeFraction(0);
      }
      LabeledDigit d(digit, label);

      digitsPerTower[tower].push_back(d);

    } catch (InvalidPositionException& e) {
      LOG(error) << "Error in creating the digit: " << e.what();
    }
  }

  std::vector<LabeledDigit> digitsVector;

  // Sum all digits in one tower
  for (auto [towerID, labeledDigits] : digitsPerTower) {

    o2::emcal::LabeledDigit Sdigit = std::accumulate(std::next(labeledDigits.begin()), labeledDigits.end(), labeledDigits.front());

    if (Sdigit.getAmplitude() < __DBL_EPSILON__) {
      continue;
    }

    digitsVector.push_back(Sdigit);
  }

  digitsPerTower.clear();

  return digitsVector;
}

//_______________________________________________________________________
void SDigitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(fatal) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID();
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void SDigitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(fatal) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID();
  }
  mCurrEvID = v;
}
