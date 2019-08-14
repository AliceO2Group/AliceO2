// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/Digitizer.h"
#include "EMCALSimulation/SimParam.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <climits>
#include <forward_list>
#include <chrono>
#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::emcal::Digitizer);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void Digitizer::init()
{
  mSimParam = SimParam::GetInstance();
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits)
{
  digits.clear();
  mDigits.clear();
  mMCTruthContainer.clear();

  for (auto hit : hits) {
    try {
      Int_t LabelIndex = mMCTruthContainer.getIndexedSize();
      Digit digit = hitToDigit(hit, LabelIndex);
      Int_t id = digit.getTower();

      if (id < 0 || id > mGeometry->GetNCells()) {
        LOG(WARNING) << "tower index out of range: " << id;
        continue;
      }

      Bool_t flag = false;
      for (auto& digit0 : mDigits[id]) {
        if (digit0.canAdd(digit)) {
          digit0 += digit;
          //LabelIndex = digit0.GetLabel();
          flag = true;
          break;
        }
      }

      if (!flag) {
        mDigits[id].push_front(digit);
      }

      o2::MCCompLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, false);
      mMCTruthContainer.addElementRandomAccess(LabelIndex, label);
    } catch (InvalidPositionException& e) {
      LOG(ERROR) << "Error in creating the digit: " << e.what();
    }
  }

  fillOutputContainer(digits);
}

//_______________________________________________________________________
o2::emcal::Digit Digitizer::hitToDigit(const Hit& hit, const Int_t label)
{
  Int_t tower = hit.GetDetectorID();
  Double_t amplitude = hit.GetEnergyLoss();
  Digit digit(tower, amplitude, mEventTime);
  return digit;
}

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t < mEventTime && mContinuous) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")";
  }
  mEventTime = t;
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  std::forward_list<Digit> l;

  for (auto tower : mDigits) {
    for (auto& digit : tower.second) {
      if (mRemoveDigitsBelowThreshold && (digit.getEnergy() < mSimParam->GetDigitThreshold() * (constants::EMCAL_ADCENERGY))) {
        continue;
      }

      if (mSmearTimeEnergy) {
        smearTimeEnergy(digit);
      }

      l.push_front(digit);
    }
  }

  l.sort();

  for (auto digit : l) {
    digits.push_back(digit);
  }
}

//_______________________________________________________________________
void Digitizer::smearTimeEnergy(Digit& digit)
{
  Double_t energy = digit.getEnergy();
  Double_t fluct = (energy * mSimParam->GetMeanPhotonElectron()) / mSimParam->GetGainFluctuations();
  energy *= mRandomGenerator->Poisson(fluct) / fluct;
  energy += mRandomGenerator->Gaus(0., mSimParam->GetPinNoise());
  digit.setEnergy(energy);

  Double_t res = mSimParam->GetTimeResolution(energy);
  if (res > 0.) {
    digit.setTimeStamp(mRandomGenerator->Gaus(digit.getTimeStamp(), res));
  }
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID();
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID();
  }
  mCurrEvID = v;
}
