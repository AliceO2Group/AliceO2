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
#include "EMCALBase/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <TRandom.h>
#include <climits>
#include <forward_list>
#include "FairLogger.h" // for LOG

ClassImp(o2::EMCAL::Digitizer);

using o2::EMCAL::Digit;
using o2::EMCAL::Hit;

using namespace o2::EMCAL;

//_______________________________________________________________________
void Digitizer::init() {}

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
      Int_t id = digit.GetTower();

      if (id < 0 || id > mGeometry->GetNCells()) {
        LOG(WARNING) << "tower index out of range: " << id << FairLogger::endl;
        continue;
      }

      Bool_t flag = false;
      for (auto& digit0 : mDigits[id]) {
        if (digit0.canAdd(digit)) {
          digit0 += digit;
          LabelIndex = digit0.GetLabel();
          flag = true;
          break;
        }
      }

      if (!flag) {
        mDigits[id].push_front(digit);
      }

      o2::EMCAL::MCLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, mEventTime);
      mMCTruthContainer.addElementRandomAccess(LabelIndex, label);
      auto labels = mMCTruthContainer.getLabels(LabelIndex);
      std::sort(labels.begin(), labels.end());
    } catch (InvalidPositionException& e) {
      LOG(ERROR) << "Error in creating the digit: " << e.what() << FairLogger::endl;
    }
  }

  fillOutputContainer(digits);
}

//_______________________________________________________________________
o2::EMCAL::Digit Digitizer::hitToDigit(const Hit& hit, const Int_t label)
{
  Int_t tower = hit.GetDetectorID();
  Double_t amplitude = hit.GetEnergyLoss();
  Digit digit(tower, amplitude, mEventTime, label);
  return digit;
}

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t < mEventTime && mContinuous) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")" << FairLogger::endl;
  }
  mEventTime = t;
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  std::forward_list<Digit> l;

  for (auto tower : mDigits) {
    for (auto& digit : tower.second) {
      l.push_front(digit);
    }
  }

  l.sort();

  for (auto digit : l) {
    digits.push_back(digit);
  }

  mMCTruthOutputContainer.clear();
  for (int index = 0; index < mMCTruthContainer.getIndexedSize(); ++index) {
    mMCTruthOutputContainer.addElements(index, mMCTruthContainer.getLabels(index));
  }
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID()
               << FairLogger::endl;
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID()
               << FairLogger::endl;
  }
  mCurrEvID = v;
}
