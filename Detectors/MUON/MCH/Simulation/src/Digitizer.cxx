// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Digitizer.h"

#include "MCHMappingInterface/Segmentation.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/Response.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include <fairlogger/Logger.h>

#include <iostream>
using namespace std;

using namespace o2::mch;

namespace
{

std::map<int, int> createDEMap()
{
  std::map<int, int> m;
  int i{0};
  o2::mch::mapping::forEachDetectionElement([&m, &i](int deid) {
    m[deid] = i++;
  });
  return m;
}

int deId2deIndex(int detElemId)
{
  static std::map<int, int> m = createDEMap();
  return m[detElemId];
}

std::vector<o2::mch::mapping::Segmentation> createSegmentations()
{
  std::vector<o2::mch::mapping::Segmentation> segs;
  o2::mch::mapping::forEachDetectionElement([&segs](int deid) {
    segs.emplace_back(deid);
  });
  return segs;
}

const o2::mch::mapping::Segmentation& segmentation(int detElemId)
{
  static auto segs = createSegmentations();
  return segs[deId2deIndex(detElemId)];
}

bool isStation1(int detID)
{
  return detID < 300;
}

Response& response(bool isStation1)
{
  static std::array<Response, 2> resp = {Response(Station::Type2345), Response(Station::Type1)};
  return resp[isStation1];
}

int getGlobalDigit(int detID, int padID)
{
  //calculate global index
  return detID * 100000 + padID;
}

} // namespace

Digitizer::Digitizer(int) {}

void Digitizer::init()
{
}

//______________________________________________________________________
void Digitizer::process(const std::vector<Hit> hits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer)
{
  digits.clear();
  mDigits.clear();
  mTrackLabels.clear();
  mcContainer.clear();
  //array of MCH hits for a given simulated event
  for (auto& hit : hits) {
    int detID = hit.GetDetectorID();
    int ndigits = processHit(hit, detID, mEventTime);
    MCCompLabel label(hit.GetTrackID(), mEventID, mSrcID, false);
    for (int i = 0; i < ndigits; ++i) {
      int digitIndex = mDigits.size() - ndigits + i;
      mMCTruthOutputContainer.addElement(digitIndex, label);
    } //loop over digits to generate MCdigits
  }   //loop over hits

  //generate noise-only digits
  if(mNoise) generateNoiseDigits();

  fillOutputContainer(digits);
  provideMC(mcContainer);
}
//______________________________________________________________________
int Digitizer::processHit(const Hit& hit, int detID, int event_time)
{
  math_utils::Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());

  Response& resp = response(isStation1(detID));

  //convert energy to charge
  auto charge = resp.etocharge(hit.GetEnergyLoss());
  
  //convert float ns time to BC counts
  auto time = event_time & int(hit.GetTime()/25.);

  //transformation from global to local
  auto t = o2::mch::getTransformation(detID, *gGeoManager);
  math_utils::Point3D<float> lpos;
  t.MasterToLocal(pos, lpos);

  auto anodpos = resp.getAnod(lpos.X());
  auto fracplane = resp.chargeCorr();
  auto chargebend = fracplane * charge;
  auto chargenon = charge / fracplane;

  //borders of charge gen.
  auto xMin = anodpos - resp.getQspreadX() * resp.getSigmaIntegration() * 0.5;
  auto xMax = anodpos + resp.getQspreadX() * resp.getSigmaIntegration() * 0.5;
  auto yMin = lpos.Y() - resp.getQspreadY() * resp.getSigmaIntegration() * 0.5;
  auto yMax = lpos.Y() + resp.getQspreadY() * resp.getSigmaIntegration() * 0.5;

  //get segmentation for detector element
  auto& seg = segmentation(detID);
  auto localX = anodpos;
  auto localY = lpos.Y();

  //get area for signal induction from segmentation
  //single pad as check
  int padidbendcent = 0;
  int padidnoncent = 0;
  int ndigits = 0;

  bool padexists = seg.findPadPairByPosition(localX, localY, padidbendcent, padidnoncent);
  if (!padexists) {
    LOG(WARNING) << "Did not find  _any_ pad for localX,Y=" << localX << "," << localY << ", for detID: " << detID;
    return 0;
  }

  seg.forEachPadInArea(xMin, yMin, xMax, yMax, [&resp, &digits = this->mDigits, chargebend, chargenon, localX, localY, &seg, &ndigits, detID, time](int padid) {
    auto dx = seg.padSizeX(padid) * 0.5;
    auto dy = seg.padSizeY(padid) * 0.5;
    auto xmin = (localX - seg.padPositionX(padid)) - dx;
    auto xmax = xmin + 2 * dx;
    auto ymin = (localY - seg.padPositionY(padid)) - dy;
    auto ymax = ymin + 2 * dy;
    auto q = resp.chargePadfraction(xmin, xmax, ymin, ymax);
    if (resp.aboveThreshold(q)) {
      if (seg.isBendingPad(padid)) {
        q *= chargebend;
      } else {
        q *= chargenon;
      }
      auto signal = (unsigned long)q * resp.getInverseChargeThreshold();
      if (signal > 0) {
        Digit::Time dtime;
        dtime.sampaTime = static_cast<uint16_t>(time) & 0x3FF;
        digits.emplace_back(detID, padid, signal, dtime);
        ++ndigits;
      }
    }
  });
  return ndigits;
}
//______________________________________________________________________
void Digitizer::generateNoiseDigits()
{

  o2::mch::mapping::forEachDetectionElement([& digits = this->mDigits, &normProbNoise = this->mNormProbNoise,
                                             &eventTime = this->mEventTime, &eventID = this->mEventID,
                                             &srcID = this->mSrcID, &mcTruthOutputContainer = this->mMCTruthOutputContainer](int detID) {
    auto& seg = segmentation(detID);
    auto nPads = seg.nofPads();
    auto nNoisyPadsAv = (float)nPads * normProbNoise;
    int nNoisyPads = TMath::Nint(gRandom->Gaus(nNoisyPadsAv, TMath::Sqrt(nNoisyPadsAv)));
    for (int i = 0; i < nNoisyPads; i++) {
      int padid = gRandom->Integer(nNoisyPads + 1);
      Digit::Time dtime;
      dtime.sampaTime = static_cast<uint16_t>(eventTime) & 0x3FF;
      digits.emplace_back(detID, padid, 0.6, dtime);
      //just to roun adbove threshold when added
      MCCompLabel label(-1, eventID, srcID, true);
      mcTruthOutputContainer.addElement(digits.size() - 1, label);
    }
  });
  //not clear how to normalise to time:
  //assume that only "one" event equivalent,
  //otherwise the probability will strongly depend on read-out-frame time length
}
//______________________________________________________________________
void Digitizer::mergeDigits()
{
  std::vector<int> indices(mDigits.size());
  std::iota(begin(indices), end(indices), 0);
  std::sort(indices.begin(), indices.end(), [& digits = this->mDigits, this](int a, int b) {
    return (getGlobalDigit(digits[a].getDetID(), digits[a].getPadID()) < getGlobalDigit(digits[b].getDetID(), digits[b].getPadID()));
  });

  auto sortedDigits = [digits = this->mDigits, &indices](int i) {
    return digits[indices[i]];
  };

  auto sortedLabels = [labels = this->mTrackLabels, &indices](int i) {
    return labels[indices[i]];
  };

  auto sizedigits = mDigits.size();

  mDigits.clear();
  mDigits.reserve(sizedigits);
  mMCTruthOutputContainer.clear();

  int count = 0;

  int i = 0;
  while (i < indices.size()) {
    int j = i + 1;
    while (j < indices.size() && (getGlobalDigit(sortedDigits(i).getDetID(), sortedDigits(i).getPadID())) == (getGlobalDigit(sortedDigits(j).getDetID(), sortedDigits(j).getPadID())) && (std::fabs(sortedDigits(i).getTime().sampaTime - sortedDigits(j).getTime().sampaTime) < mDeltat)) {
      j++;
    }
    unsigned long adc{0};
    float padc{0};
    Response& resp = response(isStation1(sortedDigits(i).getDetID()));

    for (int k = i; k < j; k++) {
      adc += sortedDigits(k).getADC();
      if (k == i) {
        MCCompLabel label(sortedLabels(i).getTrackID(), sortedLabels(i).getEventID(), sortedLabels(i).getSourceID(), false);
        mMCTruthOutputContainer.addElement(count, label);
      } else {
        if ((sortedLabels(k).getTrackID() != sortedLabels(k - 1).getTrackID()) || (sortedLabels(k).getSourceID() != sortedLabels(k - 1).getSourceID())) {
          MCCompLabel label(sortedLabels(k).getTrackID(), sortedLabels(k).getEventID(), sortedLabels(k).getSourceID(), false);
          mMCTruthOutputContainer.addElement(count, label);
        }
      }
    }
    padc = adc * resp.getChargeThreshold();
    adc = TMath::Nint(padc);
    adc = resp.response(adc);
    mDigits.emplace_back(sortedDigits(i).getDetID(), sortedDigits(i).getPadID(), adc, sortedDigits(i).getTime());
    i = j;
    ++count;
  }
  mDigits.resize(mDigits.size());
}
//______________________________________________________________________
void Digitizer::mergeDigits(std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer)
{
  mDigits.clear();
  mDigits.reserve(digits.size());
  mTrackLabels.clear();

  for (int index = 0; index < digits.size(); ++index) {
    auto digit = digits.at(index);
    mDigits.emplace_back(digit.getDetID(), digit.getPadID(), digit.getADC(), digit.getTime());
  }
  for (int index = 0; index < mcContainer.getNElements(); ++index) {
    auto label = mcContainer.getElement(index);
    mTrackLabels.emplace_back(label.getTrackID(), label.getEventID(), label.getSourceID(), false);
  }

  mergeDigits();
  fillOutputContainer(digits);
  provideMC(mcContainer);
}
//______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  // filling the digit container
  if (mDigits.empty()) {
    return;
  }

  digits.clear();
  digits.reserve(mDigits.size());

  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    digits.emplace_back(*iter);
  }
  mDigits.erase(itBeg, iter);
}
//______________________________________________________________________
void Digitizer::setSrcID(int v)
{
  //set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID();
  }
  mSrcID = v;
}
//______________________________________________________________________
void Digitizer::setEventID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storabel in the label " << MCCompLabel::maxEventID();
  }
  mEventID = v;
}
//______________________________________________________________________
void Digitizer::provideMC(o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer)
{
  //fill MCtruth info
  mcContainer.clear();
  if (mMCTruthOutputContainer.getNElements() == 0) {
    return;
  }

  //need to fill groups of labels not only  single labels, since index in addElements
  // is the data index
  //write a map that fixes?
  for (int index = 0; index < mMCTruthOutputContainer.getIndexedSize(); ++index) {
    mcContainer.addElements(index, mMCTruthOutputContainer.getLabels(index));
  }

  mMCTruthOutputContainer.clear();
}
