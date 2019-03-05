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
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include <fairlogger/Logger.h>

using namespace o2::mch;

namespace
{

std::map<int, int> createDEMap()
{
  std::map<int, int> m;
  int i{ 0 };
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
  return detID < 400;
}

Response& response(bool isStation1)
{
  static std::array<Response, 2> resp = { Response(Station::Type1), Response(Station::Type2345) };
  return resp[isStation1];
}

} // namespace

Digitizer::Digitizer(int) {}

void Digitizer::init()
{
}

//______________________________________________________________________
void Digitizer::process(const std::vector<Hit> hits, std::vector<Digit>& digits)
{
  digits.clear();
  mDigits.clear();
  mTrackLabels.clear();

  //array of MCH hits for a given simulated event
  for (auto& hit : hits) {
    //index for this hit
    int detID = hit.GetDetectorID();
    int ndigits = processHit(hit, detID, mEventTime);
    //TODO need one label per Digit
    // can use nDigit output of processHit
    MCCompLabel label(hit.GetTrackID(), mEventID, mSrcID);
    for (int i = 0; i < ndigits; ++i) {
      int digitIndex = mDigits.size() - ndigits + i;
      mTrackLabels.addElementRandomAccess(digitIndex, label);
    } //loop over digits to generate MCdigits
  }   //loop over hits

  fillOutputContainer(digits);
}
//______________________________________________________________________
int Digitizer::processHit(const Hit& hit, int detID, double event_time)
{
  Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());

  Response& resp = response(isStation1(detID));

  //convert energy to charge
  auto charge = resp.etocharge(hit.GetEnergyLoss());
  auto time = hit.GetTime(); //to be used for pile-up

  //transformation from global to local
  auto t = o2::mch::getTransformation(detID, *gGeoManager);
  Point3D<float> lpos;
  t.MasterToLocal(pos, lpos);

  auto anodpos = resp.getAnod(lpos.X());
  auto fracplane = resp.chargeCorr();
  auto chargebend = fracplane * charge;
  auto chargenon = charge / fracplane;

  //borders of charge gen.
  auto xMin = anodpos - resp.getQspreadX() * 0.5;
  auto xMax = anodpos + resp.getQspreadX() * 0.5;
  auto yMin = lpos.Y() - resp.getQspreadY() * 0.5;
  auto yMax = lpos.Y() + resp.getQspreadY() * 0.5;

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
    LOG(ERROR) << "Did not find  _any_ pad for localX,Y=" << localX << "," << localY;
    return 0;
  }

  seg.forEachPadInArea(xMin, yMin, xMax, yMax, [&resp, &digits = this->mDigits, chargebend, chargenon, localX, localY, &seg, &ndigits ](int padid) {
    auto dx = seg.padSizeX(padid) * 0.5;
    auto dy = seg.padSizeY(padid) * 0.5;
    auto xmin = (localX - seg.padPositionX(padid)) - dx;
    auto xmax = xmin + dx;
    auto ymin = (localY - seg.padPositionY(padid)) - dy;
    auto ymax = ymin + dy;
    auto q = resp.chargePadfraction(xmin, xmax, ymin, ymax);
    if (seg.isBendingPad(padid)) {
      q *= chargebend;
    } else {
      q *= chargenon;
    }
    auto signal = resp.response(q);
    digits.emplace_back(padid, signal);
    ++ndigits;
  });
  return ndigits;
}

//______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  // filling the digit container
  if (mDigits.empty())
    return;

  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    digits.emplace_back(*iter);
  }
  mDigits.erase(itBeg, iter);
  mMCTruthOutputContainer.clear();
  for (int index = 0; index < mTrackLabels.getIndexedSize(); ++index) {
    mMCTruthOutputContainer.addElements(index, mTrackLabels.getLabels(index));
  }
}
//______________________________________________________________________
void Digitizer::setSrcID(int v)
{
  //set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID() << FairLogger::endl;
  }
  mSrcID = v;
}
//______________________________________________________________________
void Digitizer::setEventID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storabel in the label " << MCCompLabel::maxEventID() << FairLogger::endl;
  }
  mEventID = v;
}
//______________________________________________________________________
void Digitizer::provideMC(o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer)
{
  //fill MCtruth info
  mcContainer.clear();
  if (mMCTruthOutputContainer.getNElements() == 0)
    return;

  for (int index = 0; index < mMCTruthOutputContainer.getIndexedSize(); ++index) {
    mcContainer.addElements(index, mMCTruthOutputContainer.getLabels(index));
  }

  mMCTruthOutputContainer.clear();
}
