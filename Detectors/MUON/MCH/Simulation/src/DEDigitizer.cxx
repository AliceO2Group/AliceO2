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
#include "CommonConstants/LHCConstants.h"
#include "DetectorsRaw/HBFUtils.h"
#include <cmath>
#include <random>

using o2::math_utils::Point3D;

o2::InteractionRecord ir2sampaIR(o2::InteractionRecord collisionTime)
{
  collisionTime.bc = collisionTime.bc - collisionTime.bc % 4;
  return collisionTime;
}

namespace o2::mch
{

DEDigitizer::DEDigitizer(int deId,
                         o2::math_utils::Transform3D transformation,
                         float timeSpread,
                         float noiseChargeMean,
                         float noiseChargeSigma,
                         int seed)
  : mDeId{deId},
    mResponse{deId < 300 ? Station::Type1 : Station::Type2345},
    mTransformation{transformation},
    mSegmentation{o2::mch::mapping::segmentation(deId)},
    mTimeDist{0.0, timeSpread},
    mChargeDist{noiseChargeMean, noiseChargeSigma},
    mGene{seed == 0 ? std::random_device{}() : seed}
{
}

void DEDigitizer::addNoise(float noiseProba,
                           const o2::InteractionRecord& firstIR,
                           const o2::InteractionRecord& lastIR)
{
  float mean = noiseProba * mSegmentation.nofPads();

  std::poisson_distribution<int> nofPadsDist(mean);
  std::uniform_int_distribution<int> ids(0, mSegmentation.nofPads() - 1);

  for (auto rof = firstIR; rof < lastIR; rof += 4) {
    // draw a number of noisy pad for this IR
    int nofNoisyPads = nofPadsDist(mGene);
    for (auto i = 0; i < nofNoisyPads; i++) {
      auto padid = ids(mGene);
      float chargeNoise = mChargeDist(mGene);
      appendDigit(rof, padid, chargeNoise, MCCompLabel{true});
    }
  }
}

void DEDigitizer::process(const o2::InteractionRecord& collisionTime, const Hit& hit, int evID, int srcID)
{
  MCCompLabel label(hit.GetTrackID(), evID, srcID);
  Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());

  //convert energy to charge
  auto charge = mResponse.etocharge(hit.GetEnergyLoss());

  //transformation from global to local
  Point3D<float> lpos;
  mTransformation.MasterToLocal(pos, lpos);

  auto anodpos = mResponse.getAnod(lpos.X());
  auto fracplane = mResponse.chargeCorr();
  auto chargebend = fracplane * charge;
  auto chargenon = charge / fracplane;

  //borders of charge gen.
  auto xMin = anodpos - mResponse.getQspreadX() * mResponse.getSigmaIntegration() * 0.5;
  auto xMax = anodpos + mResponse.getQspreadX() * mResponse.getSigmaIntegration() * 0.5;
  auto yMin = lpos.Y() - mResponse.getQspreadY() * mResponse.getSigmaIntegration() * 0.5;
  auto yMax = lpos.Y() + mResponse.getQspreadY() * mResponse.getSigmaIntegration() * 0.5;

  //get segmentation for detector element
  auto localX = anodpos;
  auto localY = lpos.Y();

  //get area for signal induction from segmentation
  //single pad as check
  int padidbendcent = 0;
  int padidnoncent = 0;
  int ndigits = 0;

  bool padexists = mSegmentation.findPadPairByPosition(localX, localY, padidbendcent, padidnoncent);
  if (!padexists) {
    LOGP(warning, "Did not find _any_ pad for localX,Y={},{} for DeId {}", localX, localY, mDeId);
    return;
  }

  std::vector<int> padids;

  // get all pads within the defined bounding box ...
  mSegmentation.forEachPadInArea(xMin, yMin, xMax, yMax, [&padids](int padid) {
    padids.emplace_back(padid);
  });

  // ... and loop over all those pads to compute the charge on each of them
  for (auto padid : padids) {
    auto dx = mSegmentation.padSizeX(padid) * 0.5;
    auto dy = mSegmentation.padSizeY(padid) * 0.5;
    auto xmin = (localX - mSegmentation.padPositionX(padid)) - dx;
    auto xmax = xmin + 2 * dx;
    auto ymin = (localY - mSegmentation.padPositionY(padid)) - dy;
    auto ymax = ymin + 2 * dy;
    auto q = mResponse.chargePadfraction(xmin, xmax, ymin, ymax);
    if (mResponse.isAboveThreshold(q)) {
      if (mSegmentation.isBendingPad(padid)) {
        q *= chargebend;
      } else {
        q *= chargenon;
      }
      if (q > 0) {
        // shift the initial ROF time by some random bc value
        // so that each digits gets its own ROF (close to the initial one
        // but not quite equal to that original one)
        o2::InteractionRecord ir = shiftTime(collisionTime);
        appendDigit(ir, padid, q, label);
      }
    }
  }
}

int64_t DEDigitizer::drawRandomTimeShift()
{
  return std::round(mTimeDist(mGene) / constants::lhc::LHCBunchSpacingNS);
}

o2::InteractionRecord DEDigitizer::shiftTime(o2::InteractionRecord ir)
{
  auto tshift = drawRandomTimeShift();
  if (tshift > 0) {
    ir += tshift;
  } else {
    ir -= -tshift;
  }
  return ir;
}

void DEDigitizer::appendDigit(o2::InteractionRecord ir, int padid, float charge, const MCCompLabel& label)
{
  // ensure that time is aligned to 4-BCs
  ir = ir2sampaIR(ir);

  auto& pads = mPadMap[ir]; // get the set of pads for that time

  // search if we already have that pad (identified by its padid) in that set
  auto f = std::find_if(pads.begin(), pads.end(), [padid](const Pad& pad) {
    return pad.padid == padid;
  });
  if (f != pads.end()) {
    // pad is already present, let's merge the charge
    (*f).charge += charge;
    (*f).labels.push_back(label);
  } else {
    // otherwise create a new pad
    pads.emplace_back(padid, charge, label);
  }
}

void DEDigitizer::clear()
{
  mPadMap.clear();
}

void DEDigitizer::extractRofs(std::set<o2::InteractionRecord>& rofs)
{
  for (const auto& p : mPadMap) {
    rofs.emplace(p.first.bc, p.first.orbit);
  }
}

void DEDigitizer::extractDigitsAndLabels(const o2::InteractionRecord& rof,
                                         std::vector<Digit>& digits,
                                         o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  auto& pads = mPadMap[rof];
  if (pads.empty()) {
    return;
  }
  int dataindex = labels.getIndexedSize();
  for (auto pad : pads) {
    auto q = pad.charge;
    if (q <= 0) {
      continue;
    }
    auto adc = std::round(q); // FIXME: trivial, should we allow for a scale factor here ?
    if (adc > 1023) {
      adc = 1023;
    }
    if (adc > 0) {
      auto time = rof.differenceInBC(o2::raw::HBFUtils::Instance().orbitFirst);
      digits.emplace_back(mDeId, pad.padid, adc, time, 1);
      for (auto element : pad.labels) {
        labels.addElement(dataindex, element);
      }
      ++dataindex;
    }
  }
}

} // namespace o2::mch
