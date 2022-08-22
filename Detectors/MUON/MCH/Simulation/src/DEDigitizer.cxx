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

#include "MCHSimulation/DigitizerParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include <algorithm>
#include <cmath>
#include <random>

using o2::math_utils::Point3D;

namespace o2::mch
{

DEDigitizer::DEDigitizer(int deId, o2::math_utils::Transform3D transformation)
  : mDeId{deId},
    mResponse{deId < 300 ? Station::Type1 : Station::Type2345},
    mTransformation{transformation},
    mSegmentation{o2::mch::mapping::segmentation(deId)},
    mCharges(mSegmentation.nofPads()),
    mLabels(mSegmentation.nofPads())
{
}

void DEDigitizer::startCollision(o2::InteractionRecord collisionTime)
{
  mIR = collisionTime;
  clear();
}

void DEDigitizer::process(const Hit& hit, int evID, int srcID)
{
  MCCompLabel label(hit.GetTrackID(), evID, srcID);

  // convert energy to charge
  auto charge = mResponse.etocharge(hit.GetEnergyLoss());
  auto chargeCorr = mResponse.chargeCorr();
  auto chargeBending = chargeCorr * charge;
  auto chargeNonBending = charge / chargeCorr;

  // local position of the charge distribution
  Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());
  Point3D<float> lpos;
  mTransformation.MasterToLocal(pos, lpos);
  auto localX = mResponse.getAnod(lpos.X());
  auto localY = lpos.Y();

  // borders of charge integration area
  auto dxy = mResponse.getSigmaIntegration() * mResponse.getChargeSpread();
  auto xMin = localX - dxy;
  auto xMax = localX + dxy;
  auto yMin = localY - dxy;
  auto yMax = localY + dxy;

  // loop over all pads within the defined bounding box to compute the charge on each of them
  mSegmentation.forEachPadInArea(xMin, yMin, xMax, yMax, [&](int padid) {
    auto dx = mSegmentation.padSizeX(padid) * 0.5;
    auto dy = mSegmentation.padSizeY(padid) * 0.5;
    auto xPad = mSegmentation.padPositionX(padid) - localX;
    auto yPad = mSegmentation.padPositionY(padid) - localY;
    auto q = mResponse.chargePadfraction(xPad - dx, xPad + dx, yPad - dy, yPad + dy);
    if (mResponse.isAboveThreshold(q)) {
      q *= mSegmentation.isBendingPad(padid) ? chargeBending : chargeNonBending;
      if (q > 0.f) {
        mCharges[padid] += q;
        mLabels[padid].emplace_back(label);
      }
    }
  });
}

void DEDigitizer::addNoise(float noiseProba)
{
  std::random_device rd;
  std::mt19937 mt(rd());

  float mean = noiseProba * mSegmentation.nofPads();
  float sigma = mean / std::sqrt(mean);

  std::normal_distribution<float> gaus(mean, sigma);

  int nofNoisyPads = std::ceil(gaus(mt));

  std::uniform_int_distribution<int> ids(0, mSegmentation.nofPads() - 1);

  float chargeNoise = 1.2;
  // FIXME: draw this also from some distribution (according to
  // some parameters in DigitizerParam)
  for (auto i = 0; i < nofNoisyPads; i++) {
    auto padid = ids(mt);
    mCharges[padid] += chargeNoise;
    mLabels[padid].emplace_back(true);
  }
}

void DEDigitizer::extractDigitsAndLabels(std::vector<Digit>& digits,
                                         o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  int dataindex = labels.getIndexedSize();
  for (auto padid = 0; padid < mCharges.size(); ++padid) {
    auto q = mCharges[padid];
    if (q <= 0.f) {
      continue;
    }
    uint32_t adc = std::round(q);
    if (adc >= DigitizerParam::Instance().minADC) {
      auto time = mIR.differenceInBC(o2::raw::HBFUtils::Instance().orbitFirst);
      auto nSamples = mResponse.nSamples(adc);
      nSamples = std::min(nSamples, 0x3FFU); // the number of samples must fit within 10 bits
      bool saturated = false;
      // the charge sum must fit within 20 bits
      // FIXME: we should better handle charge saturation here
      if (adc > 0xFFFFFU) {
        adc = 0xFFFFFU;
        saturated = true;
      }
      digits.emplace_back(mDeId, padid, adc, time, nSamples, saturated);
      for (auto element : mLabels[padid]) {
        labels.addElement(dataindex, element);
      }
      ++dataindex;
    }
  }
}

void DEDigitizer::clear()
{
  std::fill(mCharges.begin(), mCharges.end(), 0);
  for (auto& label : mLabels) {
    label.clear();
  }
}

} // namespace o2::mch
