// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/DEDigitizer.h"
#include "DetectorsRaw/HBFUtils.h"
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
  Point3D<float> pos(hit.GetX(), hit.GetY(), hit.GetZ());

  //convert energy to charge
  auto charge = mResponse.etocharge(hit.GetEnergyLoss());

  auto time = mIR.differenceInBC(o2::raw::HBFUtils::Instance().orbitFirst);
  // digit time will disappear anyway ? (same information in ROFRecord)

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
    if (mResponse.aboveThreshold(q)) {
      if (mSegmentation.isBendingPad(padid)) {
        q *= chargebend;
      } else {
        q *= chargenon;
      }
      if (q > 0) {
        mCharges[padid] += q;
        mLabels[padid].emplace_back(label);
      }
    }
  }
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
    if (q <= 0) {
      continue;
    }
    // FIXME: this is just to compare with previous MCHDigitizer
    auto signal = (uint32_t)q * mResponse.getInverseChargeThreshold();
    auto padc = signal * mResponse.getChargeThreshold();
    auto adc = mResponse.response(TMath::Nint(padc));
    if (adc > 0) {
      auto time = mIR.differenceInBC(o2::raw::HBFUtils::Instance().orbitFirst);
      digits.emplace_back(mDeId, padid, adc, time, 1);
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
