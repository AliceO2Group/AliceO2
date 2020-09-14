// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FV0Simulation/Digitizer.h"
#include "FV0Base/Geometry.h"
#include "FV0Base/Constants.h"

#include <TRandom.h>
#include <algorithm>

ClassImp(o2::fv0::Digitizer);

using namespace o2::math_utils;
using namespace o2::fv0;

void Digitizer::clear()
{
  mEventId = -1;
  mSrcId = -1;
  for (auto& analogSignal : mPmtChargeVsTime) {
    std::fill_n(std::begin(analogSignal), analogSignal.size(), 0);
  }
}

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << "V0Digitizer::init -> start = ";

  mNBins = FV0DigParam::Instance().waveformNbins;      //Will be computed using detector set-up from CDB
  mBinSize = FV0DigParam::Instance().waveformBinWidth; //Will be set-up from CDB

  for (Int_t detID = 0; detID < Constants::nFv0Channels; detID++) {
    mPmtChargeVsTime[detID].resize(mNBins);
  }

  // set up PMT response function [avg]
  TF1 signalShapeFn("signalShape", "crystalball", 0, 200);
  signalShapeFn.SetParameters(FV0DigParam::Instance().shapeConst,
                              FV0DigParam::Instance().shapeMean,
                              FV0DigParam::Instance().shapeSigma,
                              FV0DigParam::Instance().shapeAlpha,
                              FV0DigParam::Instance().shapeN);

  // PMT response per hit [Global]
  float x = mBinSize;
  mPmtResponseGlobal.resize(mNBins);
  for (Int_t j = 0; j < mPmtResponseGlobal.size(); ++j) {
    mPmtResponseGlobal[j] = signalShapeFn.Eval(x);
    //LOG(INFO)<<x<<"    "<<mPmtResponseGlobal[j];
    x += mBinSize;
  }

  LOG(INFO) << "V0Digitizer::init -> finished";
}

void Digitizer::process(const std::vector<o2::fv0::Hit>& hits)
{
  LOG(INFO) << "[FV0] Digitizer::process(): begin with " << hits.size() << " hits";

  std::vector<int> hitIdx(hits.size());
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  std::sort(std::begin(hitIdx), std::end(hitIdx), [&hits](int a, int b) { return hits[a].GetTrackID() < hits[b].GetTrackID(); });

  Int_t parentIdPrev = -10;
  // use ordered hits
  for (auto ids : hitIdx) {
    const auto& hit = hits[ids];
    Int_t detId = hit.GetDetectorID();
    Double_t hitEdep = hit.GetHitValue() * 1e3; //convert to MeV

    // TODO: check how big is inaccuracy if more than 1 'below-threshold' particles hit the same detector cell
    if (hitEdep < FV0DigParam::Instance().singleMipThreshold) {
      continue;
    }

    float distanceFromXc = 0;
    if (Geometry::instance()->isRing5(detId)) {
      distanceFromXc = getDistFromCellCenter(detId, hit.GetX(), hit.GetY());
    }

    int iChannelPerCell = 0;
    while (iChannelPerCell < 2) { // loop over 2 channels, into which signal from each cell in ring 5 is split
      if (Geometry::instance()->isRing5(detId)) {
        // The first channel number is located counter-clockwise from the cell center
        //   and remains identical to the detector number, the second one is clockwise and incremented by 8
        if (iChannelPerCell == 1) {
          detId += 8;
        }

        // Split signal magnitude to fractions depending on the distance of the hit from the cell center
        hitEdep = (hit.GetHitValue() * 1e3) * getSignalFraction(distanceFromXc, iChannelPerCell == 0);
        //LOG(INFO) << "  detId: " << detId << "-" << iChannelPerCell << " hitEdep: " << hitEdep << " distanceFromXc: " << distanceFromXc;
        ++iChannelPerCell;
      } else {
        iChannelPerCell = 2; // not a ring 5 cell -> don't repeat the loop
      }

      Double_t const nPhotons = hitEdep * DP::N_PHOTONS_PER_MEV;
      float const nPhE = SimulateLightYield(detId, nPhotons);
      float avgMip = float(nPhE / FV0DigParam::Instance().avgPhElectron);
      Float_t const hitTime = hit.GetTime() * 1e9;
      //------------------------------------------------------------------------------
      mPmtResponseTemp = mPmtResponseGlobal;
      Size_t NBinShift = std::lround(hitTime / FV0DigParam::Instance().waveformBinWidth);
      for (int m = 0; m < NBinShift; m++)
        mPmtResponseTemp.push_back(0);
      // rotate the vector element to shift all the elements by hit time
      std::rotate(mPmtResponseTemp.rbegin(), mPmtResponseTemp.rbegin() + NBinShift, mPmtResponseTemp.rend());
      mPmtResponseTemp.resize(FV0DigParam::Instance().waveformNbins);

      for (int mm = 0; mm < mPmtResponseTemp.size(); mm++) {
        mPmtChargeVsTime[detId][mm] += (mPmtResponseTemp[mm] * avgMip);
      }
      // Charged particles in MCLabel
      Int_t const parentId = hit.GetTrackID();
      if (parentId != parentIdPrev) {
        mMCLabels.emplace_back(parentId, mEventId, mSrcId, detId);
        parentIdPrev = parentId;
      }
    }
  } //hit loop
}

void Digitizer::analyseWaveformsAndStore(std::vector<fv0::BCData>& digitsBC,
                                         std::vector<fv0::ChannelData>& digitsCh,
                                         dataformats::MCTruthContainer<fv0::MCLabel>& labels)
{
  // Sum charge of all time bins to get total charge collected for a given channel
  size_t const first = digitsCh.size();
  size_t nStored = 0;
  for (Int_t ipmt = 0; ipmt < Constants::nFv0Channels; ++ipmt) {
    float time = SimulateTimeCfd(ipmt) - FV0DigParam::Instance().timeCompensate;
    if (time < -FV0DigParam::Instance().cfdCheckWindow || time > FV0DigParam::Instance().cfdCheckWindow)
      continue;
    Float_t totalCharge = 0.0f;
    auto const& analogSignal = mPmtChargeVsTime[ipmt];

    int chargeIntMin = FV0DigParam::Instance().isIntegrateFull ? 0 : FV0DigParam::Instance().chargeIntBinMin;
    int chargeIntMax = FV0DigParam::Instance().isIntegrateFull ? FV0DigParam::Instance().waveformNbins : FV0DigParam::Instance().chargeIntBinMax;

    for (Int_t iTimeBin = chargeIntMin; iTimeBin < chargeIntMax; ++iTimeBin) {
      Float_t const timeBinCharge = mPmtChargeVsTime[ipmt][iTimeBin];
      totalCharge += timeBinCharge;
    }
    totalCharge *= DP::INV_CHARGE_PER_ADC;
    time *= DP::INV_TIME_PER_TDCCHANNEL;
    digitsCh.emplace_back(ipmt, static_cast<short int>(std::round(time)), static_cast<short int>(std::round(totalCharge)));
    ++nStored;
  }

  // Send MClabels and digitsBC to storage
  size_t const nBC = digitsBC.size();
  digitsBC.emplace_back(first, nStored, mIntRecord);
  for (auto const& lbl : mMCLabels) {
    labels.addElement(nBC, lbl);
  }
  mMCLabels.clear();
}

// -------------------------------------------------------------------------------
// --- Internal helper methods related to conversion of energy-deposition into ---
// --- photons -> photoelectrons -> electrical signal                          ---
// -------------------------------------------------------------------------------
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot) const
{
  const Float_t epsilon = 0.0001f;
  const Float_t p = FV0DigParam::Instance().lightYield * FV0DigParam::Instance().photoCathodeEfficiency;
  if ((fabs(1.0f - p) < epsilon) || nPhot == 0) {
    return nPhot;
  }
  const Int_t n = Int_t(nPhot < 100
                          ? gRandom->Binomial(nPhot, p)
                          : gRandom->Gaus((p * nPhot) + 0.5, TMath::Sqrt(p * (1. - p) * nPhot)));
  return n;
}

Float_t Digitizer::SimulateTimeCfd(Int_t channel) const
{
  Float_t timeCfd = -1024.0f;
  Int_t const binShift = TMath::Nint(FV0DigParam::Instance().timeShiftCfd / mBinSize);
  Float_t sigPrev = -mPmtChargeVsTime[channel][0];
  for (Int_t iTimeBin = 1; iTimeBin < mNBins; ++iTimeBin) {
    Float_t const sigCurrent = (iTimeBin >= binShift
                                  ? 5.0f * mPmtChargeVsTime[channel][iTimeBin - binShift] - mPmtChargeVsTime[channel][iTimeBin]
                                  : -mPmtChargeVsTime[channel][iTimeBin]);
    if (sigPrev < 0.0f && sigCurrent >= 0.0f) {
      timeCfd = Float_t(iTimeBin) * mBinSize;
      break;
    }
    sigPrev = sigCurrent;
  }
  return timeCfd;
}

// The Distance is positive for top half-sectors (when the hit position is above the cell center (has higher y))
float Digitizer::getDistFromCellCenter(UInt_t cellId, double hitx, double hity)
{
  Geometry* geo = Geometry::instance();

  // Parametrize the line (ax+by+c=0) that crosses the detector center and the cell's middle point
  Point3Dsimple* pCell = &geo->getCellCenter(cellId);
  float x0, y0, z0;
  geo->getGlobalPosition(x0, y0, z0);
  double a = -(y0 - pCell->y) / (x0 - pCell->x);
  double b = 1;
  double c = -(y0 - a * x0);
  // Return the distance from hit to this line
  return (a * hitx + b * hity + c) / TMath::Sqrt(a * a + b * b);
}

float Digitizer::getSignalFraction(float distanceFromXc, bool isFirstChannel)
{
  float fraction = sigmoidPmtRing5(distanceFromXc);
  if (distanceFromXc > 0) {
    return isFirstChannel ? fraction : (1. - fraction);
  } else {
    return isFirstChannel ? (1. - fraction) : fraction;
  }
}
