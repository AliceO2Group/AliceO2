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

#include "Generators/FlowMapper.h"
#include "TH1D.h"
#include "TH3D.h"
#include "TF1.h"
#include <fairlogger/Logger.h>

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

FlowMapper::FlowMapper()
{
  // base constructor. Creates cumulative function only so that's already in place but not much else.
  mCumulative = std::make_unique<TF1>("mCumulative", "x+[0]*TMath::Sin(2*x)", 0, 2 * TMath::Pi());
  mBinsPhi = 4000; // a first guess
}

void FlowMapper::Setv2VsPt(TH1D hv2VsPtProvided)
{
  // Sets the v2 vs pT to be used.
  mhv2vsPt = std::make_unique<TH1D>(hv2VsPtProvided);
}
void FlowMapper::SetEccVsB(TH1D hEccVsBProvided)
{
  // Sets the v2 vs pT to be used.
  mhEccVsB = std::make_unique<TH1D>(hEccVsBProvided);
}
void FlowMapper::CreateLUT()
{
  if (!mhv2vsPt) {
    LOG(fatal) << "You did not specify a v2 vs pT histogram!";
    return;
  }
  if (!mhEccVsB) {
    LOG(fatal) << "You did not specify an ecc vs B histogram!";
    return;
  }
  LOG(info) << "Proceeding to creating a look-up table...";
  const Long_t nbinsB = mhEccVsB->GetNbinsX();
  const Long_t nbinsPt = mhv2vsPt->GetNbinsX();
  const Long_t nbinsPhi = mBinsPhi; // constant in this context necessary
  std::vector<double> binsB(nbinsB + 1, 0);
  std::vector<double> binsPt(nbinsPt + 1, 0);
  std::vector<double> binsPhi(nbinsPhi + 1, 0);

  for (int ii = 0; ii < nbinsB + 1; ii++) {
    binsB[ii] = mhEccVsB->GetBinLowEdge(ii + 1);
  }
  for (int ii = 0; ii < nbinsPt + 1; ii++) {
    binsPt[ii] = mhv2vsPt->GetBinLowEdge(ii + 1);
  }
  for (int ii = 0; ii < nbinsPhi + 1; ii++) {
    binsPhi[ii] = static_cast<Double_t>(ii) * 2 * TMath::Pi() / static_cast<Double_t>(nbinsPhi);
  }

  // std::make_unique<TH1F>("hSign", "Sign of electric charge;charge sign", 3, -1.5, 1.5);

  mhLUT = std::make_unique<TH3D>("mhLUT", "", nbinsB, binsB.data(), nbinsPt, binsPt.data(), nbinsPhi, binsPhi.data());

  // loop over each centrality (b) bin
  for (int ic = 0; ic < nbinsB; ic++) {
    // loop over each pt bin
    for (int ip = 0; ip < nbinsPt; ip++) {
      // find target v2 value and set cumulative for inversion
      double v2target = mhv2vsPt->GetBinContent(ip + 1) * mhEccVsB->GetBinContent(ic + 1);
      LOG(info) << "At b ~ " << mhEccVsB->GetBinCenter(ic + 1) << ", pt ~ " << mhv2vsPt->GetBinCenter(ip + 1) << ", ref v2 is " << mhv2vsPt->GetBinContent(ip + 1) << ", scale is " << mhEccVsB->GetBinContent(ic + 1) << ", target v2 is " << v2target << ", inverting...";
      mCumulative->SetParameter(0, v2target); // set up
      for (Int_t ia = 0; ia < nbinsPhi; ia++) {
        // Look systematically for the X value that gives this Y
        // There are probably better ways of doing this, but OK
        Double_t lY = mhLUT->GetZaxis()->GetBinCenter(ia + 1);
        Double_t lX = lY; // a first reasonable guess
        Bool_t lConverged = kFALSE;
        while (!lConverged) {
          Double_t lDistance = mCumulative->Eval(lX) - lY;
          if (TMath::Abs(lDistance) < mPrecision) {
            lConverged = kTRUE;
            break;
          }
          Double_t lDerivativeValue = mDerivative / (mCumulative->Eval(lX + mDerivative) - mCumulative->Eval(lX));
          lX = lX - lDistance * lDerivativeValue * 0.25; // 0.5: speed factor, don't overshoot but control reasonable
        }
        mhLUT->SetBinContent(ic + 1, ip + 1, ia + 1, lX);
      }
    }
  }
}

Double_t FlowMapper::MapPhi(Double_t lPhiInput, Double_t b, Double_t pt)
{
  Int_t lLowestPeriod = TMath::Floor(lPhiInput / (2 * TMath::Pi()));
  Double_t lPhiOld = lPhiInput - 2 * lLowestPeriod * TMath::Pi();
  Double_t lPhiNew = lPhiOld;

  // Avoid interpolation problems in dimension: pT
  Double_t lMaxPt = mhLUT->GetYaxis()->GetBinCenter(mhLUT->GetYaxis()->GetNbins());
  Double_t lMinPt = mhLUT->GetYaxis()->GetBinCenter(1);
  if (pt > lMaxPt) {
    pt = lMaxPt; // avoid interpolation problems at edge
  }

  Double_t phiWidth = mhLUT->GetZaxis()->GetBinWidth(1); // any bin, assume constant

  // Valid if not at edges. If at edges, that's ok, do not map
  bool validPhi = lPhiNew > phiWidth / 2.0f && lPhiNew < 2.0 * TMath::Pi() - phiWidth / 2.0f;

  // If at very high b, do not map
  bool validB = b < mhLUT->GetXaxis()->GetBinCenter(mhLUT->GetXaxis()->GetNbins());
  Double_t minB = mhLUT->GetXaxis()->GetBinCenter(1);

  if (validPhi && validB) {

    Double_t scaleFactor = 1.0; // no need if not special conditions
    if (pt < lMinPt) {
      scaleFactor *= pt / lMinPt; // downscale the difference, zero at zero pT
      pt = lMinPt;
    }
    if (b < minB) {
      scaleFactor *= b / minB; // downscale the difference, zero at zero b
      b = minB;
    }
    lPhiNew = mhLUT->Interpolate(b, pt, lPhiOld);

    lPhiNew = scaleFactor * lPhiNew + (1.0 - scaleFactor) * lPhiOld;
  }
  return lPhiNew + 2.0 * lLowestPeriod * TMath::Pi();
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::FlowMapper);