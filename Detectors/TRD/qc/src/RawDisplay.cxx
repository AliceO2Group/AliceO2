// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDQC/RawDisplay.h"
#include "TRDQC/RawDataManager.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

#include <TVirtualPad.h>
#include <TPad.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TLine.h>
#include <TMarker.h>

using namespace o2::trd;

namespace o2::trd
{

/// Modified version of o2::trd::Tracklet64::getPadCol returning a float
float PadColF(o2::trd::Tracklet64& tracklet)
{
  // obtain pad number relative to MCM center
  float padLocal = tracklet.getPositionBinSigned() * constants::GRANULARITYTRKLPOS;
  // MCM number in column direction (0..7)
  int mcmCol = (tracklet.getMCM() % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (tracklet.getROB() % 2);

  // original calculation
  // FIXME: understand why the offset seems to be 6 pads and not nChannels / 2 = 10.5
  // return CAMath::Nint(6.f + mcmCol * ((float)constants::NCOLMCM) + padLocal);

  // my calculation
  return float((mcmCol + 1) * constants::NCOLMCM) + padLocal - 10.0;
}

}; // namespace o2::trd

RawDisplay::RawDisplay(RawDataSpan& dataspan, TVirtualPad* pad)
  : mDataSpan(dataspan), mPad(pad)
{
}

MCMDisplay::MCMDisplay(RawDataSpan& mcmdata, TVirtualPad* pad)
  : RawDisplay(mcmdata, pad) // initializes mDataSpan, mPad
{
  int det = -1, rob = -1, mcm = -1;

  if (std::distance(mDataSpan.digits.begin(), mDataSpan.digits.end())) {
    auto x = *mDataSpan.digits.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else if (std::distance(mDataSpan.tracklets.begin(), mDataSpan.tracklets.end())) {
    auto x = *mDataSpan.tracklets.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else {
    O2ERROR("found neither digits nor tracklets in MCM");
    assert(false);
  }

  mName = Form("det%03d_rob%d_mcm%02d", det, rob, mcm);
  mDesc = Form("Detector %02d_%d_%d (%03d) - MCM %d:%02d", det / 30, (det % 30) / 6, det % 6, det, rob, mcm);
  ;

  // MCM column number on ROC [0..7]
  int mcmcol = mcm % constants::NMCMROBINCOL + HelperMethods::getROBSide(rob) * constants::NMCMROBINCOL;

  mFirstPad = mcmcol * constants::NCOLMCM - 1;
  mLastPad = (mcmcol + 1) * constants::NCOLMCM + 2;

  if (pad == nullptr) {
    mPad = new TCanvas(mName.c_str(), mDesc.c_str(), 800, 600);
  } else {
    mPad = pad;
    mPad->SetName(mName.c_str());
    mPad->SetTitle(mDesc.c_str());
  }

  mDigitsHisto = new TH2F(mName.c_str(), (mDesc + ";pad;time bin").c_str(), (mLastPad - mFirstPad), mFirstPad, mLastPad, 30, 0., 30.);

  for (auto digit : mDataSpan.digits) {
    auto adc = digit.getADC();
    for (int tb = 0; tb < 30; ++tb) {
      mDigitsHisto->Fill(digit.getPadCol(), tb, adc[tb]);
    }
  }
  mDigitsHisto->SetStats(0);
}

void RawDisplay::drawDigits(std::string opt)
{
  mPad->cd();
  mDigitsHisto->Draw(opt.c_str());
}

void RawDisplay::drawTracklets()
{
  mPad->cd();

  TLine trkl;
  trkl.SetLineColor(kRed);
  trkl.SetLineWidth(3);

  for (auto tracklet : mDataSpan.tracklets) {
    auto pos = PadColF(tracklet);
    auto slope = -tracklet.getSlopeBinSigned() * constants::GRANULARITYTRKLSLOPE / constants::ADDBITSHIFTSLOPE;
    trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
  }
}

void RawDisplay::drawClusters()
{
  mPad->cd();

  drawDigits();

  TMarker clustermarker;
  clustermarker.SetMarkerColor(kRed);
  clustermarker.SetMarkerStyle(2);
  clustermarker.SetMarkerSize(1.5);

  TMarker cogmarker;
  cogmarker.SetMarkerColor(kGreen);
  cogmarker.SetMarkerStyle(3);
  cogmarker.SetMarkerSize(1.5);
  for (int t = 1; t <= mDigitsHisto->GetNbinsY(); ++t) {
    for (int p = 2; p <= mDigitsHisto->GetNbinsX() - 1; ++p) {
      // cout << p << "/" << t << " -> " << mDigitsHisto->GetBinContent(i,j) << endl;
      double baseline = 9.5;
      double left = mDigitsHisto->GetBinContent(p - 1, t) - baseline;
      double centre = mDigitsHisto->GetBinContent(p, t) - baseline;
      double right = mDigitsHisto->GetBinContent(p + 1, t) - baseline;
      if (centre > left && centre > right && (centre + left + right) > mClusterThreshold) {
        double pos = 0.5 * log(right / left) / log(centre * centre / left / right);
        double clpos = mDigitsHisto->GetXaxis()->GetBinCenter(p) + pos;
        double cog = (right - left) / (right + centre + left) + mDigitsHisto->GetXaxis()->GetBinCenter(p);
        // cout << "t=" << t << " p=" << p
        //      << ":   ADCs = " << left << " / " << centre << " / " << right
        //      << "   pos = " << pos << " ~ " << clpos
        //      << endl;
        clustermarker.DrawMarker(clpos, t - 0.5);
        // cogmarker.DrawMarker(cog, t - 0.5);
      }
    }
  }
}

void RawDisplay::drawHits()
{
  TMarker hitmarker;
  hitmarker.SetMarkerColor(kBlue);
  hitmarker.SetMarkerStyle(38);
  for (auto hit : mDataSpan.hits) {
    if (hit.getCharge() > 0.0) {
      hitmarker.SetMarkerSize(log10(hit.getCharge()));
      hitmarker.DrawMarker(hit.getPadCol(), hit.getTimeBin());
    }
  }
}

void RawDisplay::drawMCTrackSegments()
{
  TLine line;
  line.SetLineColor(kBlue);
  line.SetLineWidth(2.0);

  for (auto& trkl : mDataSpan.makeMCTrackSegments()) {
    line.DrawLine(trkl.getStartPoint().getPadCol(), trkl.getStartPoint().getTimeBin(), trkl.getEndPoint().getPadCol(), trkl.getEndPoint().getTimeBin());
  }
}
