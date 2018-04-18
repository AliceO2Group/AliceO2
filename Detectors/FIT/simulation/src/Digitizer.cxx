// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITSimulation/Digitizer.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::fit;
using o2::fit::Geometry;

ClassImp(Digitizer);

void Digitizer::process(const std::vector<HitType>* hits, std::vector<Digit>* digits)
{
  // hits array of FIT hits for a given simulated event
  mDigits = digits;
  Double_t timeframe = 0;
  Int_t bc = 0;
  constexpr Int_t nMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4;

  Int_t amp[nMCPs] = {};
  Double_t cfd[nMCPs] = {};

  for (auto& hit : *hits) {
    // TODO: put timeframe counting/selection
    // if (timeframe == mTimeFrameCurrent) {
    // timeframe = Int_t((mEventTime + hit.GetTime())); // to be replaced with uncalibrated time
    Int_t mcp = hit.GetDetectorID();
    Double_t hittime = hit.GetTime();
    if (mcp > 4 * Geometry::NCellsA)
      hittime += mTimeDiffAC;
    if (hittime > mLowTime && hittime < mHighTime) {
      cfd[mcp] += hittime;
      amp[mcp]++;
    }
  } // end of loop over hits

  for (Int_t ipmt = 0; ipmt < nMCPs; ipmt++) {
    if (amp[ipmt] > mAmpThreshold) {
      cfd[ipmt] = cfd[ipmt] / Float_t(amp[ipmt]); //mean time on 1 quadrant
      cfd[ipmt] = (gRandom->Gaus(cfd[ipmt], 50)) / Geometry::ChannelWidth;
      mDigits->emplace_back(timeframe, ipmt, cfd[ipmt], Float_t(amp[ipmt]), bc);
    }
  } // end of loop over PMT
}

void Digitizer::initParameters()
{
  mAmpThreshold = 100;
  mLowTime = 10000;
  mHighTime = 12500;
  mEventTime = 0;
  // murmur
}
//_______________________________________________________________________
void Digitizer::init() {}

//_______________________________________________________________________
void Digitizer::finish() {}
/*
void Digitizer::printParameters()
{
  //murmur
}
*/
