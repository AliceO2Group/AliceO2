// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITReconstruction/RecPoints.h"
#include "FITBase/Geometry.h"
#include <cassert>
#include <iostream>

using namespace o2::fit;

ClassImp(o2::fit::RecPoints);

void RecPoints::FillFromDigits(const Digit& digit)
{
  mTimeAmp.clear();
  mCollisionTime = {};

  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * o2::fit::Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * o2::fit::Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t cfd[nMCPs] = {}, amp[nMCPs] = {};
  Float_t sideAtime = 0, sideCtime = 0;

  Float_t eventTimeFromDigits = digit.getTime();
  constexpr Double_t BC_clk = 25.;                //ns event clk lenght
  constexpr Double_t BC_clk_center = BC_clk / 2.; // clk center
  Int_t nClk = floor(eventTimeFromDigits / BC_clk);
  Double_t BCEventTime = eventTimeFromDigits - BC_clk * nClk;
  mEventTime = BC_clk_center + BCEventTime;
  //   std::cout << " Event Time from digits" << eventTimeFromDigits <<" BCEventTime "<<BCEventTime<< " mEveentTime "<< mEventTime<<std::endl;
  for (const auto& d : digit.getChDgData()) {
    Int_t mcp = d.ChId;
    cfd[mcp] = d.CFDTime /*- BC_clk_center - BCEventTime*/;
    amp[mcp] = d.QTCAmpl;
    mTimeAmp.push_back(ChannelData{ mcp, cfd[mcp], amp[mcp] });
    //   LOG(DEBUG) << " mcp " << mcp<<" time "<< cfd[mcp]<<  FairLogger::endl;
  }

  for (Int_t imcp = 0; imcp < nMCPsA; imcp++) {
    if (cfd[imcp] > (BC_clk_center + BCEventTime - 2) &&
        cfd[imcp] < (BC_clk_center + BCEventTime + 2)) {
      sideAtime += (cfd[imcp]);
      ndigitsA++;
    }
  }
  for (Int_t imcp = 0; imcp < nMCPsC; imcp++) {
    if (cfd[imcp + nMCPsA] > 0) {
      sideCtime += (cfd[imcp + nMCPsA]);
      ndigitsC++;
    }
  }
  if (ndigitsA > 0)
    sideAtime = sideAtime / Float_t(ndigitsA);
  if (ndigitsC > 0)
    sideCtime = sideCtime / Float_t(ndigitsC);

  if (sideAtime > 0 && sideCtime > 0) {
    mVertex = (sideAtime - sideCtime) / 2.;
    mCollisionTime[0] = (sideAtime + sideCtime) / 2.;
  }
  if (sideAtime > 0) {
    mCollisionTime[1] = sideAtime;
  }
  if (sideCtime > 0) {
    mCollisionTime[2] = sideCtime;
  }
}
