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
//#include <iostream>
#include "FITBase/Geometry.h"

using namespace o2::fit;

ClassImp(o2::fit::RecPoints);
ClassImp(o2::fit::Channel);

//void RecPoints::FillFromDigits(const std::vector<Digit>& digits)
void RecPoints::FillFromDigits(const Digit* digit)
{
  mTimeAmp.clear();
  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * o2::fit::Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * o2::fit::Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t meancfd = 454; //!!!!!!should be calibrate with sliding window;
  Float_t cfd[nMCPs] = {}, amp[nMCPs] = {};
  Float_t sideAtime = 0, sideCtime = 0;

  //for (const auto& d : digits) {
//  Int_t mcp = d.getChannel();
//  cfd[mcp] = d.getCFD();
//  amp[mcp] = d.getQTC();
//  mTimeAmp.push_back(Channel{ mcp, cfd[mcp], amp[mcp] });
//}

  for (const auto& d : digit->getChDgData()) {
    Int_t mcp = d.ChId;
    cfd[mcp] = d.CFDTime;
    amp[mcp] = d.QTCAmpl;
    mTimeAmp.push_back(Channel{ mcp, cfd[mcp], amp[mcp] });
  }

  for (Int_t imcp = 0; imcp < nMCPsA; imcp++) {
    if (cfd[imcp] > 0) {
      sideAtime += (cfd[imcp] - meancfd);
      ndigitsA++;
    }
  }
  for (Int_t imcp = 0; imcp < nMCPsC; imcp++) {
    if (cfd[imcp + nMCPsA] > 0) {
      sideCtime += (cfd[imcp + nMCPsA] - meancfd);
      ndigitsC++;
    }
  }

  if (ndigitsA > 0)
    sideAtime = sideAtime / Float_t(ndigitsA);
  if (ndigitsC > 0)
    sideCtime = sideCtime / Float_t(ndigitsC);

  mCollisionTime[0] = (sideAtime + sideCtime) / 2.;
  mCollisionTime[1] = sideAtime;
  mCollisionTime[2] = sideCtime;
  mVertex = (sideAtime - sideCtime) / 2.;
}
