// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SVertexHypothesis.cxx
/// \brief V0 hypothesis checker
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/SVertexHypothesis.h"

using namespace o2::vertexing;

void SVertexHypothesis::set(PID v0, PID ppos, PID pneg, float sig, float nSig, float margin, float cpt, float bz)
{
  mPIDV0 = v0;
  mPIDPosProng = ppos;
  mPIDNegProng = pneg;
  mPars[SigmaM] = sig;
  mPars[NSigmaM] = nSig;
  mPars[MarginM] = margin;
  mPars[CPt] = std::abs(bz) > 1e-3 ? cpt * 5.0066791 / bz : 0.; // assume that pT dependent sigma is linear with B
}

void SVertexHypothesis::set(PID v0, PID ppos, PID pneg, const float pars[NPIDParams], float bz)
{
  set(v0, ppos, pneg, pars[SigmaM], pars[NSigmaM], pars[MarginM], pars[CPt], bz);
}
