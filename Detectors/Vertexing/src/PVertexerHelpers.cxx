// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PVertexerHelpers.cxx
/// \brief Primary vertex finder helper classes
/// \author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/PVertexerHelpers.h"

using namespace o2::vertexing;

void VertexSeed::print() const
{
  auto terr2 = tMeanAccErr > 0 ? 1. / tMeanAccErr : 0.;
  LOGF(INFO, "VtxSeed: Scale: %+e ScalePrev: %+e |NScaleIncreased: %d NSlowDecrease: %d| WChi2: %e WSum: %e | TMean: %e TMeanE: %e\n",
       scaleSigma2, scaleSigma2Prev, nScaleIncrease, nScaleSlowConvergence, wghChi2, wghSum, tMeanAcc * terr2, std::sqrt(terr2));
  double dZP, rmsZP, dZN, rmsZN, dTP, rmsTP, dTN, rmsTN;
  double dZ, rmsZ, dT, rmsT;
  Vertex::print();
}
