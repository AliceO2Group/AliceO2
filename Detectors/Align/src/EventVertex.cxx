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

/// @file   EventVertex.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Special fake "sensor" for event vertex.

#include "Align/EventVertex.h"
//#include "AliTrackPointArray.h" FIXME(milettri): needs AliTrackPointArray
//#include "AliESDtrack.h" FIXME(milettri): needs AliESDtrack
#include "Align/AlignmentPoint.h"
#include "Align/AlignableDetector.h"
#include "Framework/Logger.h"
#include <TMath.h>

using namespace TMath;

ClassImp(o2::align::EventVertex);

namespace o2
{
namespace align
{

//_________________________________________________________
EventVertex::EventVertex() : AlignableSensor("Vertex", 0, 1)
{
  // def c-tor
  setVarFrame(kLOC);
  setFreeDOFPattern(BIT(kDOFTX) | BIT(kDOFTY) | BIT(kDOFTZ));
  //
}

//____________________________________________
void EventVertex::prepareMatrixT2L()
{
  // T2L matrix for vertex needs to be adjusted for every track
  // in order to have X axis along the track direction.
  // This method assumes that the mAlp was already set accordingly
  // fX is fixed to 0
  //
  mMatT2L.Clear();
  mMatT2L.RotateZ(mAlp * RadToDeg());
  //  mMatT2L.MultiplyLeft(&getMatrixL2GIdeal().Inverse()); L2G=I !!!
  //
}

//____________________________________________
void EventVertex::applyCorrection(double* vtx) const
{
  // apply eventual correction to supplied vertex position
  vtx[kDOFTX] += getParVal(kDOFTX);
  vtx[kDOFTY] += getParVal(kDOFTY);
  vtx[kDOFTZ] += getParVal(kDOFTZ);
  //
}

//FIXME(milettri): needs AliTrackPointArray, AliESDtrack
////____________________________________________
//AlignmentPoint* EventVertex::TrackPoint2AlgPoint(int, const AliTrackPointArray*, const AliESDtrack*)
//{
//  // convert the pntId-th point to AlignmentPoint
//  static int cnt = 0;
//  LOG(ERROR) << "This method shound not have been called," << cnt++;
//  return 0;
//}

} // namespace align
} // namespace o2
