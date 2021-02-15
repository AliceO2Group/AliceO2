// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgVtx.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Special fake "sensor" for event vertex.

#include "Align/AliAlgVtx.h"
//#include "AliTrackPointArray.h" FIXME(milettri): needs AliTrackPointArray
//#include "AliESDtrack.h" FIXME(milettri): needs AliESDtrack
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgDet.h"
#include "Framework/Logger.h"
#include <TMath.h>

using namespace TMath;

ClassImp(o2::align::AliAlgVtx);

namespace o2
{
namespace align
{

//_________________________________________________________
AliAlgVtx::AliAlgVtx() : AliAlgSens("Vertex", 0, 1)
{
  // def c-tor
  SetVarFrame(kLOC);
  SetFreeDOFPattern(BIT(kDOFTX) | BIT(kDOFTY) | BIT(kDOFTZ));
  //
}

//____________________________________________
void AliAlgVtx::PrepareMatrixT2L()
{
  // T2L matrix for vertex needs to be adjusted for every track
  // in order to have X axis along the track direction.
  // This method assumes that the fAlp was already set accordingly
  // fX is fixed to 0
  //
  fMatT2L.Clear();
  fMatT2L.RotateZ(fAlp * RadToDeg());
  //  fMatT2L.MultiplyLeft(&GetMatrixL2GIdeal().Inverse()); L2G=I !!!
  //
}

//____________________________________________
void AliAlgVtx::ApplyCorrection(double* vtx) const
{
  // apply eventual correction to supplied vertex position
  vtx[kDOFTX] += GetParVal(kDOFTX);
  vtx[kDOFTY] += GetParVal(kDOFTY);
  vtx[kDOFTZ] += GetParVal(kDOFTZ);
  //
}

//FIXME(milettri): needs AliTrackPointArray, AliESDtrack
////____________________________________________
//AliAlgPoint* AliAlgVtx::TrackPoint2AlgPoint(int, const AliTrackPointArray*, const AliESDtrack*)
//{
//  // convert the pntId-th point to AliAlgPoint
//  static int cnt = 0;
//  LOG(ERROR) << "This method shound not have been called," << cnt++;
//  return 0;
//}

} // namespace align
} // namespace o2
