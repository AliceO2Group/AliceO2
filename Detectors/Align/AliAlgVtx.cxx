/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include "AliAlgVtx.h"
#include "AliTrackPointArray.h"
#include "AliESDtrack.h"
#include "AliAlgPoint.h"
#include "AliAlgDet.h"
#include "AliLog.h"
#include <TMath.h>

using namespace TMath;

ClassImp(AliAlgVtx)

//_________________________________________________________
AliAlgVtx::AliAlgVtx() : AliAlgSens("Vertex",0,1)
{
  // def c-tor
  SetVarFrame(kLOC);
  SetFreeDOFPattern( BIT(kDOFTX) | BIT(kDOFTY) | BIT(kDOFTZ) );
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
  fMatT2L.RotateZ(fAlp*RadToDeg());
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

//____________________________________________
AliAlgPoint* AliAlgVtx::TrackPoint2AlgPoint(int, const AliTrackPointArray*, const AliESDtrack*)
{
  // convert the pntId-th point to AliAlgPoint
  static int cnt=0;
  AliErrorF("This method shound not have been called, %d",cnt++);
  return 0;
}
