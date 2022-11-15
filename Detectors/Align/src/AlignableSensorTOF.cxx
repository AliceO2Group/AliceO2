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

/// @file   AlignableSensorTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TOF sensor

#include "Align/AlignableSensorTOF.h"
#include "Align/utils.h"
#include "Align/AlignableDetectorTOF.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"
//#include "AliTrackPointArray.h"
//#include "AliESDtrack.h"

ClassImp(o2::align::AlignableSensorTOF);

using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AlignableSensorTOF::AlignableSensorTOF(const char* name, int vid, int iid, int isec, Controller* ctr) : AlignableSensor(name, vid, iid, ctr), mSector(isec)
{
  // def c-tor
}

//____________________________________________
void AlignableSensorTOF::prepareMatrixT2L()
{
  // extract from geometry T2L matrix
  double alp = math_utils::detail::sector2Angle<float>(mSector);
  mAlp = alp;
  TGeoHMatrix t2l;
  double loc[3] = {0, 0, 0}, glo[3];
  getMatrixL2GIdeal().LocalToMaster(loc, glo);
  mX = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  t2l.RotateZ(alp * RadToDeg());
  const TGeoHMatrix l2gi = getMatrixL2GIdeal().Inverse();
  t2l.MultiplyLeft(&l2gi);
  setMatrixT2L(t2l);
  //
}

} // namespace align
} // namespace o2
