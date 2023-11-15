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

/// @file   AlignableSensorTPC.h
/// @author ruben.shahoyan@cern.ch
/// @brief  TPC sensor

#include "Align/AlignableSensorTPC.h"
#include "Align/AlignableDetectorTPC.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"

namespace o2
{
namespace align
{
using namespace o2::align::utils;
using namespace TMath;

//_________________________________________________________
AlignableSensorTPC::AlignableSensorTPC(const char* name, int vid, int iid, int isec, Controller* ctr) : AlignableSensor(name, vid, iid, ctr), mSector(isec)
{
  // def c-tor
}

//____________________________________________
void AlignableSensorTPC::prepareMatrixL2G(bool reco)
{
  double alp = math_utils::detail::sector2Angle<float>(mSector % 18);
  TGeoHMatrix m;
  m.RotateZ(alp * RadToDeg());
  reco ? setMatrixL2GReco(m) : setMatrixL2G(m);
}

//____________________________________________
void AlignableSensorTPC::prepareMatrixL2GIdeal()
{
  double alp = math_utils::detail::sector2Angle<float>(mSector % 18);
  TGeoHMatrix m;
  m.RotateZ(alp * RadToDeg());
  setMatrixL2GIdeal(m);
}

//____________________________________________
void AlignableSensorTPC::prepareMatrixT2L()
{
  // local and tracking matrices are the same
  double alp = math_utils::detail::sector2Angle<float>(mSector % 18);
  mAlp = alp;
  TGeoHMatrix m;
  setMatrixT2L(m);
  //
}

} // namespace align
} // namespace o2
