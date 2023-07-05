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

/// @file   AlignableSensorITS.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS sensor
#include "ITSBase/GeometryTGeo.h"
#include "Align/AlignableSensorITS.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableDetector.h"

ClassImp(o2::align::AlignableSensorITS);

using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AlignableSensorITS::AlignableSensorITS(const char* name, int vid, int iid, Controller* ctr)
  : AlignableSensor(name, vid, iid, ctr)
{
  // def c-tor
}

//____________________________________________
void AlignableSensorITS::prepareMatrixT2L()
{
  // extract geometry T2L matrix
  TGeoHMatrix t2l;
  const auto& l2g = getMatrixL2GIdeal();
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  l2g.LocalToMaster(locA, gloA);
  l2g.LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  double xp = gloB[0] - dx * t, yp = gloB[1] - dy * t;
  mX = std::sqrt(xp * xp + yp * yp);
  float alp = std::atan2(yp, xp);
  o2::math_utils::bringTo02Pi(alp);
  mAlp = alp;
  /* // this would proved x, alpha accounting for the corrections, we need ideal ones ?
     float x, alp;
     auto geom = o2::its::GeometryTGeo::Instance();
     geom->getSensorXAlphaRefPlane(getVolID(), x, alp);
     mAlp = alp;
     mX = x;
  */
  t2l.RotateZ(mAlp * RadToDeg()); // rotate in direction of normal to the sensor plane
  const TGeoHMatrix l2gi = l2g.Inverse();
  t2l.MultiplyLeft(&l2gi);
  setMatrixT2L(t2l);
}

} // namespace align
} // namespace o2
