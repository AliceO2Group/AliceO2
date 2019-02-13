// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/GeometryTransformer.cxx
/// \brief  Geometry transformer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 July 2017

#include "MIDBase/GeometryTransformer.h"

#include <fstream>
#include <string>
#include "TMath.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"

namespace o2
{
namespace mid
{

void GeometryTransformer::setMatrix(int deId, const ROOT::Math::Transform3D& matrix)
{
  /// Sets the transformation matrix for detection element deId
  Constants::assertDEId(deId);
  mTransformations[deId] = matrix;
}

ROOT::Math::Transform3D getDefaultChamberTransform(int ichamber)
{
  /// Returns the default chamber transformation
  const double degToRad = TMath::DegToRad();
  ROOT::Math::Rotation3D planeRot(ROOT::Math::RotationX(Constants::sBeamAngle * degToRad));
  ROOT::Math::Translation3D planeTrans(0., 0., Constants::sDefaultChamberZ[ichamber]);
  return planeTrans * planeRot;
}

ROOT::Math::Transform3D getDefaultRPCTransform(bool isRight, int chamber, int rpc)
{
  /// Returns the default RPC transformation in the chamber plane
  const double degToRad = TMath::DegToRad();
  double angle = isRight ? 0. : 180.;
  ROOT::Math::Rotation3D rot(ROOT::Math::RotationY(angle * degToRad));
  double xSign = isRight ? 1. : -1.;
  double xPos = xSign * Constants::getRPCCenterPosX(chamber, rpc);
  double sign = (rpc % 2 == 0) ? 1. : -1;
  if (!isRight) {
    sign *= -1.;
  }
  double zPos = sign * Constants::sRPCZShift;
  double newZ = Constants::sDefaultChamberZ[0] + zPos;
  double oldZ = Constants::sDefaultChamberZ[0] - zPos;
  double yPos = Constants::getRPCHalfHeight(chamber) * (rpc - 4) * (1. + newZ / oldZ);
  ROOT::Math::Translation3D trans(xPos, yPos, zPos);
  return trans * rot;
}

GeometryTransformer createDefaultTransformer()
{
  /// Creates the default transformer
  GeometryTransformer geoTrans;
  for (int ich = 0; ich < Constants::sNChambers; ++ich) {
    for (int iside = 0; iside < 2; ++iside) {
      bool isRight = (iside == 0);
      for (int irpc = 0; irpc < Constants::sNRPCLines; ++irpc) {
        int deId = Constants::getDEId(isRight, ich, irpc);
        ROOT::Math::Transform3D matrix = getDefaultChamberTransform(ich) * getDefaultRPCTransform(isRight, ich, irpc);
        geoTrans.setMatrix(deId, matrix);
      }
    }
  }
  return geoTrans;
}

} // namespace mid
} // namespace o2
