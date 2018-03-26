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
#include "FairLogger.h"
#include "MIDBase/Constants.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
GeometryTransformer::GeometryTransformer()
{
  /// Default constructor
  init();
}

//______________________________________________________________________________
void GeometryTransformer::init()
{
  /// Initializes default geometry
  const double degToRad = TMath::DegToRad();

  ROOT::Math::Rotation3D planeRot(ROOT::Math::RotationX(Constants::sBeamAngle * degToRad));

  for (int iside = 0; iside < 2; ++iside) {
    bool isRight = (iside == 0);
    double angle = isRight ? 0. : 180.;
    ROOT::Math::Rotation3D rot(ROOT::Math::RotationY(angle * degToRad));
    double xSign = (iside == 0) ? 1. : -1.;
    for (int ichamber = 0; ichamber < 4; ++ichamber) {
      ROOT::Math::Translation3D planeTrans(0., 0., Constants::sDefaultChamberZ[ichamber]);
      for (int irpc = 0; irpc < 9; ++irpc) {
        double xPos = xSign * Constants::getRPCCenterPosX(ichamber, irpc);
        double sign = (irpc % 2 == 0) ? 1. : -1;
        if (iside == 1) {
          sign *= -1.;
        }
        double zPos = sign * Constants::sRPCZShift;
        double newZ = Constants::sDefaultChamberZ[0] + zPos;
        double oldZ = Constants::sDefaultChamberZ[0] - zPos;
        double yPos = Constants::getRPCHalfHeight(ichamber) * (irpc - 4) * (1. + newZ / oldZ);
        int deId = Constants::getDEId(isRight, ichamber, irpc);
        ROOT::Math::Translation3D trans(xPos, yPos, zPos);
        mTransformations[deId] = planeTrans * planeRot * trans * rot;
        LOG(DEBUG) << "DeID " << deId << "\n" << mTransformations[deId];
      }
    }
  }
}

} // namespace mid
} // namespace o2
