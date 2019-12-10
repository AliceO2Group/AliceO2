// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    VisualisationTrack.cxx
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include <iostream>

using namespace std;

namespace o2
{
namespace event_visualisation
{

VisualisationTrack::VisualisationTrack() = default;

VisualisationTrack::VisualisationTrack(
  int charge,
  double energy,
  int ID,
  int PID,
  double mass,
  double signedPT,
  double startXYZ[],
  double endXYZ[],
  double pxpypz[],
  int parentID,
  double phi,
  double theta,
  double helixCurvature,
  int type)
  : mCharge(charge),
    mEnergy(energy),
    mParentID(parentID),
    mPID(PID),
    mSignedPT(signedPT),
    mMass(mass),
    mHelixCurvature(helixCurvature),
    mTheta(theta),
    mPhi(phi)
{
  addMomentum(pxpypz);
  addStartCoordinates(startXYZ);
  addEndCoordinates(endXYZ);
  mID = ID;
  mType = gTrackTypes[type];
}

void VisualisationTrack::addChild(int childID)
{
  mChildrenIDs.push_back(childID);
}

void VisualisationTrack::addMomentum(double pxpypz[3])
{
  for (int i = 0; i < 3; i++)
    mMomentum[i] = pxpypz[i];
}

void VisualisationTrack::addStartCoordinates(double xyz[3])
{
  for (int i = 0; i < 3; i++)
    mStartCoordinates[i] = xyz[i];
}

void VisualisationTrack::addEndCoordinates(double xyz[3])
{
  for (int i = 0; i < 3; i++)
    mEndCoordinates[i] = xyz[i];
}

void VisualisationTrack::addPolyPoint(double x, double y, double z)
{
  mPolyX.push_back(x);
  mPolyY.push_back(y);
  mPolyZ.push_back(z);
}

void VisualisationTrack::addPolyPoint(double* xyz)
{
  mPolyX.push_back(xyz[0]);
  mPolyY.push_back(xyz[1]);
  mPolyZ.push_back(xyz[2]);
}

void VisualisationTrack::setTrackType(ETrackType type)
{
  mType = gTrackTypes[type];
}

} // namespace event_visualisation
} // namespace o2
