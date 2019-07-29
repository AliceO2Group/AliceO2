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
/// \file    MinimalisticTrack.cxx
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#include "EventVisualisationDataConverter/MinimalisticTrack.h"
#include <iostream>

using namespace std;

namespace o2  {
namespace event_visualisation {


MinimalisticTrack::MinimalisticTrack() = default;

MinimalisticTrack::MinimalisticTrack(
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

void MinimalisticTrack::addChild(int childID)
{
  mChildrenIDs.push_back(childID);
}

void MinimalisticTrack::addMomentum(double pxpypz[3])
{
  for (int i = 0; i < 3; i++)
    mMomentum[i] = pxpypz[i];
}

void MinimalisticTrack::addStartCoordinates(double xyz[3])
{
  for (int i = 0; i < 3; i++)
    mStartCoordinates[i] = xyz[i];
}

void MinimalisticTrack::addEndCoordinates(double xyz[3])
{
  for (int i = 0; i < 3; i++)
    mEndCoordinates[i] = xyz[i];
}

void MinimalisticTrack::addPolyPoint(double x, double y, double z)
{
  mPolyX.push_back(x);
  mPolyY.push_back(y);
  mPolyZ.push_back(z);
}

void MinimalisticTrack::addPolyPoint(double *xyz)
{
  mPolyX.push_back(xyz[0]);
  mPolyY.push_back(xyz[1]);
  mPolyZ.push_back(xyz[2]);
}

void MinimalisticTrack::setTrackType(ETrackType type)
{
  mType = gTrackTypes[type];
}

void MinimalisticTrack::fillWithRandomData()
{
  mCharge = ((double)rand()/RAND_MAX>0.5) ? 1 : -1;
  
  mStartCoordinates[0] = 0.0;
  mStartCoordinates[1] = 0.0;
  mStartCoordinates[2] = 0.0;
  
  mMomentum[0] = 2*((double)rand()/RAND_MAX)-1;
  mMomentum[1] = 2*((double)rand()/RAND_MAX)-1;
  mMomentum[2] = 2*((double)rand()/RAND_MAX)-1;
  
  mMass = 1000*(double)rand()/RAND_MAX + 0.5;
  mEnergy = mMass + 1000*(double)rand()/RAND_MAX;
  
  int PID[10] = {-2212, -321, -211, -13, -11, 11, 13 , 211, 321, 2212 };
  mPID = PID[(int)(rand()%10)];
}
    
}
}
