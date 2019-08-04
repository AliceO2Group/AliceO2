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
/// \file    MinimalisticTrack.h
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_MINIMALISTICTRACK_H
#define ALICE_O2_EVENTVISUALISATION_BASE_MINIMALISTICTRACK_H

#include "ConversionConstants.h"

#include <iosfwd>
#include <string>
#include <vector>
#include <cmath>

namespace o2
{
namespace event_visualisation
{

/// Minimalistic description of particles track
///
/// This class is used mainly for visualisation purpose.
/// It keeps basic information about a track, such as its vertex,
/// momentum, PID, phi and theta or helix curvature.

class MinimalisticTrack
{
 public:
  // Default constructor
  MinimalisticTrack();

  // Constructor with properties initialisation
  MinimalisticTrack(
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
    int type);

  // Add child particle (coming from decay of this particle)
  void addChild(int childID);
  // Add xyz coordinates of the point along the track
  void addPolyPoint(double x, double y, double z);
  // Add xyz coordinates of the point along the track
  void addPolyPoint(double xyz[3]);
  // Track type setter (standard track, V0, kink, cascade)
  void setTrackType(ETrackType type);

  // Vertex getter
  double* getVertex() { return mStartCoordinates; }
  // Momentum vector getter
  double* getMomentum() { return mMomentum; }
  // Beta (velocity) getter
  double getBeta() { return sqrt(1 - std::pow(mMass / mEnergy, 2)); }
  // Charge getter
  int getCharge() { return mCharge; }
  // PID (particle identification code) getter
  int getPID() { return mPID; }

  // Generates random track
  void fillWithRandomData();

 private:
  // Set coordinates of the beginning of the track
  void addStartCoordinates(double xyz[3]);
  // Set coordinates of the end of the track
  void addEndCoordinates(double xyz[3]);
  /// Set momentum vector
  void addMomentum(double pxpypz[3]);

  int mID;                     /// Unique identifier of the track
  std::string mType;           /// Type (standard, V0 mother, daughter etc.)
  int mCharge;                 /// Charge of the particle
  double mEnergy;              /// Energy of the particle
  int mParentID;               /// ID of the parent-track (-1 means no parent)
  int mPID;                    /// PDG code of the particle
  double mSignedPT;            /// Signed transverse momentum
  double mMass;                /// Mass of the particle
  double mMomentum[3];         /// Momentum vector
  double mStartCoordinates[3]; /// Vector of track's start coordinates
  double mEndCoordinates[3];   /// Vector of track's end coordinates
  double mHelixCurvature;      /// Helix curvature of the trajectory
  double mTheta;               /// An angle from Z-axis to the radius vector pointing to the particle
  double mPhi;                 /// An angle from X-axis to the radius vector pointing to the particle

  std::vector<int> mChildrenIDs; /// Uniqe IDs of children particles

  /// Polylines -- array of points along the trajectory of the track
  std::vector<double> mPolyX;
  std::vector<double> mPolyY;
  std::vector<double> mPolyZ;
};

#endif
}
}
