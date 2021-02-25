// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_TRACKLETTRANSFORMER_H
#define O2_TRD_TRACKLETTRANSFORMER_H

#include "TRDBase/Geometry.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"

namespace o2
{
namespace trd
{

class TrackletTransformer
{
 public:
  TrackletTransformer();
  ~TrackletTransformer() = default;

  float getXCathode() { return mXCathode; }
  float getXAnode() { return mXAnode; }
  float getXDrift() { return mXDrift; }
  float getXtb0() { return mXtb0; }

  void setXCathode(float x) { mXCathode = x; }
  void setXAnode(float x) { mXAnode = x; }
  void setXDrift(float x) { mXDrift = x; }
  void setXtb0(float x) { mXtb0 = x; }

  void loadPadPlane(int hcid);

  float calculateY(int hcid, int column, int position);

  float calculateZ(int padrow);

  float calculateDy(int slope, double oldLorentzAngle, double lorentzAngle, double driftVRatio);

  float calibrateX(double x, double t0Correction);

  std::array<float, 3> transformL2T(int hcid, std::array<double, 3> spacePoint);

  CalibratedTracklet transformTracklet(Tracklet64 tracklet);

  double getTimebin(double x);

 private:
  o2::trd::Geometry* mGeo;
  const o2::trd::PadPlane* mPadPlane;

  float mXCathode;
  float mXAnode;
  float mXDrift;
  float mXtb0;

  float mt0Correction;
  float mOldLorentzAngle;
  float mLorentzAngle;
  float mDriftVRatio;
};

} // namespace trd
} // namespace o2

#endif
