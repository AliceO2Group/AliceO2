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

#include "TRDBase/CommonParam.h"
#include "TRDBase/DiffAndTimeStructEstimator.h"
#include "TRDBase/Geometry.h"
#include <cmath>

namespace o2::trd
{

using namespace garfield;
using namespace constants;

//_____________________________________________________________________________
bool DiffusionAndTimeStructEstimator::sampleTimeStruct(float vdrift)
{
  //
  // Samples the timing structure of a drift cell
  // Drift Time data calculated with Garfield (by C.Lippmann)
  //

  bool retVal = true;

  // Nothing to do
  if (std::abs(mTimeLastVdrift - vdrift) < 1.e-3) {
    return retVal;
  }
  mTimeLastVdrift = vdrift;

  // Drift time maps are saved for some drift velocity values (in drift region):
  constexpr float fVDsmp[8] = {1.032, 1.158, 1.299, 1.450, 1.610, 1.783, 1.959, 2.134};

  if (vdrift < fVDsmp[0]) {
    LOG(debug) << "TRD: Drift Velocity too small " << vdrift << " < " << fVDsmp[0];
    vdrift = fVDsmp[0];
    retVal = false;
  } else if (vdrift > fVDsmp[7]) {
    LOG(debug) << "TRD: Drift Velocity too large " << vdrift << " > " << fVDsmp[7];
    vdrift = fVDsmp[7];
    retVal = false;
  }
  if (vdrift > fVDsmp[6]) {
    mVDlo = fVDsmp[6];
    mVDhi = fVDsmp[7];
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time2100[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time2200[ctrt][ctrz];
      }
    }
  } else if (vdrift > fVDsmp[5]) {
    mVDlo = fVDsmp[5];
    mVDhi = fVDsmp[6];
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time2000[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time2100[ctrt][ctrz];
      }
    }
  } else if (vdrift > fVDsmp[4]) {
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time1900[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time2000[ctrt][ctrz];
      }
    }
    mVDlo = fVDsmp[4];
    mVDhi = fVDsmp[5];
  } else if (vdrift > fVDsmp[3]) {
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time1800[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time1900[ctrt][ctrz];
      }
    }
    mVDlo = fVDsmp[3];
    mVDhi = fVDsmp[4];
  } else if (vdrift > fVDsmp[2]) {
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time1700[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time1800[ctrt][ctrz];
      }
    }
    mVDlo = fVDsmp[2];
    mVDhi = fVDsmp[3];
  } else if (vdrift > fVDsmp[1]) {
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time1600[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time1700[ctrt][ctrz];
      }
    }
    mVDlo = fVDsmp[1];
    mVDhi = fVDsmp[2];
  } else if (vdrift > (fVDsmp[0] - 1.0e-5)) {
    for (int ctrt = 0; ctrt < TIMEBINSGARFIELD; ctrt++) {
      for (int ctrz = 0; ctrz < ZBINSGARFIELD; ctrz++) {
        mTimeStruct1[ctrt + ctrz * TIMEBINSGARFIELD] = Time1500[ctrt][ctrz];
        mTimeStruct2[ctrt + ctrz * TIMEBINSGARFIELD] = Time1600[ctrt][ctrz];
      }
    }
    mVDlo = fVDsmp[0];
    mVDhi = fVDsmp[1];
  }
  mInvBinWidth = 1. / (mVDhi - mVDlo);
  return retVal;
}

//_____________________________________________________________________________
float DiffusionAndTimeStructEstimator::timeStruct(float vdrift, float dist, float z, bool* errFlag)
{
  //
  // Applies the time structure of the drift cells (by C.Lippmann).
  // The drift time of electrons to the anode wires depends on the
  // distance to the wire (z) and on the position in the drift region.
  //
  // input :
  // dist = radial distance from (cathode) pad plane [cm]
  // z    = distance from anode wire (parallel to cathode planes) [cm]
  //
  // output :
  // tdrift = the drift time of an electron at the given position
  //
  // We interpolate between the drift time values at the two drift
  // velocities fVDlo and fVDhi, being smaller and larger than
  // fDriftVelocity. We use the two stored drift time maps fTimeStruct1
  // and fTimeStruct2, calculated for the two mentioned drift velocities.
  //

  if (!sampleTimeStruct(vdrift)) {
    *errFlag = true;
  }

  // Indices:
  int r1 = (int)(10 * dist);
  if (r1 < 0) {
    r1 = 0;
  }
  if (r1 > 37) {
    r1 = 37;
  }
  int r2 = r1 + 1;
  if (r2 > 37) {
    r2 = 37;
  }
  const int kz1 = ((int)(100 * z / 2.5));
  const int kz2 = kz1 + 1;

  if ((r1 < 0) || (r1 > 37) || (kz1 < 0) || (kz1 > 10)) {
    LOG(warn) << Form("TRD: Time struct indices out of range: dist=%.2f, z=%.2f, r1=%d, kz1=%d", dist, z, r1, kz1);
  }

  const float ky111 = mTimeStruct1[r1 + 38 * kz1];
  const float ky221 = ((r2 <= 37) && (kz2 <= 10)) ? mTimeStruct1[r2 + 38 * kz2] : mTimeStruct1[37 + 38 * 10];
  const float ky121 = (kz2 <= 10) ? mTimeStruct1[r1 + 38 * kz2] : mTimeStruct1[r1 + 38 * 10];
  const float ky211 = mTimeStruct1[r2 + 38 * kz1];

  const float ky112 = mTimeStruct2[r1 + 38 * kz1];
  const float ky222 = ((r2 <= 37) && (kz2 <= 10)) ? mTimeStruct2[r2 + 38 * kz2] : mTimeStruct2[37 + 38 * 10];
  const float ky122 = (kz2 <= 10) ? mTimeStruct2[r1 + 38 * kz2] : mTimeStruct2[r1 + 38 * 10];
  const float ky212 = mTimeStruct2[r2 + 38 * kz1];

  // Interpolation in dist-directions, lower drift time map
  const float ky11 = (ky211 - ky111) * 10 * dist + ky111 - (ky211 - ky111) * r1;
  const float ky21 = (ky221 - ky121) * 10 * dist + ky121 - (ky221 - ky121) * r1;

  // Interpolation in dist-direction, larger drift time map
  const float ky12 = (ky212 - ky112) * 10 * dist + ky112 - (ky212 - ky112) * r1;
  const float ky22 = (ky222 - ky122) * 10 * dist + ky122 - (ky222 - ky122) * r1;

  // Dist now is the drift distance to the anode wires (negative if electrons are
  // between anode wire plane and cathode pad plane)
  dist -= Geometry::amThick() * 0.5;

  // Interpolation in z-directions, lower drift time map
  const bool condition = ((std::abs(dist) > 0.005) || (z > 0.005));
  const float ktdrift1 =
    condition ? (ky21 - ky11) * 100 * z / 2.5 + ky11 - (ky21 - ky11) * kz1 : 0.0;
  // Interpolation in z-directions, larger drift time map
  const float ktdrift2 =
    condition ? (ky22 - ky12) * 100 * z / 2.5 + ky12 - (ky22 - ky12) * kz1 : 0.0;

  // Interpolation between the values at fVDlo and fVDhi
  const float a = (ktdrift2 - ktdrift1) * mInvBinWidth; // 1./(mVDhi - mVDlo);
  const float b = ktdrift2 - a * mVDhi;
  const float t = a * vdrift + b;

  return t;
}

//_____________________________________________________________________________
bool DiffusionAndTimeStructEstimator::getDiffCoeff(float& dl, float& dt, float vdrift)
{
  //
  // Calculates the diffusion coefficients in longitudinal <dl> and
  // transverse <dt> direction for a given drift velocity <vdrift>
  //

  // Nothing to do
  if (std::abs(mDiffLastVdrift - vdrift) < 0.001) {
    dl = mDiffusionL;
    dt = mDiffusionT;
    return true;
  }
  mDiffLastVdrift = vdrift;

  if (CommonParam::instance()->isXenon()) {
    //
    // Vd and B-field dependent diffusion and Lorentz angle
    //

    // kNb = kNb = kNb
    constexpr int kNb = 5;

    // If looking at compatibility with AliRoot:
    // ibL and ibT are calculated the same way so, just define ib = ibL = ibT
    int ib = ((int)(10 * (CommonParam::instance()->getCachedField() - 0.15)));
    ib = std::max(0, ib);
    ib = std::min(kNb - 1, ib);

    // DiffusionT
    constexpr float p0T[kNb] = {0.009550, 0.009599, 0.009674, 0.009757, 0.009850};
    constexpr float p1T[kNb] = {0.006667, 0.006539, 0.006359, 0.006153, 0.005925};
    constexpr float p2T[kNb] = {-0.000853, -0.000798, -0.000721, -0.000635, -0.000541};
    constexpr float p3T[kNb] = {0.000131, 0.000122, 0.000111, 0.000098, 0.000085};
    // DiffusionL
    constexpr float p0L[kNb] = {0.007440, 0.007493, 0.007513, 0.007672, 0.007831};
    constexpr float p1L[kNb] = {0.019252, 0.018912, 0.018636, 0.018012, 0.017343};
    constexpr float p2L[kNb] = {-0.005042, -0.004926, -0.004867, -0.004650, -0.004424};
    constexpr float p3L[kNb] = {0.000195, 0.000189, 0.000195, 0.000182, 0.000169};

    const float v2 = vdrift * vdrift;
    const float v3 = v2 * vdrift;
    mDiffusionL = p0L[ib] + p1L[ib] * vdrift + p2L[ib] * v2 + p3L[ib] * v3;
    mDiffusionT = p0T[ib] + p1T[ib] * vdrift + p2T[ib] * v2 + p3T[ib] * v3;

    dl = mDiffusionL;
    dt = mDiffusionT;
    return true;
  } else if (CommonParam::instance()->isArgon()) {
    //
    // Diffusion constants and Lorentz angle only for B = 0.5T
    //
    mDiffusionL = 0.0182;
    mDiffusionT = 0.0159;
    dl = mDiffusionL;
    dt = mDiffusionT;
    return true;
  } else {
    return false;
  }
}

} // namespace o2::trd
