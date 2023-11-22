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

#include "RICHSimulation/RICHRing.h"
#include "RICHBase/GeometryTGeo.h"
#include "RICHBase/RICHBaseParam.h"
#include "Framework/Logger.h"

#include <TGeoTube.h>
#include <TGeoVolume.h>

namespace o2
{
namespace rich
{
Ring::Ring(int rPosId, int nTiles, float angleM)
  : mNTiles{nTiles}, mPosId{rPosId}, mMAngle{angleM}
{
  // LOGP(info, "Created ring: id: {} | angleM: {} | t: {} | parA: {} | parB: {} | valM: {} | thetab: {} | mRTilt: {} | mZTilt: {} |", rPosId, angleM, t, parA, parB, valM, thetab, mRTilt, mZTilt);
}

void Ring::generateAllRings(TGeoVolume* mother)
{
  // Mere transcription of Nicola's code
  auto& richPars = RICHBaseParam::Instance();
  std::vector<double> thetaBi(richPars.nRings),
    r0Tilt(richPars.nRings),
    z0Tilt(richPars.nRings),
    lAerogelZ(richPars.nRings),
    tRplusG(richPars.nRings),
    minRadialMirror(richPars.nRings),
    maxRadialMirror(richPars.nRings),
    maxRadialRadiator(richPars.nRings),
    vMirror1(richPars.nRings),
    vMirror2(richPars.nRings),
    vTile1(richPars.nRings),
    vTile2(richPars.nRings);
  // Start from middle one
  double mVal = TMath::Tan(0.0);
  thetaBi[richPars.nRings / 2] = TMath::ATan(mVal);
  r0Tilt[richPars.nRings / 2] = richPars.rMax;
  z0Tilt[richPars.nRings / 2] = r0Tilt[richPars.nRings / 2] * TMath::Tan(thetaBi[richPars.nRings / 2]);
  lAerogelZ[richPars.nRings / 2] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
  tRplusG[richPars.nRings / 2] = richPars.rMax - richPars.rMin;
  double t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
  minRadialMirror[richPars.nRings / 2] = richPars.rMax;
  maxRadialMirror[richPars.nRings / 2] = richPars.rMin;

  // Configure rest of the rings
  for (auto iRing{richPars.nRings / 2 + 1}; iRing < richPars.nRings; ++iRing) {
    double parA = t;
    double parB = 2.0 * richPars.rMax / richPars.zBaseSize;
    mVal = (TMath::Sqrt(parA * parA * parB * parB + parB * parB - 1.0) + parA * parB * parB) / (parB * parB - 1.0);
    t = tan(atan(mVal) + atan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
    // forward rings
    thetaBi[iRing] = atan(mVal);
    r0Tilt[iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(atan(mVal));
    z0Tilt[iRing] = r0Tilt[iRing] * tan(thetaBi[iRing]);
    lAerogelZ[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    tRplusG[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + lAerogelZ[iRing]);
    minRadialMirror[iRing] = r0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(atan(mVal));
    maxRadialMirror[iRing] = richPars.rMin + 2.0 * lAerogelZ[iRing] / 2.0 * sin(atan(mVal));
    // backward rings
    thetaBi[2 * richPars.nRings / 2 - iRing] = -atan(mVal);
    r0Tilt[2 * richPars.nRings / 2 - iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(atan(mVal));
    z0Tilt[2 * richPars.nRings / 2 - iRing] = -r0Tilt[iRing] * tan(thetaBi[iRing]);
    lAerogelZ[2 * richPars.nRings / 2 - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    tRplusG[2 * richPars.nRings / 2 - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + lAerogelZ[iRing]);
    minRadialMirror[2 * richPars.nRings / 2 - iRing] = r0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(atan(mVal));
    maxRadialMirror[2 * richPars.nRings / 2 - iRing] = richPars.rMin + 2.0 * lAerogelZ[iRing] / 2.0 * sin(atan(mVal));
  }

  // Dimensioning tiles
  double percentage = 0.999;
  for (int iRing = 0; iRing < richPars.nRings; iRing++) {
    if (iRing == richPars.nRings / 2) {
      vMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    } else if (iRing > richPars.nRings / 2) {
      vMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vMirror2[iRing] = percentage * 2.0 * minRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile1[iRing] = percentage * 2.0 * maxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    } else if (iRing < richPars.nRings / 2) {
      vMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vMirror1[iRing] = percentage * 2.0 * minRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile2[iRing] = percentage * 2.0 * maxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      vTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    }
  }
  // Aerogel blocs translation parameters
  std::vector<double> r0Aerogel(richPars.nRings);
  for (int iRing{0}; iRing < richPars.nRings; iRing++) {
    r0Aerogel[iRing] = r0Tilt[iRing] - (tRplusG[iRing] - richPars.radiatorThickness / 2.0) * TMath::Cos(thetaBi[iRing]);
  }
  // Detector planes translation parameters
  std::vector<double> r0Detector(richPars.nRings);
  for (int iRing{0}; iRing < richPars.nRings; iRing++) {
    r0Detector[iRing] = r0Tilt[iRing] + (richPars.detectorThickness / 2.0) * TMath::Cos(thetaBi[iRing]);
  }
}


} // namespace rich
} // namespace o2