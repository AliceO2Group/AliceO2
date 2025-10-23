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

/// \file Geometry.cxx
/// \brief Geometry helper class
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include "TMath.h"
#include <Math/Point2D.h>
#include <Math/Vector2D.h>
#include <ECalBase/Geometry.h>
#include <ECalBase/ECalBaseParam.h>
#include "CommonConstants/MathConstants.h"
using namespace o2::ecal;
using o2::constants::math::PIHalf;
using o2::constants::math::TwoPI;

//==============================================================================
Geometry::Geometry()
{
  auto& pars = ECalBaseParam::Instance();
  pars.updateFromFile("o2sim_configuration.ini", "ECalBase");
  pars.printKeyValues(false, true, true, false);
  mRMin = pars.rMin;
  mNSuperModules = pars.nSuperModules;
  mNCrystalModulesZ = pars.nCrystalModulesZ;
  mNSamplingModulesZ = pars.nSamplingModulesZ;
  mNCrystalModulesPhi = pars.nCrystalModulesPhi;
  mNSamplingModulesPhi = pars.nSamplingModulesPhi;
  mCrystalModW = pars.crystalModuleWidth;
  mSamplingModW = pars.samplingModuleWidth;
  mMarginCrystalToSampling = pars.marginCrystalToSampling;
  mCrystalAlpha = pars.crystalAlphaDeg * TMath::DegToRad();
  mSamplingAlpha = pars.samplingAlphaDeg * TMath::DegToRad();
  mNModulesZ = 2 * mNSamplingModulesZ + 2 * mNCrystalModulesZ;
  fillFrontFaceCenterCoordinates();
}

//==============================================================================
int Geometry::getNcols() const
{
  return mNModulesZ;
}

//==============================================================================
int Geometry::getNrows() const
{
  return mNSuperModules * (mNCrystalModulesPhi > mNSamplingModulesPhi ? mNCrystalModulesPhi : mNSamplingModulesPhi);
}

//==============================================================================
double Geometry::getCrystalPhiMin()
{
  double superModuleDeltaPhi = TwoPI / mNSuperModules;
  double crystalDeltaPhi = getCrystalDeltaPhi();
  return (superModuleDeltaPhi - crystalDeltaPhi * mNCrystalModulesPhi) / 2.;
}

//==============================================================================
double Geometry::getSamplingPhiMin()
{
  double superModuleDeltaPhi = TwoPI / mNSuperModules;
  double samplingDeltaPhi = getSamplingDeltaPhi();
  return (superModuleDeltaPhi - samplingDeltaPhi * mNSamplingModulesPhi) / 2.;
}

double Geometry::getFrontFaceMaxEta(int i)
{
  double theta = std::atan(mRMin / getFrontFaceZatMinR(i));
  return -std::log(std::tan(theta / 2.));
}

//==============================================================================
void Geometry::fillFrontFaceCenterCoordinates()
{
  if (mFrontFaceCenterR.size() > 0) {
    return;
  }
  mFrontFaceCenterTheta.resize(mNCrystalModulesZ + mNSamplingModulesZ);
  mFrontFaceZatMinR.resize(mNCrystalModulesZ + mNSamplingModulesZ);
  mFrontFaceCenterR.resize(mNCrystalModulesZ + mNSamplingModulesZ);
  mFrontFaceCenterZ.resize(mNCrystalModulesZ + mNSamplingModulesZ);
  mTanBeta.resize(mNCrystalModulesZ + mNSamplingModulesZ);
  mFrontFaceCenterSamplingPhi.resize(mNSuperModules * mNSamplingModulesPhi);
  mFrontFaceCenterCrystalPhi.resize(mNSuperModules * mNCrystalModulesPhi);

  double superModuleDeltaPhi = TwoPI / mNSuperModules;
  double crystalDeltaPhi = getCrystalDeltaPhi();
  double samplingDeltaPhi = getSamplingDeltaPhi();
  double crystalPhiMin = getCrystalPhiMin();
  double samplingPhiMin = getSamplingPhiMin();
  for (int ism = 0; ism < mNSuperModules; ism++) {
    // crystal
    for (int i = 0; i < mNCrystalModulesPhi; i++) {
      double phi0 = superModuleDeltaPhi * ism + crystalPhiMin + crystalDeltaPhi * i;
      mFrontFaceCenterCrystalPhi[ism * mNCrystalModulesPhi + i] = phi0;
    }
    // sampling
    for (int i = 0; i < mNSamplingModulesPhi; i++) {
      double phi0 = superModuleDeltaPhi * ism + samplingPhiMin + samplingDeltaPhi * i;
      mFrontFaceCenterSamplingPhi[ism * mNSamplingModulesPhi + i] = phi0;
    }
  }

  double theta0 = PIHalf - mCrystalAlpha;
  double zAtMinR = mCrystalModW * std::cos(mCrystalAlpha);

  for (int m = 0; m < mNCrystalModulesZ; m++) {
    mTanBeta[m] = std::sin(theta0 - mCrystalAlpha) * mCrystalModW / 2 / mRMin;
    ROOT::Math::Polar2DVector vMid21(mCrystalModW / 2., PIHalf + theta0);
    ROOT::Math::XYPoint pAtMinR(zAtMinR, mRMin);
    ROOT::Math::XYPoint pc = pAtMinR + vMid21;
    mFrontFaceZatMinR[m] = zAtMinR;
    mFrontFaceCenterZ[m] = pc.x();
    mFrontFaceCenterR[m] = pc.y();
    mFrontFaceCenterTheta[m] = theta0;
    theta0 -= 2 * mCrystalAlpha;
    zAtMinR += mCrystalModW * std::cos(mCrystalAlpha) / std::sin(theta0 + mCrystalAlpha);
  }

  theta0 = mFrontFaceCenterTheta[mNCrystalModulesZ - 1] - mCrystalAlpha - mSamplingAlpha;
  zAtMinR = mFrontFaceZatMinR[mNCrystalModulesZ - 1];
  zAtMinR += mSamplingModW * std::cos(mSamplingAlpha) / std::sin(theta0 + mSamplingAlpha);
  zAtMinR += mMarginCrystalToSampling;

  for (int m = 0; m < mNSamplingModulesZ; m++) {
    int i = m + mNCrystalModulesZ;
    mTanBeta[i] = std::sin(theta0 - mSamplingAlpha) * mSamplingModW / 2 / mRMin;
    ROOT::Math::Polar2DVector vMid21(mSamplingModW / 2., PIHalf + theta0);
    ROOT::Math::XYPoint pAtMinR(zAtMinR, mRMin);
    ROOT::Math::XYPoint pc = pAtMinR + vMid21;
    mFrontFaceZatMinR[i] = zAtMinR;
    mFrontFaceCenterZ[i] = pc.x();
    mFrontFaceCenterR[i] = pc.y();
    mFrontFaceCenterTheta[i] = theta0;
    theta0 -= 2 * mSamplingAlpha;
    zAtMinR += mSamplingModW * std::cos(mSamplingAlpha) / std::sin(theta0 + mSamplingAlpha);
  }
}

int Geometry::getCellID(int moduleId, int sectorId, bool isCrystal)
{
  int cellID = 0;
  if (isCrystal) {
    if (moduleId % 2 == 0) { // crystal at positive eta
      cellID = sectorId * mNModulesZ + moduleId / 2 + mNSamplingModulesZ + mNCrystalModulesZ;
    } else { // crystal at negative eta
      cellID = sectorId * mNModulesZ - moduleId / 2 + mNSamplingModulesZ + mNCrystalModulesZ - 1;
    }
  } else {
    if (sectorId % 2 == 0) { // sampling at positive eta
      cellID = sectorId / 2 * mNModulesZ + moduleId + mNSamplingModulesZ + mNCrystalModulesZ * 2;
    } else { // sampling at negative eta
      cellID = sectorId / 2 * mNModulesZ - moduleId + mNSamplingModulesZ - 1;
    }
  }
  return cellID;
}

//==============================================================================
std::pair<int, int> Geometry::globalRowColFromIndex(int cellID) const
{
  int ip = cellID / mNModulesZ; // row
  int iz = cellID % mNModulesZ; // col
  return {ip, iz};
}

//==============================================================================
bool Geometry::isCrystal(int cellID)
{
  auto [row, col] = globalRowColFromIndex(cellID);
  return (col >= mNSamplingModulesZ && col < mNSamplingModulesZ + 2 * mNCrystalModulesZ);
}

//==============================================================================
std::pair<int, int> Geometry::getSectorChamber(int cellId) const
{
  int iphi = cellId / mNModulesZ;
  int iz = cellId % mNModulesZ;
  return getSectorChamber(iphi, iz);
}

//==============================================================================
std::pair<int, int> Geometry::getSectorChamber(int iphi, int iz) const
{
  int chamber = iz < mNSamplingModulesZ ? 0 : (iz < mNSamplingModulesZ + 2 * mNCrystalModulesZ ? 1 : 2);
  int sector = iphi / (chamber == 1 ? mNCrystalModulesPhi : mNSamplingModulesPhi);
  return {sector, chamber};
}

//==============================================================================
void Geometry::detIdToRelIndex(int cellId, int& chamber, int& sector, int& iphi, int& iz) const
{
  // 3 chambers - sampling/crystal/sampling
  iphi = cellId / mNModulesZ;
  iz = cellId % mNModulesZ;
  auto pair = getSectorChamber(iphi, iz);
  sector = pair.first;
  chamber = pair.second;
}

//==============================================================================
void Geometry::detIdToGlobalPosition(int detId, double& x, double& y, double& z)
{
  int chamber, sector, iphi, iz;
  detIdToRelIndex(detId, chamber, sector, iphi, iz);
  double r = 0;
  if (iz < mNSamplingModulesZ + mNCrystalModulesZ) {
    z = -mFrontFaceCenterZ[mNSamplingModulesZ + mNCrystalModulesZ - iz - 1];
    r = mFrontFaceCenterR[mNSamplingModulesZ + mNCrystalModulesZ - iz - 1];
  } else {
    z = mFrontFaceCenterZ[iz % (mNSamplingModulesZ + mNCrystalModulesZ)];
    r = mFrontFaceCenterR[iz % (mNSamplingModulesZ + mNCrystalModulesZ)];
  }
  double phi = chamber == 1 ? mFrontFaceCenterCrystalPhi[iphi] : mFrontFaceCenterSamplingPhi[iphi];
  x = r * std::cos(phi);
  y = r * std::sin(phi);
}

//==============================================================================
int Geometry::areNeighboursVertex(int detId1, int detId2) const
{
  int ch1, sector1, iphi1, iz1;
  int ch2, sector2, iphi2, iz2;
  detIdToRelIndex(detId1, ch1, sector1, iphi1, iz1);
  detIdToRelIndex(detId2, ch2, sector2, iphi2, iz2);
  if (sector1 != sector2 || ch1 != ch2) {
    return 0;
  }
  if (std::abs(iphi1 - iphi2) <= 1 && std::abs(iz1 - iz2) <= 1) {
    return 1;
  }
  return 0;
}

//==============================================================================
bool Geometry::isAtTheEdge(int cellId)
{
  auto [row, col] = globalRowColFromIndex(cellId);
  if (col == 0) {
    return 1;
  } else if (col == mNSamplingModulesZ) {
    return 1;
  } else if (col == mNSamplingModulesZ - 1) {
    return 1;
  } else if (col == mNSamplingModulesZ + 2 * mNCrystalModulesZ) {
    return 1;
  } else if (col == mNSamplingModulesZ + 2 * mNCrystalModulesZ - 1) {
    return 1;
  } else if (col == mNModulesZ - 1) {
    return 1;
  }
  for (int m = 0; m <= mNSuperModules; m++) {
    if (isCrystal(cellId)) {
      if (row == m * mNCrystalModulesPhi) {
        return 1;
      } else if (row == m * mNCrystalModulesPhi - 1) {
        return 1;
      }
    } else {
      if (row == m * mNSamplingModulesPhi) {
        return 1;
      } else if (row == m * mNSamplingModulesPhi - 1) {
        return 1;
      }
    }
  }
  return 0;
}
