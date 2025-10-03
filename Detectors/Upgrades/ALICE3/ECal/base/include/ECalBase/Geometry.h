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

/// \file Geometry.h
/// \brief Geometry helper class
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_GEOMETRY_H
#define ALICEO2_ECAL_GEOMETRY_H

#include <vector>
#include <math.h>
#include <Rtypes.h>

namespace o2
{
namespace ecal
{
class Geometry
{
 public:
  static Geometry& instance()
  {
    static Geometry sGeom;
    return sGeom;
  }

  int getNcols() const;
  int getNrows() const;
  std::pair<int, int> getSectorChamber(int cellId) const;
  std::pair<int, int> getSectorChamber(int iphi, int iz) const;

  void fillFrontFaceCenterCoordinates();
  int getCellID(int moduleId, int sectorId, bool isCrystal);
  void detIdToRelIndex(int cellId, int& chamber, int& sector, int& iphi, int& iz) const;
  void detIdToGlobalPosition(int detId, double& x, double& y, double& z);
  std::pair<int, int> globalRowColFromIndex(int cellID) const;
  bool isCrystal(int cellID);
  int areNeighboursVertex(int detId1, int detId2) const;

  double getTanBeta(int i) { return mTanBeta[i]; }
  double getFrontFaceZatMinR(int i) { return mFrontFaceZatMinR[i]; }
  double getFrontFaceCenterR(int i) { return mFrontFaceCenterR[i]; }
  double getFrontFaceCenterZ(int i) { return mFrontFaceCenterZ[i]; }
  double getFrontFaceCenterSamplingPhi(int i) { return mFrontFaceCenterSamplingPhi[i]; }
  double getFrontFaceCenterCrystalPhi(int i) { return mFrontFaceCenterCrystalPhi[i]; }
  double getFrontFaceCenterTheta(int i) { return mFrontFaceCenterTheta[i]; }
  double getRMin() { return mRMin; }
  double getCrystalModW() { return mCrystalModW; }
  double getSamplingModW() { return mSamplingModW; }
  double getCrystalAlpha() { return mCrystalAlpha; }
  double getSamplingAlpha() { return mSamplingAlpha; }
  double getCrystalDeltaPhi() { return 2 * std::atan(mCrystalModW / 2 / mRMin); }
  double getSamplingDeltaPhi() { return 2 * std::atan(mSamplingModW / 2 / mRMin); }
  double getCrystalPhiMin();
  double getSamplingPhiMin();
  int getNModulesZ() { return mNModulesZ; }
  bool isAtTheEdge(int cellId);

 private:
  Geometry();
  Geometry(const Geometry&) = delete;
  Geometry& operator=(const Geometry&) = delete;
  ~Geometry() = default;
  double mRMin{0.};
  int mNSuperModules{0};
  int mNCrystalModulesZ{0};
  int mNSamplingModulesZ{0};
  int mNCrystalModulesPhi{0};
  int mNSamplingModulesPhi{0};
  double mCrystalModW{0.};
  double mSamplingModW{0.};
  double mSamplingAlpha{0.};
  double mCrystalAlpha{0.};
  double mMarginCrystalToSampling{0.};
  int mNModulesZ{0};
  std::vector<double> mFrontFaceZatMinR;
  std::vector<double> mFrontFaceCenterR;
  std::vector<double> mFrontFaceCenterZ;
  std::vector<double> mFrontFaceCenterSamplingPhi;
  std::vector<double> mFrontFaceCenterCrystalPhi;
  std::vector<double> mFrontFaceCenterTheta;
  std::vector<double> mTanBeta;

  ClassDefNV(Geometry, 1);
};
} // namespace ecal
} // namespace o2

#endif // ALICEO2_ECAL_GEOMETRY_H
