// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Propagator
/// \brief Singleton class for track propagation routines
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_PROPAGATOR_
#define ALICEO2_BASE_PROPAGATOR_

#include <string>
#include "CommonConstants/PhysicsConstants.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/GeometryManager.h"

namespace o2
{
namespace parameters
{
class GRPObject;
}

namespace field
{
class MagFieldFast;
}

namespace base
{
class Propagator
{
 public:
  static constexpr int USEMatCorrNONE = 0; // flag to not use material corrections
  static constexpr int USEMatCorrTGeo = 1; // flag to use TGeo for material queries
  static constexpr int USEMatCorrLUT = 2;  // flag to use LUT for material queries (user must provide a pointer

  static Propagator* Instance()
  {
    static Propagator instance;
    return &instance;
  }

  bool PropagateToXBxByBz(o2::track::TrackParCov& track, float x, float mass = o2::constants::physics::MassPionCharged,
                          float maxSnp = 0.85, float maxStep = 2.0, int matCorr = 1,
                          o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  bool propagateToX(o2::track::TrackParCov& track, float x, float bZ, float mass = o2::constants::physics::MassPionCharged,
                    float maxSnp = 0.85, float maxStep = 2.0, int matCorr = 1,
                    o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  bool propagateToDCA(const Point3D<float>& vtx, o2::track::TrackParCov& track, float bZ,
                      float mass = o2::constants::physics::MassPionCharged, float maxStep = 2.0, int matCorr = 1,
                      o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0, float maxD = 999.f) const;

  Propagator(Propagator const&) = delete;
  Propagator(Propagator&&) = delete;
  Propagator& operator=(Propagator const&) = delete;
  Propagator& operator=(Propagator&&) = delete;

  // Bz at the origin
  float getNominalBz() const { return mBz; }

  void setMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }
  const o2::base::MatLayerCylSet* getMatLUT() const { return mMatLUT; }

  static int initFieldFromGRP(const o2::parameters::GRPObject* grp);
  static int initFieldFromGRP(const std::string grpFileName, std::string grpName = "GRP");

 private:
  Propagator();
  ~Propagator() = default;

  MatBudget getMatBudget(int corrType, const Point3D<float>& p0, const Point3D<float>& p1) const
  {
    return (corrType == USEMatCorrTGeo) ? GeometryManager::meanMaterialBudget(p0, p1) : mMatLUT->getMatBudget(p0.X(), p0.Y(), p0.Z(), p1.X(), p1.Y(), p1.Z());
  }

  const o2::field::MagFieldFast* mField = nullptr; ///< External fast field (barrel only for the moment)
  float mBz = 0;                                   // nominal field

  const o2::base::MatLayerCylSet* mMatLUT = nullptr; // externally set LUT

  ClassDef(Propagator, 0);
};
} // namespace base
} // namespace o2

#endif
