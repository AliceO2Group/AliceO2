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

#include "GPUCommonRtypes.h"
#include "GPUCommonArray.h"
#include "CommonConstants/PhysicsConstants.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "DetectorsBase/MatLayerCylSet.h"

#ifndef GPUCA_GPUCODE
#include <string>
#endif

namespace o2
{
namespace parameters
{
class GRPObject;
}

namespace dataformats
{
class VertexBase;
}

namespace field
{
class MagFieldFast;
}

namespace gpu
{
class GPUTPCGMPolynomialField;
}

namespace base
{
class Propagator
{
 public:
  enum class MatCorrType : int {
    USEMatCorrNONE, // flag to not use material corrections
    USEMatCorrTGeo, // flag to use TGeo for material queries
    USEMatCorrLUT
  }; // flag to use LUT for material queries (user must provide a pointer

  static constexpr float MAX_SIN_PHI = 0.85f;
  static constexpr float MAX_STEP = 2.0f;

  GPUd() bool PropagateToXBxByBz(o2::track::TrackParCov& track, float x,
                                 float maxSnp = MAX_SIN_PHI, float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                 o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool PropagateToXBxByBz(o2::track::TrackPar& track, float x,
                                 float maxSnp = MAX_SIN_PHI, float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                 o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToX(o2::track::TrackParCov& track, float x, float bZ,
                           float maxSnp = MAX_SIN_PHI, float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                           o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToX(o2::track::TrackPar& track, float x, float bZ,
                           float maxSnp = MAX_SIN_PHI, float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                           o2::track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToDCA(const o2::dataformats::VertexBase& vtx, o2::track::TrackParCov& track, float bZ,
                             float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                             o2::dataformats::DCA* dcaInfo = nullptr, o2::track::TrackLTIntegral* tofInfo = nullptr,
                             int signCorr = 0, float maxD = 999.f) const;

  GPUd() bool propagateToDCABxByBz(const o2::dataformats::VertexBase& vtx, o2::track::TrackParCov& track,
                                   float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                   o2::dataformats::DCA* dcaInfo = nullptr, o2::track::TrackLTIntegral* tofInfo = nullptr,
                                   int signCorr = 0, float maxD = 999.f) const;

  GPUd() bool propagateToDCA(const o2::math_utils::Point3D<float>& vtx, o2::track::TrackPar& track, float bZ,
                             float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                             gpu::gpustd::array<float, 2>* dca = nullptr, o2::track::TrackLTIntegral* tofInfo = nullptr,
                             int signCorr = 0, float maxD = 999.f) const;

  GPUd() bool propagateToDCABxByBz(const o2::math_utils::Point3D<float>& vtx, o2::track::TrackPar& track,
                                   float maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                   gpu::gpustd::array<float, 2>* dca = nullptr, o2::track::TrackLTIntegral* tofInfo = nullptr,
                                   int signCorr = 0, float maxD = 999.f) const;

  Propagator(Propagator const&) = delete;
  Propagator(Propagator&&) = delete;
  Propagator& operator=(Propagator const&) = delete;
  Propagator& operator=(Propagator&&) = delete;

  // Bz at the origin
  GPUd() float getNominalBz() const { return mBz; }

  GPUd() void setMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }
  GPUd() const o2::base::MatLayerCylSet* getMatLUT() const { return mMatLUT; }
  GPUd() void setGPUField(const o2::gpu::GPUTPCGMPolynomialField* field) { mGPUField = field; }
  GPUd() const o2::gpu::GPUTPCGMPolynomialField* getGPUField() const { return mGPUField; }
  GPUd() void setBz(float bz) { mBz = bz; }

#ifndef GPUCA_GPUCODE
  static Propagator* Instance(bool uninitialized = false)
  {
    static Propagator instance(uninitialized);
    return &instance;
  }

  static int initFieldFromGRP(const o2::parameters::GRPObject* grp, bool verbose = false);
  static int initFieldFromGRP(const std::string grpFileName, std::string grpName = "GRP", bool verbose = false);
#endif

  GPUd() MatBudget getMatBudget(MatCorrType corrType, const o2::math_utils::Point3D<float>& p0, const o2::math_utils::Point3D<float>& p1) const;
  GPUd() void getFiedXYZ(const math_utils::Point3D<float> xyz, float* bxyz) const;

 private:
#ifndef GPUCA_GPUCODE
  Propagator(bool uninitialized = false);
  ~Propagator() = default;
#endif

  const o2::field::MagFieldFast* mField = nullptr; ///< External fast field (barrel only for the moment)
  float mBz = 0;                                   // nominal field

  const o2::base::MatLayerCylSet* mMatLUT = nullptr;           // externally set LUT
  const o2::gpu::GPUTPCGMPolynomialField* mGPUField = nullptr; // externally set GPU Field

  ClassDef(Propagator, 0);
};
} // namespace base
} // namespace o2

#endif
