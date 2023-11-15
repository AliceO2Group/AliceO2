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
class GRPMagField;
} // namespace parameters

namespace dataformats
{
class VertexBase;
}

namespace field
{
class MagFieldFast;
class MagneticField;
} // namespace field

namespace gpu
{
class GPUTPCGMPolynomialField;
}

namespace base
{

template <typename value_T>
class PropagatorImpl
{
 public:
  using value_type = value_T;
  using TrackPar_t = track::TrackParametrization<value_type>;
  using TrackParCov_t = track::TrackParametrizationWithError<value_type>;

  enum class MatCorrType : int {
    USEMatCorrNONE, // flag to not use material corrections
    USEMatCorrTGeo, // flag to use TGeo for material queries
    USEMatCorrLUT
  }; // flag to use LUT for material queries (user must provide a pointer

  static constexpr float MAX_SIN_PHI = 0.85f;
  static constexpr float MAX_STEP = 2.0f;

  GPUd() bool PropagateToXBxByBz(TrackParCov_t& track, value_type x,
                                 value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                 track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool PropagateToXBxByBz(TrackPar_t& track, value_type x,
                                 value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                 track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToX(TrackParCov_t& track, value_type x, value_type bZ,
                           value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                           track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToX(TrackPar_t& track, value_type x, value_type bZ,
                           value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                           track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  template <typename track_T>
  GPUd() bool propagateTo(track_T& track, value_type x, bool bzOnly = false, value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP,
                          MatCorrType matCorr = MatCorrType::USEMatCorrLUT, track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const
  {
    return bzOnly ? propagateToX(track, x, getNominalBz(), maxSnp, maxStep, matCorr, tofInfo, signCorr) : PropagateToXBxByBz(track, x, maxSnp, maxStep, matCorr, tofInfo, signCorr);
  }

  template <typename track_T>
  GPUd() bool propagateToAlphaX(track_T& track, value_type alpha, value_type x, bool bzOnly = false, value_type maxSnp = MAX_SIN_PHI, value_type maxStep = MAX_STEP, int minSteps = 1,
                                MatCorrType matCorr = MatCorrType::USEMatCorrLUT, track::TrackLTIntegral* tofInfo = nullptr, int signCorr = 0) const;

  GPUd() bool propagateToDCA(const o2::dataformats::VertexBase& vtx, o2::track::TrackParametrizationWithError<value_type>& track, value_type bZ,
                             value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                             o2::dataformats::DCA* dcaInfo = nullptr, track::TrackLTIntegral* tofInfo = nullptr,
                             int signCorr = 0, value_type maxD = 999.f) const;

  GPUd() bool propagateToDCABxByBz(const o2::dataformats::VertexBase& vtx, o2::track::TrackParametrizationWithError<value_type>& track,
                                   value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                   o2::dataformats::DCA* dcaInfo = nullptr, track::TrackLTIntegral* tofInfo = nullptr,
                                   int signCorr = 0, value_type maxD = 999.f) const;

  GPUd() bool propagateToDCA(const o2::math_utils::Point3D<value_type>& vtx, o2::track::TrackParametrization<value_type>& track, value_type bZ,
                             value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                             gpu::gpustd::array<value_type, 2>* dca = nullptr, track::TrackLTIntegral* tofInfo = nullptr,
                             int signCorr = 0, value_type maxD = 999.f) const;

  GPUd() bool propagateToDCABxByBz(const o2::math_utils::Point3D<value_type>& vtx, o2::track::TrackParametrization<value_type>& track,
                                   value_type maxStep = MAX_STEP, MatCorrType matCorr = MatCorrType::USEMatCorrLUT,
                                   gpu::gpustd::array<value_type, 2>* dca = nullptr, track::TrackLTIntegral* tofInfo = nullptr,
                                   int signCorr = 0, value_type maxD = 999.f) const;

  PropagatorImpl(PropagatorImpl const&) = delete;
  PropagatorImpl(PropagatorImpl&&) = delete;
  PropagatorImpl& operator=(PropagatorImpl const&) = delete;
  PropagatorImpl& operator=(PropagatorImpl&&) = delete;

  // Bz at the origin
  GPUd() void updateField();
  GPUd() value_type getNominalBz() const { return mBz; }
  GPUd() void setTGeoFallBackAllowed(bool v) { mTGeoFallBackAllowed = v; }
  GPUd() bool isTGeoFallBackAllowed() const { return mTGeoFallBackAllowed; }
  GPUd() void setMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }
  GPUd() const o2::base::MatLayerCylSet* getMatLUT() const { return mMatLUT; }
  GPUd() void setGPUField(const o2::gpu::GPUTPCGMPolynomialField* field) { mGPUField = field; }
  GPUd() const o2::gpu::GPUTPCGMPolynomialField* getGPUField() const { return mGPUField; }
  GPUd() void setBz(value_type bz) { mBz = bz; }
  GPUd() bool hasMagFieldSet() const { return mField != nullptr; }

  GPUd() void estimateLTFast(o2::track::TrackLTIntegral& lt, const o2::track::TrackParametrization<value_type>& trc) const;

#ifndef GPUCA_GPUCODE
  static PropagatorImpl* Instance(bool uninitialized = false)
  {
    static PropagatorImpl instance(uninitialized);
    return &instance;
  }
  static int initFieldFromGRP(const o2::parameters::GRPMagField* grp, bool verbose = false);

  static int initFieldFromGRP(const o2::parameters::GRPObject* grp, bool verbose = false);
  static int initFieldFromGRP(const std::string grpFileName = "", bool verbose = false);
#endif

  GPUd() MatBudget getMatBudget(MatCorrType corrType, const o2::math_utils::Point3D<value_type>& p0, const o2::math_utils::Point3D<value_type>& p1) const;

  GPUd() void getFieldXYZ(const math_utils::Point3D<float> xyz, float* bxyz) const;

  GPUd() void getFieldXYZ(const math_utils::Point3D<double> xyz, double* bxyz) const;

 private:
#ifndef GPUCA_GPUCODE
  PropagatorImpl(bool uninitialized = false);
  ~PropagatorImpl() = default;
#endif
  static constexpr value_type Epsilon = 0.00001; // precision of propagation to X
  template <typename T>
  GPUd() void getFieldXYZImpl(const math_utils::Point3D<T> xyz, T* bxyz) const;

  const o2::field::MagFieldFast* mFieldFast = nullptr; ///< External fast field map (barrel only for the moment)
  o2::field::MagneticField* mField = nullptr;          ///< External nominal field map
  value_type mBz = 0;                                  ///< nominal field

  bool mTGeoFallBackAllowed = true;                            ///< allow fall back to TGeo if requested MatLUT is not available
  const o2::base::MatLayerCylSet* mMatLUT = nullptr;           // externally set LUT
  const o2::gpu::GPUTPCGMPolynomialField* mGPUField = nullptr; // externally set GPU Field

  ClassDefNV(PropagatorImpl, 0);
};

using PropagatorF = PropagatorImpl<float>;
using PropagatorD = PropagatorImpl<double>;
using Propagator = PropagatorF;

} // namespace base
} // namespace o2

#endif
