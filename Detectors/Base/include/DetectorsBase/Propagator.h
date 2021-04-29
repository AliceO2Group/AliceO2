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
  GPUd() value_type getNominalBz() const { return mBz; }

  GPUd() void setMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }
  GPUd() const o2::base::MatLayerCylSet* getMatLUT() const { return mMatLUT; }
  GPUd() void setGPUField(const o2::gpu::GPUTPCGMPolynomialField* field) { mGPUField = field; }
  GPUd() const o2::gpu::GPUTPCGMPolynomialField* getGPUField() const { return mGPUField; }
  GPUd() void setBz(value_type bz) { mBz = bz; }

#ifndef GPUCA_GPUCODE
  static PropagatorImpl* Instance(bool uninitialized = false)
  {
    static PropagatorImpl instance(uninitialized);
    return &instance;
  }

  static int initFieldFromGRP(const o2::parameters::GRPObject* grp, bool verbose = false);
  static int initFieldFromGRP(const std::string grpFileName = "", std::string grpName = "GRP", bool verbose = false);
#endif

  GPUd() MatBudget getMatBudget(MatCorrType corrType, const o2::math_utils::Point3D<value_type>& p0, const o2::math_utils::Point3D<value_type>& p1) const;

  GPUd() void getFieldXYZ(const math_utils::Point3D<float> xyz, float* bxyz) const;

  GPUd() void getFieldXYZ(const math_utils::Point3D<double> xyz, double* bxyz) const;

 private:
#ifndef GPUCA_GPUCODE
  PropagatorImpl(bool uninitialized = false);
  ~PropagatorImpl() = default;
#endif

  template <typename T>
  GPUd() void getFieldXYZImpl(const math_utils::Point3D<T> xyz, T* bxyz) const;

  const o2::field::MagFieldFast* mField = nullptr; ///< External fast field (barrel only for the moment)
  value_type mBz = 0;                              // nominal field

  const o2::base::MatLayerCylSet* mMatLUT = nullptr;           // externally set LUT
  const o2::gpu::GPUTPCGMPolynomialField* mGPUField = nullptr; // externally set GPU Field

  ClassDefNV(PropagatorImpl, 0);
};

using PropagatorF = PropagatorImpl<float>;
using PropatatorD = PropagatorImpl<double>;
using Propagator = PropagatorF;

} // namespace base
} // namespace o2

#endif
