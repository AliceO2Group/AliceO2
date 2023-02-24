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

/// @file   TrackParametrizationWithError.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

#ifndef INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_
#define INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_

#include "ReconstructionDataFormats/TrackParametrization.h"

namespace o2
{
namespace track
{

template <typename value_T = float>
class TrackParametrizationWithError : public TrackParametrization<value_T>
{ // track+error parameterization
 public:
  using typename TrackParametrization<value_T>::value_t;
  using typename TrackParametrization<value_T>::dim3_t;
  using typename TrackParametrization<value_T>::dim2_t;
  using typename TrackParametrization<value_T>::params_t;

#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_t>);
#endif

  using covMat_t = gpu::gpustd::array<value_t, kCovMatSize>;
  using MatrixDSym5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepSym<double, kNParams>>;
  using MatrixD5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepStd<double, kNParams, kNParams>>;

  GPUd() TrackParametrizationWithError();
  GPUd() TrackParametrizationWithError(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1, const PID pid = PID::Pion);
  GPUd() TrackParametrizationWithError(const dim3_t& xyz, const dim3_t& pxpypz,
                                       const gpu::gpustd::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true, const PID pid = PID::Pion);

  GPUdDefault() TrackParametrizationWithError(const TrackParametrizationWithError& src) = default;
  GPUdDefault() TrackParametrizationWithError(TrackParametrizationWithError&& src) = default;
  GPUdDefault() TrackParametrizationWithError& operator=(const TrackParametrizationWithError& src) = default;
  GPUdDefault() TrackParametrizationWithError& operator=(TrackParametrizationWithError&& src) = default;
  GPUdDefault() ~TrackParametrizationWithError() = default;
  using TrackParametrization<value_T>::TrackParametrization;

  using TrackParametrization<value_T>::set;
  GPUd() void set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1, const PID pid = PID::Pion);
  GPUd() void set(value_t x, value_t alpha, const value_t* par, const value_t* cov, int charge = 1, const PID pid = PID::Pion);
  GPUd() void set(const dim3_t& xyz, const dim3_t& pxpypz, const gpu::gpustd::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true, const PID pid = PID::Pion);
  GPUd() const covMat_t& getCov() const;
  GPUd() value_t getSigmaY2() const;
  GPUd() value_t getSigmaZY() const;
  GPUd() value_t getSigmaZ2() const;
  GPUd() value_t getSigmaSnpY() const;
  GPUd() value_t getSigmaSnpZ() const;
  GPUd() value_t getSigmaSnp2() const;
  GPUd() value_t getSigmaTglY() const;
  GPUd() value_t getSigmaTglZ() const;
  GPUd() value_t getSigmaTglSnp() const;
  GPUd() value_t getSigmaTgl2() const;
  GPUd() value_t getSigma1PtY() const;
  GPUd() value_t getSigma1PtZ() const;
  GPUd() value_t getSigma1PtSnp() const;
  GPUd() value_t getSigma1PtTgl() const;
  GPUd() value_t getSigma1Pt2() const;
  GPUd() value_t getCovarElem(int i, int j) const;
  GPUd() value_t getDiagError2(int i) const;

  GPUd() bool getCovXYZPxPyPzGlo(gpu::gpustd::array<value_t, kLabCovMatSize>& c) const;

  GPUd() void print() const;
#ifndef GPUCA_ALIGPUCODE
  std::string asString() const;
#endif

  // parameters + covmat manipulation
  GPUd() bool rotate(value_t alpha);
  GPUd() bool propagateTo(value_t xk, value_t b);
  GPUd() bool propagateTo(value_t xk, const dim3_t& b);
  GPUd() bool propagateToDCA(const o2::dataformats::VertexBase& vtx, value_t b, o2::dataformats::DCA* dca = nullptr, value_t maxD = 999.f);
  GPUd() void invert();

  GPUd() value_t getPredictedChi2(const dim2_t& p, const dim3_t& cov) const;
  GPUd() value_t getPredictedChi2(const value_t* p, const value_t* cov) const;

  template <typename T>
  GPUd() value_t getPredictedChi2(const BaseCluster<T>& p) const;

  void buildCombinedCovMatrix(const TrackParametrizationWithError& rhs, MatrixDSym5& cov) const;
  value_t getPredictedChi2(const TrackParametrizationWithError& rhs, MatrixDSym5& covToSet) const;
  value_t getPredictedChi2(const TrackParametrizationWithError& rhs) const;
  bool update(const TrackParametrizationWithError& rhs, const MatrixDSym5& covInv);
  bool update(const TrackParametrizationWithError& rhs);

  GPUd() bool update(const dim2_t& p, const dim3_t& cov);
  GPUd() bool update(const value_t* p, const value_t* cov);
  GPUd() value_T update(const o2::dataformats::VertexBase& vtx, value_T maxChi2 = 1e15);

  template <typename T>
  GPUd() bool update(const BaseCluster<T>& p);

  GPUd() bool correctForMaterial(value_t x2x0, value_t xrho, bool anglecorr = false, value_t dedx = kCalcdEdxAuto);

  GPUd() void resetCovariance(value_t s2 = 0);
  GPUd() void checkCovariance();
  GPUd() void checkCorrelations();
  GPUd() void setCov(value_t v, size_t i, size_t j);
  GPUd() void setCov(value_t v, int i);
  GPUd() void setCov(const covMat_t& mat);

  GPUd() void updateCov(const covMat_t& delta);
  GPUd() void updateCov(value_t delta, size_t i, size_t j);
  GPUd() void updateCov(value_t delta, size_t i);

  GPUd() void updateCov(const params_t delta2, bool preserveCorrelations);
  GPUd() void updateCov(const value_t* delta2, bool preserveCorrelations);

  GPUd() void updateCovCorr(const params_t delta2);
  GPUd() void updateCovCorr(const value_t* delta2);

  GPUd() void updateCov(const params_t delta2);
  GPUd() void updateCov(const value_t* delta2);

 protected:
  covMat_t mC{0.f}; // 15 covariance matrix elements

  ClassDefNV(TrackParametrizationWithError, 2);
};

//__________________________________________________________________________
template <typename value_T>
GPUdi() TrackParametrizationWithError<value_T>::TrackParametrizationWithError() : TrackParametrization<value_T>{}
{
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() TrackParametrizationWithError<value_T>::TrackParametrizationWithError(value_t x, value_t alpha, const params_t& par,
                                                                              const covMat_t& cov, int charge, const PID pid)
  : TrackParametrization<value_T>{x, alpha, par, charge, pid}
{
  // explicit constructor
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge, const PID pid)
{
  set(x, alpha, par.data(), cov.data(), charge, pid);
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::set(value_t x, value_t alpha, const value_t* par, const value_t* cov, int charge, const PID pid)
{
  TrackParametrization<value_T>::set(x, alpha, par, charge, pid);
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getCov() const -> const covMat_t&
{
  return mC;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaY2() const -> value_t
{
  return mC[kSigY2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaZY() const -> value_t
{
  return mC[kSigZY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaZ2() const -> value_t
{
  return mC[kSigZ2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaSnpY() const -> value_t
{
  return mC[kSigSnpY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaSnpZ() const -> value_t
{
  return mC[kSigSnpZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaSnp2() const -> value_t
{
  return mC[kSigSnp2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaTglY() const -> value_t
{
  return mC[kSigTglY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaTglZ() const -> value_t
{
  return mC[kSigTglZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaTglSnp() const -> value_t
{
  return mC[kSigTglSnp];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigmaTgl2() const -> value_t
{
  return mC[kSigTgl2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigma1PtY() const -> value_t
{
  return mC[kSigQ2PtY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigma1PtZ() const -> value_t
{
  return mC[kSigQ2PtZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigma1PtSnp() const -> value_t
{
  return mC[kSigQ2PtSnp];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigma1PtTgl() const -> value_t
{
  return mC[kSigQ2PtTgl];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getSigma1Pt2() const -> value_t
{
  return mC[kSigQ2Pt2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getCovarElem(int i, int j) const -> value_t
{
  return mC[CovarMap[i][j]];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getDiagError2(int i) const -> value_t
{
  return mC[DiagMap[i]];
}

//__________________________________________________________________________
template <typename value_T>
template <typename T>
GPUdi() auto TrackParametrizationWithError<value_T>::getPredictedChi2(const BaseCluster<T>& p) const -> value_t
{
  const dim2_t pyz = {p.getY(), p.getZ()};
  const dim3_t cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
  return getPredictedChi2(pyz, cov);
}

//______________________________________________
template <typename value_T>
GPUdi() auto TrackParametrizationWithError<value_T>::getPredictedChi2(const dim2_t& p, const dim3_t& cov) const -> value_t
{
  return getPredictedChi2(p.data(), cov.data());
}

//______________________________________________
template <typename value_T>
GPUdi() bool TrackParametrizationWithError<value_T>::update(const dim2_t& p, const dim3_t& cov)
{
  return update(p.data(), cov.data());
}

//__________________________________________________________________________
template <typename value_T>
template <typename T>
GPUdi() bool TrackParametrizationWithError<value_T>::update(const BaseCluster<T>& p)
{
  const dim2_t pyz = {p.getY(), p.getZ()};
  const dim3_t cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
  return update(pyz, cov);
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::setCov(value_t v, int i)
{
  mC[i] = v;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::setCov(value_t v, size_t i, size_t j)
{
  mC[CovarMap[i][j]] = v;
}

template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::setCov(const covMat_t& cov)
{
  mC = cov;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(value_t delta, size_t i, size_t j)
{
  mC[CovarMap[i][j]] += delta;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(value_t delta, size_t i)
{
  mC[i] += delta;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const covMat_t& delta)
{
  for (size_t i = 0; i < kCovMatSize; ++i) {
    mC[i] += delta[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const params_t delta2)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas
  updateCov(delta2.data());
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const value_t* delta2)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas
  for (int i = 0; i < kNParams; i++) {
    mC[DiagMap[i]] += delta2[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCovCorr(const params_t delta2)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas, modify non-diagonal elements to preserve correlations
  updateCovCorr(delta2.data());
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCovCorr(const value_t* delta2)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas, modify non-diagonal elements to preserve correlations
#pragma GCC diagnostic push // FIXME: remove in the future, GCC compiler bug reports incorrect uninitialized warning for oldDiag
#pragma GCC diagnostic ignored "-Wuninitialized"
  value_t oldDiag[kNParams];
  for (int i = 0; i < kNParams; i++) {
    auto diagI = DiagMap[i];
    oldDiag[i] = mC[diagI];
    mC[diagI] += delta2[i];
    for (int j = 0; j < i; i++) {
      mC[CovarMap[i][j]] *= gpu::CAMath::Sqrt(mC[diagI] * mC[DiagMap[j]] / (oldDiag[i] * oldDiag[j]));
    }
  }
#pragma GCC diagnostic pop
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const params_t delta2, bool preserveCorrelations)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas. If requested, modify non-diagonal elements to preserve correlations
  updateCov(delta2.data(), preserveCorrelations);
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const value_t* delta2, bool preserveCorrelations)
{
  // Increment cov.matrix diagonal elements by the vector of squared deltas. If requested, modify non-diagonal elements to preserve correlations
  if (preserveCorrelations) {
    updateCovCorr(delta2);
  } else {
    updateCov(delta2);
  }
}

} // namespace track
} // namespace o2
#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_ */
