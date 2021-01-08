// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  using MatrixDSym5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepSym<double, kNParams>>;
  using MatrixD5 = ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepStd<double, kNParams, kNParams>>;

  using typename TrackParametrization<value_T>::value_t;
  using typename TrackParametrization<value_T>::dim3_t;
  using typename TrackParametrization<value_T>::dim2_t;
  using typename TrackParametrization<value_T>::params_t;

#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_t>);
#endif

 public:
  using covMat_t = gpu::gpustd::array<value_t, kCovMatSize>;

  GPUd() TrackParametrizationWithError();
  GPUd() TrackParametrizationWithError(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1);
  GPUd() TrackParametrizationWithError(const dim3_t& xyz, const dim3_t& pxpypz,
                                       const gpu::gpustd::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true);

  GPUdDefault() TrackParametrizationWithError(const TrackParametrizationWithError& src) = default;
  GPUdDefault() TrackParametrizationWithError(TrackParametrizationWithError&& src) = default;
  GPUdDefault() TrackParametrizationWithError& operator=(const TrackParametrizationWithError& src) = default;
  GPUdDefault() TrackParametrizationWithError& operator=(TrackParametrizationWithError&& src) = default;
  GPUdDefault() ~TrackParametrizationWithError() = default;
  using TrackParametrization<value_T>::TrackParametrization;

  GPUd() void set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1);
  GPUd() const value_t* getCov() const;
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

  template <typename T>
  GPUd() value_t getPredictedChi2(const BaseCluster<T>& p) const;

  void buildCombinedCovMatrix(const TrackParametrizationWithError& rhs, MatrixDSym5& cov) const;
  value_t getPredictedChi2(const TrackParametrizationWithError& rhs, MatrixDSym5& covToSet) const;
  value_t getPredictedChi2(const TrackParametrizationWithError& rhs) const;
  bool update(const TrackParametrizationWithError& rhs, const MatrixDSym5& covInv);
  bool update(const TrackParametrizationWithError& rhs);

  GPUd() bool update(const dim2_t& p, const dim3_t& cov);

  template <typename T>
  GPUd() bool update(const BaseCluster<T>& p);

  GPUd() bool correctForMaterial(value_t x2x0, value_t xrho, bool anglecorr = false, value_t dedx = kCalcdEdxAuto);

  GPUd() void resetCovariance(value_t s2 = 0);
  GPUd() void checkCovariance();
  GPUd() void setCov(value_t v, int i);

  GPUd() void updateCov(const value_t delta[kCovMatSize]);
  GPUd() void updateCov(value_t delta, int i);

 protected:
  value_t mC[kCovMatSize] = {0.f}; // 15 covariance matrix elements

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
                                                                              const covMat_t& cov, int charge)
  : TrackParametrization<value_T>{x, alpha, par, charge}
{
  // explicit constructor
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge)
{
  TrackParametrization<value_T>::set(x, alpha, par, charge);
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() const typename TrackParametrizationWithError<value_T>::value_t* TrackParametrizationWithError<value_T>::getCov() const
{
  return mC;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaY2() const
{
  return mC[kSigY2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaZY() const
{
  return mC[kSigZY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaZ2() const
{
  return mC[kSigZ2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnpY() const
{
  return mC[kSigSnpY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnpZ() const
{
  return mC[kSigSnpZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnp2() const
{
  return mC[kSigSnp2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglY() const
{
  return mC[kSigTglY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglZ() const
{
  return mC[kSigTglZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglSnp() const
{
  return mC[kSigTglSnp];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTgl2() const
{
  return mC[kSigTgl2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtY() const
{
  return mC[kSigQ2PtY];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtZ() const
{
  return mC[kSigQ2PtZ];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtSnp() const
{
  return mC[kSigQ2PtSnp];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtTgl() const
{
  return mC[kSigQ2PtTgl];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1Pt2() const
{
  return mC[kSigQ2Pt2];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getCovarElem(int i, int j) const
{
  return mC[CovarMap[i][j]];
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getDiagError2(int i) const
{
  return mC[DiagMap[i]];
}

//__________________________________________________________________________
template <typename value_T>
template <typename T>
GPUdi() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getPredictedChi2(const BaseCluster<T>& p) const
{
  const dim2_t pyz = {p.getY(), p.getZ()};
  const dim3_t cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
  return getPredictedChi2(pyz, cov);
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
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(value_t delta, int i)
{
  mC[i] += delta;
}

//__________________________________________________________________________
template <typename value_T>
GPUdi() void TrackParametrizationWithError<value_T>::updateCov(const value_t delta[kCovMatSize])
{
  for (int i = kCovMatSize; i--;) {
    mC[i] += delta[i];
  }
}

} // namespace track
} // namespace o2
#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_ */
