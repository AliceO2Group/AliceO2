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

  static_assert(std::is_floating_point_v<value_t>);

 public:
  TrackParametrizationWithError();
  TrackParametrizationWithError(value_t x, value_t alpha, const std::array<value_t, kNParams>& par, const std::array<value_t, kCovMatSize>& cov, int charge = 1);
  TrackParametrizationWithError(const std::array<value_t, 3>& xyz, const std::array<value_t, 3>& pxpypz,
                                const std::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true);

  TrackParametrizationWithError(const TrackParametrizationWithError& src) = default;
  TrackParametrizationWithError(TrackParametrizationWithError&& src) = default;
  TrackParametrizationWithError& operator=(const TrackParametrizationWithError& src) = default;
  TrackParametrizationWithError& operator=(TrackParametrizationWithError&& src) = default;
  ~TrackParametrizationWithError() = default;

  const value_t* getCov() const;
  value_t getSigmaY2() const;
  value_t getSigmaZY() const;
  value_t getSigmaZ2() const;
  value_t getSigmaSnpY() const;
  value_t getSigmaSnpZ() const;
  value_t getSigmaSnp2() const;
  value_t getSigmaTglY() const;
  value_t getSigmaTglZ() const;
  value_t getSigmaTglSnp() const;
  value_t getSigmaTgl2() const;
  value_t getSigma1PtY() const;
  value_t getSigma1PtZ() const;
  value_t getSigma1PtSnp() const;
  value_t getSigma1PtTgl() const;
  value_t getSigma1Pt2() const;
  value_t getCovarElem(int i, int j) const;
  value_t getDiagError2(int i) const;

  bool getCovXYZPxPyPzGlo(std::array<value_t, 21>& c) const;

#ifndef GPUCA_ALIGPUCODE
  void print() const;
  std::string asString() const;
#endif

  // parameters + covmat manipulation
  bool rotate(value_t alpha);
  bool propagateTo(value_t xk, value_t b);
  bool propagateTo(value_t xk, const std::array<value_t, 3>& b);
  bool propagateToDCA(const o2::dataformats::VertexBase& vtx, value_t b, o2::dataformats::DCA* dca = nullptr, value_t maxD = 999.f);
  void invert();

  value_t getPredictedChi2(const std::array<value_t, 2>& p, const std::array<value_t, 3>& cov) const;

  template <typename T>
  value_t getPredictedChi2(const BaseCluster<T>& p) const;

  value_t getPredictedChi2(const TrackParametrizationWithError& rhs) const;

  void buildCombinedCovMatrix(const TrackParametrizationWithError& rhs, MatrixDSym5& cov) const;
  value_t getPredictedChi2(const TrackParametrizationWithError& rhs, MatrixDSym5& covToSet) const;
  bool update(const TrackParametrizationWithError& rhs, const MatrixDSym5& covInv);

  bool update(const std::array<value_t, 2>& p, const std::array<value_t, 3>& cov);

  template <typename T>
  bool update(const BaseCluster<T>& p);

  bool update(const TrackParametrizationWithError& rhs);

  bool correctForMaterial(value_t x2x0, value_t xrho, value_t mass, bool anglecorr = false, value_t dedx = kCalcdEdxAuto);

  void resetCovariance(value_t s2 = 0);
  void checkCovariance();
  void setCov(value_t v, int i);

 protected:
  void updateCov(const value_t delta[kCovMatSize]);

 protected:
  value_t mC[kCovMatSize] = {0.f}; // 15 covariance matrix elements

  ClassDefNV(TrackParametrizationWithError, 2);
};

//__________________________________________________________________________
template <typename value_T>
inline TrackParametrizationWithError<value_T>::TrackParametrizationWithError() : TrackParametrization<value_T>{}
{
}

//__________________________________________________________________________
template <typename value_T>
inline TrackParametrizationWithError<value_T>::TrackParametrizationWithError(value_t x, value_t alpha, const std::array<value_t, kNParams>& par,
                                                                             const std::array<value_t, kCovMatSize>& cov, int charge)
  : TrackParametrization<value_T>{x, alpha, par, charge}
{
  // explicit constructor
  std::copy(cov.begin(), cov.end(), mC);
}

//__________________________________________________________________________
template <typename value_T>
inline const typename TrackParametrizationWithError<value_T>::value_t* TrackParametrizationWithError<value_T>::getCov() const
{
  return mC;
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaY2() const
{
  return mC[kSigY2];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaZY() const
{
  return mC[kSigZY];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaZ2() const
{
  return mC[kSigZ2];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnpY() const
{
  return mC[kSigSnpY];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnpZ() const
{
  return mC[kSigSnpZ];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaSnp2() const
{
  return mC[kSigSnp2];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglY() const
{
  return mC[kSigTglY];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglZ() const
{
  return mC[kSigTglZ];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTglSnp() const
{
  return mC[kSigTglSnp];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigmaTgl2() const
{
  return mC[kSigTgl2];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtY() const
{
  return mC[kSigQ2PtY];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtZ() const
{
  return mC[kSigQ2PtZ];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtSnp() const
{
  return mC[kSigQ2PtSnp];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1PtTgl() const
{
  return mC[kSigQ2PtTgl];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getSigma1Pt2() const
{
  return mC[kSigQ2Pt2];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getCovarElem(int i, int j) const
{
  return mC[CovarMap[i][j]];
}

//__________________________________________________________________________
template <typename value_T>
inline typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getDiagError2(int i) const
{
  return mC[DiagMap[i]];
}

//__________________________________________________________________________
template <typename value_T>
template <typename T>
typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getPredictedChi2(const BaseCluster<T>& p) const
{
  const std::array<value_t, 2> pyz = {p.getY(), p.getZ()};
  const std::array<value_t, 3> cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
  return getPredictedChi2(pyz, cov);
}

//__________________________________________________________________________
template <typename value_T>
template <typename T>
bool TrackParametrizationWithError<value_T>::update(const BaseCluster<T>& p)
{
  const std::array<value_t, 2> pyz = {p.getY(), p.getZ()};
  const std::array<value_t, 3> cov = {p.getSigmaY2(), p.getSigmaYZ(), p.getSigmaZ2()};
  return update(pyz, cov);
}

//__________________________________________________________________________
template <typename value_T>
inline void TrackParametrizationWithError<value_T>::setCov(value_t v, int i)
{
  mC[i] = v;
}

//__________________________________________________________________________
template <typename value_T>
inline void TrackParametrizationWithError<value_T>::updateCov(const value_t delta[kCovMatSize])
{
  for (int i = kCovMatSize; i--;)
    mC[i] += delta[i];
}

} // namespace track
} // namespace o2
#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_ */
