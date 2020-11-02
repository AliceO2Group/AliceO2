// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackMCH.h
/// \brief Definition of the MCH track
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_TRACKMCH_H_
#define ALICEO2_MCH_TRACKMCH_H_

#include <TMatrixD.h>

#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace mch
{

/// MCH track external format
class TrackMCH
{
  using ClusRef = o2::dataformats::RangeRefComp<5>;

 public:
  TrackMCH() = default;
  TrackMCH(double z, const TMatrixD& param, const TMatrixD& cov, double chi2, int firstClIdx, int nClusters);
  ~TrackMCH() = default;

  TrackMCH(const TrackMCH& track) = default;
  TrackMCH& operator=(const TrackMCH& track) = default;
  TrackMCH(TrackMCH&&) = default;
  TrackMCH& operator=(TrackMCH&&) = default;

  /// get the track x position
  double getX() const { return mParam[0]; }
  /// get the track y position
  double getY() const { return mParam[2]; }
  /// get the track z position where the parameters are evaluated
  double getZ() const { return mZ; }
  /// set the track z position where the parameters are evaluated
  void setZ(double z) { mZ = z; }

  double getPx() const;
  double getPy() const;
  double getPz() const;
  double getP() const;

  /// get the muon sign
  short getSign() const { return (mParam[4] < 0) ? -1 : 1; }

  /// get the track parameters
  const double* getParameters() const { return mParam; }
  /// set the track parameters
  void setParameters(const TMatrixD& param) { param.GetMatrix2Array(mParam); }

  /// get the track parameter covariances
  const double* getCovariances() const { return mCov; }
  /// get the covariance between track parameters i and j
  double getCovariance(int i, int j) const { return mCov[SCovIdx[i][j]]; }
  // set the track parameter covariances
  void setCovariances(const TMatrixD& cov);

  /// get the track chi2
  double getChi2() const { return mChi2; }
  /// set the track chi2
  void setChi2(double chi2) { mChi2 = chi2; }
  /// get the number of degrees of freedom of the track
  int getNDF() const { return 2 * mClusRef.getEntries() - 5; }
  /// get the track normalized chi2
  double getChi2OverNDF() const { return mChi2 / getNDF(); }

  /// get the number of clusters attached to the track
  int getNClusters() const { return mClusRef.getEntries(); }
  /// get the index of the first cluster attached to the track
  int getFirstClusterIdx() const { return mClusRef.getFirstEntry(); }
  /// get the index of the last cluster attached to the track
  int getLastClusterIdx() const { return mClusRef.getFirstEntry() + mClusRef.getEntries() - 1; }
  /// set the number of the clusters attached to the track and the index of the first one
  void setClusterRef(int firstClusterIdx, int nClusters) { mClusRef.set(firstClusterIdx, nClusters); }

 private:
  static constexpr int SNParams = 5;  ///< number of track parameters
  static constexpr int SCovSize = 15; ///< number of different elements in the symmetric covariance matrix
  /// corresponding indices to access the covariance matrix elements by row and column
  static constexpr int SCovIdx[SNParams][SNParams] = {{0, 1, 3, 6, 10},
                                                      {1, 2, 4, 7, 11},
                                                      {3, 4, 5, 8, 12},
                                                      {6, 7, 8, 9, 13},
                                                      {10, 11, 12, 13, 14}};

  double mZ = 0.;                 ///< z position where the parameters are evaluated
  double mParam[SNParams] = {0.}; ///< 5 parameters: X (cm), SlopeX, Y (cm), SlopeY, q/pYZ ((GeV/c)^-1)
  /// reduced covariance matrix of track parameters, formated as follow: <pre>
  /// [0] = <X,X>
  /// [1] = <SlopeX,X>  [2] = <SlopeX,SlopeX>
  /// [3] = <Y,X>       [4] = <Y,SlopeX>       [5] = <Y,Y>
  /// [6] = <SlopeY,X>  [7] = <SlopeY,SlopeX>  [8] = <SlopeY,Y>  [9] = <SlopeY,SlopeY>
  /// [10]= <q/pYZ,X>   [11]= <q/pYZ,SlopeX>   [12]= <q/pYZ,Y>   [13]= <q/pYZ,SlopeY>   [14]= <q/pYZ,q/pYZ> </pre>
  double mCov[SCovSize] = {0.};
  double mChi2 = 0.;  ///< chi2 of track
  ClusRef mClusRef{}; ///< reference to external cluster indices

  ClassDefNV(TrackMCH, 1);
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_TRACKMCH_H_
