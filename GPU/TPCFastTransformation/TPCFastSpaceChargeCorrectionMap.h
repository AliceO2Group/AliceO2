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

/// \file  TPCFastSpaceChargeCorrectionMap.h
/// \brief Definition of TPCFastSpaceChargeCorrectionMap class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTIONMAP_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTIONMAP_H

#include "GPUCommonDef.h"
#include <vector>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCFastSpaceChargeCorrectionMap class represents correction of nominal coordinates of TPC clusters
/// using best-fit splines
///
/// Row, U, V -> dX,dU,dV
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
///
class TPCFastSpaceChargeCorrectionMap
{
 public:
  ///
  /// \brief The struct contains necessary info for TPC padrow
  ///
  struct CorrectionPoint {
    double mY, mZ;        // not-distorted local coordinates
    double mDx, mDy, mDz; // corrections to the local coordinates
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrectionMap(int nRocs, int nRows)
  {
    init(nRocs, nRows);
  }

  /// Destructor
  ~TPCFastSpaceChargeCorrectionMap() = default;

  /// (re-)init the map
  void init(int nRocs, int nRows)
  {
    mNrocs = nRocs;
    mNrows = nRows;
    int n = mNrocs * mNrows;
    fDataPoints.resize(n);
    for (unsigned int i = 0; i < fDataPoints.size(); ++i) {
      fDataPoints[i].clear();
    }
  }

  /// Starts the construction procedure, reserves temporary memory
  void addCorrectionPoint(int iRoc, int iRow,
                          double y, double z,
                          double dx, double dy, double dz)
  {
    int ind = mNrows * iRoc + iRow;
    fDataPoints.at(ind).push_back(CorrectionPoint{y, z,
                                                  dx, dy, dz});
  }

  const std::vector<CorrectionPoint>& getPoints(int iRoc, int iRow) const
  {
    int ind = mNrows * iRoc + iRow;
    return fDataPoints.at(ind);
  }

  int getNrocs() const { return mNrocs; }

  int getNrows() const { return mNrows; }

  bool isInitialized() const { return mNrocs > 0 && mNrows > 0; }

 private:
  /// _______________  Data members  _______________________________________________
  int mNrocs{0};
  int mNrows{0};
  std::vector<std::vector<CorrectionPoint>> fDataPoints; //! (transient!!) points with space charge correction
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
