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
#include "GPUCommonRtypes.h"
#include <vector>

namespace o2::gpu
{

///
/// The TPCFastSpaceChargeCorrectionMap class represents a maps of corrections of nominal coordinates of TPC clusters
///
/// Row, U, V -> dX,dU,dV
///
/// It is used by TPCFastSpoaceChargeCorrectionHelper
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
    double mY{0.}, mZ{0.};            // not-distorted local coordinates
    double mDx{0.}, mDy{0.}, mDz{0.}; // corrections to the local coordinates
    double mWeight{0.};               // weight of the point
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrectionMap(int32_t nSectors, int32_t nRows)
  {
    init(nSectors, nRows);
  }

  /// Destructor
  ~TPCFastSpaceChargeCorrectionMap() = default;

  /// (re-)init the map
  void init(int32_t nSectors, int32_t nRows)
  {
    mNsectors = nSectors;
    mNrows = nRows;
    int32_t n = mNsectors * mNrows;
    fDataPoints.resize(n);
    for (uint32_t i = 0; i < fDataPoints.size(); ++i) {
      fDataPoints[i].clear();
    }
  }

  /// Starts the construction procedure, reserves temporary memory
  void addCorrectionPoint(int32_t iSector, int32_t iRow,
                          double y, double z,
                          double dx, double dy, double dz, double weight)
  {
    int32_t ind = mNrows * iSector + iRow;
    fDataPoints.at(ind).push_back(CorrectionPoint{y, z,
                                                  dx, dy, dz, weight});
  }

  const std::vector<CorrectionPoint>& getPoints(int32_t iSector, int32_t iRow) const
  {
    int32_t ind = mNrows * iSector + iRow;
    return fDataPoints.at(ind);
  }

  int32_t getNsectors() const { return mNsectors; }

  int32_t getNrows() const { return mNrows; }

  bool isInitialized() const { return mNsectors > 0 && mNrows > 0; }

 private:
  /// _______________  Data members  _______________________________________________
  int32_t mNsectors{0};
  int32_t mNrows{0};
  std::vector<std::vector<CorrectionPoint>> fDataPoints; //! (transient!!) points with space charge correction

  ClassDefNV(TPCFastSpaceChargeCorrectionMap, 0);
};

} // namespace o2::gpu

#endif
