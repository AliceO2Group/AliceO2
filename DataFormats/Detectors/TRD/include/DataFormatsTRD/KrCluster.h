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

/// \file KrCluster.h
/// \brief A cluster formed from digits during TRD Krypton calibration

#ifndef O2_TRD_KRCLUSTER_H
#define O2_TRD_KRCLUSTER_H

#include "Rtypes.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

namespace o2
{
namespace trd
{

class KrCluster
{
 public:
  KrCluster() = default;
  KrCluster(const KrCluster&) = default;
  ~KrCluster() = default;

  void setGlobalPadID(int det, int row, int col);
  void setAdcData(int adcSum, int adcRms, int adcMaxA, int adcMaxB, int adcEoT, int adcIntegral, int adcSumTrunc);
  void setTimeData(int timeMaxA, int timeMaxB, int timeRms);
  void setClusterSizeData(int rowSize, int colSize, int timeSize, int nAdcs);

  // identify global pad number (row and column with the maximum ADC value contained in the cluster)
  int getDetector() const { return mDet; }
  int getSector() const { return HelperMethods::getSector(mDet); }
  int getStack() const { return HelperMethods::getStack(mDet); }
  int getLayer() const { return HelperMethods::getLayer(mDet); }
  int getRow() const { return mRow; }
  int getColumn() const { return mCol; }

  // ADC related members
  int getAdcSum() const { return mAdcSum; }
  int getAdcRms() const { return mAdcRms; }
  int getAdcMaxA() const { return mAdcMaxA; }
  int getAdcMaxB() const { return mAdcMaxB; }
  int getAdcSumEoverT() const { return mAdcSumEoverT; }
  int getAdcIntegral() const { return mAdcIntegral; }
  int getAdcSumTruncated() const { return mAdcSumTruncated; }

  // cluster size
  int getClSizeRow() const { return mDeltaRow; }
  int getClSizeCol() const { return mDeltaCol; }
  int getClSizeTime() const { return mDeltaTime; }
  int getClSize() const { return mClusterSize; }

  // time bin related members
  int getTimeMaxA() const { return mTimeMaxA; }
  int getTimeMaxB() const { return mTimeMaxB; }
  int getTimeRms() const { return mTimeRms; }

 private:
  uint16_t mDet;             ///< chamber number [0..539]
  uint16_t mAdcSum;          ///< sum of all ADCs contributing to this cluster (baseline subtracted)
  uint16_t mAdcRms;          ///< RMS of the ADCs constributing to this cluster
  uint16_t mAdcMaxA;         ///< sum of the ADCs of the first maximum
  uint16_t mAdcMaxB;         ///< sum of the ADCs of the second maximum (the contribution from the first maximum is subtracted here)
  uint16_t mAdcSumEoverT;    ///< same as mAdcSum, but ADCs below KrClusterFinder::mMinAdcClEoverT are ignored
  uint16_t mAdcIntegral;     ///< integral of the Landau fit for all time bins
  uint16_t mAdcSumTruncated; ///< same as mAdcSum, but only ADCs close to the RMS are counted
  uint8_t mRow;              ///< pad row number contributing max ADC value contained in this cluster [0..15]
  uint8_t mCol;              ///< pad column number contributing max ADC value contained in this cluster [0..143]
  uint8_t mDeltaRow;         ///< cluster size in row direction
  uint8_t mDeltaCol;         ///< cluster size in column direction
  uint8_t mDeltaTime;        ///< cluster size in time bin direction
  uint8_t mClusterSize;      ///< number of ADC values contributing to the cluster (ADC needs to be above KrClusterFinder::mMinAdcClContrib)
  uint8_t mTimeMaxA;         ///< time bin of the first maximum
  uint8_t mTimeMaxB;         ///< time bin of the second maximum
  uint8_t mTimeRms;          ///< RMS of the time bins for ADCs contributing to this cluster

  ClassDefNV(KrCluster, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_KRCLUSTER_H
