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

/// \file KrCluster.cxx
/// \brief A cluster formed from digits during TRD Krypton calibration

#include "DataFormatsTRD/KrCluster.h"

using namespace o2::trd;

void KrCluster::setGlobalPadID(int det, int row, int col)
{
  mDet = det;
  mRow = row;
  mCol = col;
}

void KrCluster::setAdcData(int adcSum, int adcRms, int adcMaxA, int adcMaxB, int adcEoT, int adcIntegral, int adcSumTrunc)
{
  mAdcSum = adcSum;
  mAdcRms = adcRms;
  mAdcMaxA = adcMaxA;
  mAdcMaxB = adcMaxB;
  mAdcSumEoverT = adcEoT;
  mAdcIntegral = adcIntegral;
  mAdcSumTruncated = adcSumTrunc;
}

void KrCluster::setTimeData(int timeMaxA, int timeMaxB, int timeRms)
{
  mTimeMaxA = timeMaxA;
  mTimeMaxB = timeMaxB;
  mTimeRms = timeRms;
}

void KrCluster::setClusterSizeData(int rowSize, int colSize, int timeSize, int nAdcs)
{
  mDeltaRow = rowSize;
  mDeltaCol = colSize;
  mDeltaTime = timeSize;
  mClusterSize = nAdcs;
}
