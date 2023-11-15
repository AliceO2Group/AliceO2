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

/// \file SpacePointsCalibParam.h
/// \brief Parameters used for TPC space point calibration
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_TPC_PARAM_H_
#define ALICEO2_TPC_PARAM_H_

#include "DataFormatsTPC/Constants.h"

namespace o2
{
namespace tpc
{
namespace param
{

/// TPC geometric constants for Run 3+
static constexpr int NPadRows = o2::tpc::constants::MAXGLOBALPADROW;
static constexpr int NROCTypes = 4;
static constexpr int NRowsPerROC[NROCTypes] = {63, 34, 30, 25};
static constexpr int NRowsAccumulated[NROCTypes] = {63, 97, 127, 152};
static constexpr float ZLimit[2] = {2.49725e2f, 2.49698e2f}; ///< max z-positions for A/C side
static constexpr float RowDX[NROCTypes] = {.75f, 1.f, 1.2f, 1.5f};
static constexpr float MinX = 84.85f;
static constexpr float MaxX = 246.4f;
static constexpr float RowX[NPadRows] = {
  85.225, 85.975, 86.725, 87.475, 88.225, 88.975, 89.725, 90.475, 91.225, 91.975, 92.725, 93.475, 94.225, 94.975, 95.725, 96.475,
  97.225, 97.975, 98.725, 99.475, 100.225, 100.975, 101.725, 102.475, 103.225, 103.975, 104.725, 105.475, 106.225, 106.975, 107.725,
  108.475, 109.225, 109.975, 110.725, 111.475, 112.225, 112.975, 113.725, 114.475, 115.225, 115.975, 116.725, 117.475, 118.225, 118.975,
  119.725, 120.475, 121.225, 121.975, 122.725, 123.475, 124.225, 124.975, 125.725, 126.475, 127.225, 127.975, 128.725, 129.475, 130.225,
  130.975, 131.725, 135.200, 136.200, 137.200, 138.200, 139.200, 140.200, 141.200, 142.200, 143.200, 144.200, 145.200, 146.200, 147.200,
  148.200, 149.200, 150.200, 151.200, 152.200, 153.200, 154.200, 155.200, 156.200, 157.200, 158.200, 159.200, 160.200, 161.200, 162.200,
  163.200, 164.200, 165.200, 166.200, 167.200, 168.200, 171.400, 172.600, 173.800, 175.000, 176.200, 177.400, 178.600, 179.800, 181.000,
  182.200, 183.400, 184.600, 185.800, 187.000, 188.200, 189.400, 190.600, 191.800, 193.000, 194.200, 195.400, 196.600, 197.800, 199.000,
  200.200, 201.400, 202.600, 203.800, 205.000, 206.200, 209.650, 211.150, 212.650, 214.150, 215.650, 217.150, 218.650, 220.150, 221.650,
  223.150, 224.650, 226.150, 227.650, 229.150, 230.650, 232.150, 233.650, 235.150, 236.650, 238.150, 239.650, 241.150, 242.650, 244.150,
  245.650};

// TPC voxel binning
static constexpr int NY2XBins = 15; ///< number of bins in y/x
static constexpr int NZ2XBins = 5;  ///< number of bins in z/x

// define ranges for compression to shorts in TPCClusterResiduals
static constexpr float MaxResid = 20.f; ///< max residual in y and z
static constexpr float MaxY = 50.f;     ///< max value for y position (sector coordinates)
static constexpr float MaxZ = 300.f;    ///< max value for z position
static constexpr float MaxTgSlp = 1.f;  ///< max value for phi and lambda angles

// miscellaneous
static constexpr float sEps = 1e-6f; ///< small number for float comparisons

} // namespace param
} // namespace tpc
} // namespace o2
#endif
