// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Param.h
/// \brief Parameters used for TPC space point calibration
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_CALIB_PARAM_H_
#define ALICEO2_CALIB_PARAM_H_

namespace o2
{
namespace calib
{

namespace param
{
// voxel definition
enum { VoxZ,
       VoxF,
       VoxX,
       VoxV };
enum { ResX,
       ResY,
       ResZ,
       ResD };

enum { DistDone = 1,
       DispDone = 2,
       SmoothDone = 4,
       Masked = 8 };

// TPC geometric constants
static constexpr int NSectors = 18;
static constexpr int NSectors2 = 2 * NSectors;
static constexpr int NRoc = 4 * NSectors;
static constexpr int NPadRows = 159;
static constexpr int NRowIROC = 63;
static constexpr int NRowOROC1 = 64;
static constexpr int NRowOROC2 = 32;
static constexpr float MinX = 85.f;
static constexpr float MaxX = 246.f;
static constexpr float RowX[NPadRows] = {
  85.225, 85.975, 86.725, 87.475, 88.225, 88.975, 89.725, 90.475, 91.225, 91.975, 92.725, 93.475, 94.225, 94.975, 95.725,
  96.475, 97.225, 97.975, 98.725, 99.475, 100.225, 100.975, 101.725, 102.475, 103.225, 103.975, 104.725, 105.475, 106.225, 106.975,
  107.725, 108.475, 109.225, 109.975, 110.725, 111.475, 112.225, 112.975, 113.725, 114.475, 115.225, 115.975, 116.725, 117.475, 118.225,
  118.975, 119.725, 120.475, 121.225, 121.975, 122.725, 123.475, 124.225, 124.975, 125.725, 126.475, 127.225, 127.975, 128.725, 129.475,
  130.225, 130.975, 131.725, 135.100, 136.100, 137.100, 138.100, 139.100, 140.100, 141.100, 142.100, 143.100, 144.100, 145.100, 146.100,
  147.100, 148.100, 149.100, 150.100, 151.100, 152.100, 153.100, 154.100, 155.100, 156.100, 157.100, 158.100, 159.100, 160.100, 161.100,
  162.100, 163.100, 164.100, 165.100, 166.100, 167.100, 168.100, 169.100, 170.100, 171.100, 172.100, 173.100, 174.100, 175.100, 176.100,
  177.100, 178.100, 179.100, 180.100, 181.100, 182.100, 183.100, 184.100, 185.100, 186.100, 187.100, 188.100, 189.100, 190.100, 191.100,
  192.100, 193.100, 194.100, 195.100, 196.100, 197.100, 198.100, 199.350, 200.850, 202.350, 203.850, 205.350, 206.850, 208.350, 209.850,
  211.350, 212.850, 214.350, 215.850, 217.350, 218.850, 220.350, 221.850, 223.350, 224.850, 226.350, 227.850, 229.350, 230.850, 232.350,
  233.850, 235.350, 236.850, 238.350, 239.850, 241.350, 242.850, 244.350, 245.850
};
static constexpr float RowDX[NPadRows] = { // distance between pad rows
  0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750,
  0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750,
  0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750,
  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
  1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,
  1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500
};
static constexpr float SecDPhi = 20.f * 0.01745; // 20.f * pi / 180.f
static constexpr float MaxY2X = 0.176;           // TMath::Tan(0.5f * SecDPhi);
static constexpr float DeadZone = 1.5f;
static constexpr float MaxZ2X = 1.f;

// for internal data structures
static constexpr int ResDim = 4;  // there are 4 dimensions for the results (X-distortions, Y-distortions, Z-distortions and dispersions)
static constexpr int VoxDim = 3;  // the voxels are defined in a 3 dimensional system
static constexpr int VoxHDim = 4; // for the smoothing we add for each voxel next to the distance for each dimension also the kernel weight
static constexpr float MaxResid = 20.f;

// smoothing parameters
enum { EpanechnikovKernel,
       GaussianKernel };

static constexpr int SmtLinDim = 4; // max matrix size for smoothing (pol1)
static constexpr int MaxSmtDim = 7; // max matrix size for smoothing (pol2)

// binning
static constexpr int Y2XBins = 15;
static constexpr int Z2XBins = 5;

// miscellaneous
static constexpr float FloatEps = 1.e-7f;

} // namespace param

} // namespace calib

} // namespace o2
#endif
