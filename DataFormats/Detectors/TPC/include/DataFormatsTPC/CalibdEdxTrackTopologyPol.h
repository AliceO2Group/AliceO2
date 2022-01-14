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

/// \file CalibdEdxTrackTopologyPol.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_CalibdEdxTrackTopologyPol_H_
#define ALICEO2_TPC_CalibdEdxTrackTopologyPol_H_

#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"
#ifndef GPUCA_ALIGPUCODE
#include <string_view>
#endif

// o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2::tpc
{

class CalibdEdxTrackTopologyPol
{
 public:
#if !defined(GPUCA_GPUCODE)
  CalibdEdxTrackTopologyPol()
  {
    clear();
  }
  CalibdEdxTrackTopologyPol(std::string_view fileName) { loadFromFile(fileName); }
#else
  CalibdEdxTrackTopologyPol() CON_DEFAULT;
#endif
  ~CalibdEdxTrackTopologyPol() CON_DEFAULT;

  /// \return returns the track topology correction
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  /// \param tanTheta tan of local inclination angle theta
  /// \param sinPhi track parameter sinphi
  /// \param z z position of the cluster
  /// \param relPad absolute relative pad position of the track
  /// \param relTime relative time position of the track
  GPUd() float getCorrection(const int region, const ChargeType charge, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime) const
  {
    const auto& param = mParams[regionIndex(region, charge)];
    const float x[FXDim]{tanTheta, sinPhi, z, relPad, relTime};
    const float corr = evalPol4_5D(x, param);
    return corr;
  }

  /// returns the maximum tanTheta for which the splines are valid
  GPUd() float getMaxTanTheta() const { return mMaxTanTheta; };

  /// returns the maximum sinPhi for which the splines are valid
  GPUd() float getMaxSinPhi() const { return mMaxSinPhi; };

#if !defined(GPUCA_GPUCODE)
  /// \return returns number of dimensions of the polynomial
  int getDims() const { return FXDim; }

  /// set the parameters for the polynomials
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  /// \param params parameter for the coefficients
  void setParams(const int region, const ChargeType charge, const float* params) { std::copy(params, params + FParams, mParams[regionIndex(region, charge)]); }

  /// \return returns the paramaters of the coefficients
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  const float* getParams(const int region, const ChargeType charge) const { return mParams[regionIndex(region, charge)]; }

  /// resetting the parameter
  void clear();

  /// set the maximum tanTheta for which the splines are valid
  /// \param maxTanTheta maximum tanTheta
  void setMaxTanTheta(const float maxTanTheta) { mMaxTanTheta = maxTanTheta; };

  /// set the maximum sinPhi for which the splines are valid
  /// \param maxSinPhi maximum sinPhi
  void setMaxSinPhi(const float maxSinPhi) { mMaxSinPhi = maxSinPhi; };

  /// dump the object to a file
  /// \param fileName name of the output file
  void saveFile(std::string_view fileName) const;

  /// load an object from a file
  /// \param fileName name of the file
  void loadFromFile(std::string_view fileName);
#endif

 private:
  /// \return returns the index for the stored parameters
  GPUd() static size_t regionIndex(const int region, const ChargeType charge) { return static_cast<size_t>(region + charge * 10); }

  /// evaluate the polynyomial for given coordinates and parameters
  GPUd() static constexpr float evalPol4_5D(const float* x, const float* param)
  {
    return param[0] * 1 + param[1] * x[0] + param[2] * x[1] + param[3] * x[2] + param[4] * x[3] + param[5] * x[4] + param[6] * x[0] * x[0] + param[7] * x[0] * x[1] + param[8] * x[0] * x[2] + param[9] * x[0] * x[3] + param[10] * x[0] * x[4] + param[11] * x[1] * x[1] + param[12] * x[1] * x[2] + param[13] * x[1] * x[3] + param[14] * x[1] * x[4] + param[15] * x[2] * x[2] + param[16] * x[2] * x[3] + param[17] * x[2] * x[4] + param[18] * x[3] * x[3] + param[19] * x[3] * x[4] + param[20] * x[4] * x[4] + param[21] * x[0] * x[0] * x[0] + param[22] * x[0] * x[0] * x[1] + param[23] * x[0] * x[0] * x[2] + param[24] * x[0] * x[0] * x[3] + param[25] * x[0] * x[0] * x[4] + param[26] * x[0] * x[1] * x[1] + param[27] * x[0] * x[1] * x[2] + param[28] * x[0] * x[1] * x[3] + param[29] * x[0] * x[1] * x[4] + param[30] * x[0] * x[2] * x[2] + param[31] * x[0] * x[2] * x[3] + param[32] * x[0] * x[2] * x[4] + param[33] * x[0] * x[3] * x[3] + param[34] * x[0] * x[3] * x[4] + param[35] * x[0] * x[4] * x[4] + param[36] * x[1] * x[1] * x[1] + param[37] * x[1] * x[1] * x[2] + param[38] * x[1] * x[1] * x[3] + param[39] * x[1] * x[1] * x[4] + param[40] * x[1] * x[2] * x[2] + param[41] * x[1] * x[2] * x[3] + param[42] * x[1] * x[2] * x[4] + param[43] * x[1] * x[3] * x[3] + param[44] * x[1] * x[3] * x[4] + param[45] * x[1] * x[4] * x[4] + param[46] * x[2] * x[2] * x[2] + param[47] * x[2] * x[2] * x[3] + param[48] * x[2] * x[2] * x[4] + param[49] * x[2] * x[3] * x[3] + param[50] * x[2] * x[3] * x[4] + param[51] * x[2] * x[4] * x[4] + param[52] * x[3] * x[3] * x[3] + param[53] * x[3] * x[3] * x[4] + param[54] * x[3] * x[4] * x[4] + param[55] * x[4] * x[4] * x[4] + param[56] * x[0] * x[0] * x[0] * x[0] + param[57] * x[0] * x[0] * x[0] * x[1] + param[58] * x[0] * x[0] * x[0] * x[2] + param[59] * x[0] * x[0] * x[0] * x[3] + param[60] * x[0] * x[0] * x[0] * x[4] + param[61] * x[0] * x[0] * x[1] * x[1] + param[62] * x[0] * x[0] * x[1] * x[2] + param[63] * x[0] * x[0] * x[1] * x[3] + param[64] * x[0] * x[0] * x[1] * x[4] + param[65] * x[0] * x[0] * x[2] * x[2] + param[66] * x[0] * x[0] * x[2] * x[3] + param[67] * x[0] * x[0] * x[2] * x[4] + param[68] * x[0] * x[0] * x[3] * x[3] + param[69] * x[0] * x[0] * x[3] * x[4] + param[70] * x[0] * x[0] * x[4] * x[4] + param[71] * x[0] * x[1] * x[1] * x[1] + param[72] * x[0] * x[1] * x[1] * x[2] + param[73] * x[0] * x[1] * x[1] * x[3] + param[74] * x[0] * x[1] * x[1] * x[4] + param[75] * x[0] * x[1] * x[2] * x[2] + param[76] * x[0] * x[1] * x[2] * x[3] + param[77] * x[0] * x[1] * x[2] * x[4] + param[78] * x[0] * x[1] * x[3] * x[3] + param[79] * x[0] * x[1] * x[3] * x[4] + param[80] * x[0] * x[1] * x[4] * x[4] + param[81] * x[0] * x[2] * x[2] * x[2] + param[82] * x[0] * x[2] * x[2] * x[3] + param[83] * x[0] * x[2] * x[2] * x[4] + param[84] * x[0] * x[2] * x[3] * x[3] + param[85] * x[0] * x[2] * x[3] * x[4] + param[86] * x[0] * x[2] * x[4] * x[4] + param[87] * x[0] * x[3] * x[3] * x[3] + param[88] * x[0] * x[3] * x[3] * x[4] + param[89] * x[0] * x[3] * x[4] * x[4] + param[90] * x[0] * x[4] * x[4] * x[4] + param[91] * x[1] * x[1] * x[1] * x[1] + param[92] * x[1] * x[1] * x[1] * x[2] + param[93] * x[1] * x[1] * x[1] * x[3] + param[94] * x[1] * x[1] * x[1] * x[4] + param[95] * x[1] * x[1] * x[2] * x[2] + param[96] * x[1] * x[1] * x[2] * x[3] + param[97] * x[1] * x[1] * x[2] * x[4] + param[98] * x[1] * x[1] * x[3] * x[3] + param[99] * x[1] * x[1] * x[3] * x[4] + param[100] * x[1] * x[1] * x[4] * x[4] + param[101] * x[1] * x[2] * x[2] * x[2] + param[102] * x[1] * x[2] * x[2] * x[3] + param[103] * x[1] * x[2] * x[2] * x[4] + param[104] * x[1] * x[2] * x[3] * x[3] + param[105] * x[1] * x[2] * x[3] * x[4] + param[106] * x[1] * x[2] * x[4] * x[4] + param[107] * x[1] * x[3] * x[3] * x[3] + param[108] * x[1] * x[3] * x[3] * x[4] + param[109] * x[1] * x[3] * x[4] * x[4] + param[110] * x[1] * x[4] * x[4] * x[4] + param[111] * x[2] * x[2] * x[2] * x[2] + param[112] * x[2] * x[2] * x[2] * x[3] + param[113] * x[2] * x[2] * x[2] * x[4] + param[114] * x[2] * x[2] * x[3] * x[3] + param[115] * x[2] * x[2] * x[3] * x[4] + param[116] * x[2] * x[2] * x[4] * x[4] + param[117] * x[2] * x[3] * x[3] * x[3] + param[118] * x[2] * x[3] * x[3] * x[4] + param[119] * x[2] * x[3] * x[4] * x[4] + param[120] * x[2] * x[4] * x[4] * x[4] + param[121] * x[3] * x[3] * x[3] * x[3] + param[122] * x[3] * x[3] * x[3] * x[4] + param[123] * x[3] * x[3] * x[4] * x[4] + param[124] * x[3] * x[4] * x[4] * x[4] + param[125] * x[4] * x[4] * x[4] * x[4];
  }

  static constexpr unsigned short FXDim{5}; ///< number of dimensionality of the polynomial
  constexpr static int FParams{126};        ///< number of parameters per polynomial
  constexpr static int FFits{20};           ///< total number of fits: 10 regions * 2 charge types
  float mParams[FFits][FParams];            ///< paramters of the polynomial
  float mMaxTanTheta{2.f};                  ///< max tanTheta for which the correction is stored
  float mMaxSinPhi{0.99f};                  ///< max snp for which the correction is stored
};

} // namespace o2::tpc

#endif
