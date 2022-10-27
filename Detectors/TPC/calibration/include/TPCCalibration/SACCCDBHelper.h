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

/// \file SACCCDBHelper.h
/// \brief helper class for accessing SACs from CCDB
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_SACCCDBHELPER_H_
#define ALICEO2_TPC_SACCCDBHELPER_H_

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"

namespace o2::tpc
{

struct SACZero;
struct SACOne;
struct FourierCoeffSAC;
template <typename DataT>
struct SACDelta;

/// \tparam DataT the data type for the SACDelta which are stored in the CCDB (unsigned short, unsigned char, float)
template <typename DataT = unsigned short>
class SACCCDBHelper
{
 public:
  /// constructor
  SACCCDBHelper() = default;

  /// setting the SACDelta class member
  void setSACDelta(SACDelta<DataT>* SACDelta) { mSACDelta = SACDelta; }

  /// setting the 0D-SACs
  void setSACZero(SACZero* SACZero) { mSACZero = SACZero; }

  /// setting the 1D-SACs
  void setSACOne(SACOne* SACOne) { mSACOne = SACOne; }

  /// setting the fourier coefficients
  void setFourierCoeffs(FourierCoeffSAC* fourier) { mFourierCoeff = fourier; }

  /// \return returns the number of integration intervals for SACDelta
  unsigned int getNIntegrationIntervalsSACDelta(const o2::tpc::Side side) const;

  /// \return returns the number of integration intervals for SACOne
  unsigned int getNIntegrationIntervalsSACOne(const o2::tpc::Side side) const;

  /// \return returns the stored SAC0 value
  /// \param sector sector
  /// \param stack local stack in sector
  float getSACZeroVal(const unsigned int sector, const unsigned int stack) const;

  /// \return returns the stored DeltaSAC value
  /// \param sector sector
  /// \param stack local stack in sector
  /// \param integrationInterval integration interval
  float getSACDeltaVal(const unsigned int sector, const unsigned int stack, unsigned int integrationInterval) const;

  /// \return returns SACOne value
  /// \param side side of the TPC
  /// \param integrationInterval integration interval
  float getSACOneVal(const o2::tpc::Side side, const unsigned int integrationInterval) const;

  /// \return returns the SAC value which is calculated with: (SACDelta + 1) * SACOne * SACZero
  /// \param sector sector
  /// \param stack local stack in sector
  /// \param integrationInterval integration interval
  float getSACVal(const unsigned int sector, const unsigned int stack, unsigned int integrationInterval) const;

  /// draw SAC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroSide(const o2::tpc::Side side, const std::string filename = "SACZeroSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACZeroHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw SACDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACDeltaSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "SACDeltaSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename, minZ, maxZ); }

  /// draw SACs which is calculated with: (SACDelta + 1) * SACOne * SACZero
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "SACSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename, minZ, maxZ); }

  /// draw SAC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroSector(const unsigned int sector, const std::string filename = "SACZeroSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACZeroHelper(false, Sector(sector), filename, minZ, maxZ); }

  /// draw SACDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACDeltaSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "SACDeltaSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACDeltaHelper(false, Sector(sector), integrationInterval, filename, minZ, maxZ); }

  /// draw SAC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "SACSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawSACHelper(false, Sector(sector), integrationInterval, filename, minZ, maxZ); }

  /// dumping the loaded SAC0, SAC1 to a tree
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "SACCCDBTree.root") const;

  /// dumping the loaded fourier coefficients to a tree
  /// \param outFileName name of the output file
  void dumpToFourierCoeffToTree(const char* outFileName = "FourierCCDBTree.root") const;

 private:
  SACZero* mSACZero = nullptr;              ///< 0D-SACs: ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  SACDelta<DataT>* mSACDelta = nullptr;     ///< compressed or uncompressed Delta SAC: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  SACOne* mSACOne = nullptr;                ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  FourierCoeffSAC* mFourierCoeff = nullptr; ///< fourier coefficients of SACOne

  /// helper function for drawing SACZero
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing SACDelta
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing SAC
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const;
};

} // namespace o2::tpc

#endif
