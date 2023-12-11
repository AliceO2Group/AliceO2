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

/// \file SACFactorization.h
/// \brief TPC factorization of SACs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jul 6, 2022

#ifndef ALICEO2_SACFACTORIZATION_H_
#define ALICEO2_SACFACTORIZATION_H_

#include <vector>
#include "Rtypes.h"
#include "TPCCalibration/IDCContainer.h"
#include "DataFormatsTPC/Defs.h"
#include <boost/property_tree/ptree.hpp>

namespace o2::tpc
{

class SACFactorization
{
 public:
  using SACDeltaCompression = IDCDeltaCompression;

  /// constructor
  /// \param timeFrames number of time frames which will be stored
  SACFactorization(const unsigned int timeFrames) : mTimeFrames{timeFrames} {};

  /// default constructor
  SACFactorization() = default;

  /// calculate I_0(r,\phi) = <I(r,\phi,t)>_t
  /// calculate I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  /// calculate \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  void factorizeSACs();

  /// calculate I_0(r,\phi) = <I(r,\phi,t)>_t
  void calcSACZero();

  /// calculate I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  void calcSACOne();

  /// calculate \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  void calcSACDelta();

  /// create outlier map using the SAC0
  void createStatusMap();

  /// \return returns the stored SAC value
  /// \param stack stack
  /// \param interval integration interval
  auto getSACValue(const unsigned int stack, const unsigned int interval) const { return mSACs[stack][interval]; }

  /// \return returns the stored SAC0 value
  /// \param stack stack
  float getSACZeroVal(const unsigned int stack) const { return mSACZero.getValueIDCZero(getSide(stack), stack % GEMSTACKSPERSIDE); }

  /// \return returns the stored DeltaSAC value
  /// \param stack stack
  /// \param interval integration interval
  float getSACDeltaVal(const unsigned int stack, unsigned int interval) const { return mSACDelta.getValue(getSide(stack), getSACDeltaIndex(stack, interval)); }

  /// \return returns SAC1 value
  /// \param Side TPC side
  /// \param interval integration interval
  float getSACOneVal(const Side side, unsigned int interval) const { return mSACOne.getValue(side, interval); }

  /// \return returns index for SAC delta
  /// \param stack stack
  /// \param interval local integration interval
  static unsigned int getSACDeltaIndex(const unsigned int stack, unsigned int interval) { return stack % GEMSTACKSPERSIDE + GEMSTACKSPERSIDE * interval; }

  /// \return returns number of timeframes for which the SACs are stored
  unsigned int getNTimeframes() const { return mTimeFrames; }

  /// \return returns the total number of stored integration intervals
  unsigned long getNintervals(const int stack = 0) const { return mSACs[stack].size(); }

  /// \return returns the total number of stored integration intervals
  unsigned long getNIntegrationIntervals(const int stack = 0) const { return mSACs[stack].size(); }

  /// \return returns stored SAC0 I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side TPC side
  const std::vector<float>& getSACZero(const o2::tpc::Side side) const { return mSACZero.mSACZero[side].mIDCZero; }

  /// \return returns stored SAC1 I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  /// \param side TPC side
  const std::vector<float>& getSACOne(const o2::tpc::Side side) const { return mSACOne.mSACOne[side].mIDCOne; }

  /// \return returns stored SACDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param side TPC side
  const std::vector<float>& getSACDeltaUncompressed(const o2::tpc::Side side) const { return mSACDelta.mSACDelta[side].getIDCDelta(); }

  /// \return returns stored SACDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  const auto& getSACDeltaUncompressed() const& { return mSACDelta; }

  /// \return returns returns stored SACDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) ) using move semantics
  auto getSACDeltaUncompressed() && { return std::move(mSACDelta); }

  /// \return creates and returns medium compressed SACDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  auto getSACDeltaMediumCompressed() const { return IDCDeltaCompressionHelper<unsigned short>::getCompressedSACs(mSACDelta); }

  /// \return creates and returns high compressed SACDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  auto getSACDeltaHighCompressed() const { return IDCDeltaCompressionHelper<unsigned char>::getCompressedSACs(mSACDelta); }

  /// \return returns struct containing SAC0
  const auto& getSACZero() const { return mSACZero; }

  /// \return returns struct containing SAC1
  const auto& getSACOne() const& { return mSACOne; }

  /// \return returns struct containing SAC1 using move semantics
  auto getSACOne() && { return std::move(mSACOne); }

  /// \return returns grouped SACs
  const auto& getSACs() const { return mSACs; }

  /// \return returns outliers obtained with SAC0
  const auto& getOutlierMap() const { return mOutlierMap; }

  /// \return returns the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the SAC data
  /// \param SACs vector containing the SACs
  /// \param cru CRU
  /// \param timeframe time frame of the SACs
  void setSACs(std::vector<int32_t>&& SACs, const unsigned int stack) { mSACs[stack] = std::move(SACs); }

  /// set the number of threads used for some of the calculations
  /// \param nThreads number of threads
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// \param minSACDeltaValue minimum SAC delta value for compressed SAC delta
  static void setMinCompressedSACDelta(const float minSACDeltaValue) { o2::conf::ConfigurableParam::setValue<float>("TPCSACCompressionParam", "minSACDeltaValue", minSACDeltaValue); }

  /// \param maxSACDeltaValue maximum SAC delta value for compressed SAC delta
  static void setMaxCompressedSACDelta(const float maxSACDeltaValue) { o2::conf::ConfigurableParam::setValue<float>("TPCSACCompressionParam", "maxSACDeltaValue", maxSACDeltaValue); }

  /// draw SACs for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param interval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACsSector(const unsigned int sector, const unsigned int interval, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACsSector.pdf") const { drawSACHelper(false, Sector(sector), interval, filename, minZ, maxZ); }

  /// draw SAC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroSector(const unsigned int sector, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACZeroSector.pdf") const { drawSACZeroHelper(false, Sector(sector), filename, minZ, maxZ); }

  /// draw outliers of SACs obtained with SAC0
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroOutlierSector(const unsigned int sector, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACZeroOutlierSector.pdf") const { drawSACZeroOutlierHelper(false, Sector(sector), filename, minZ, maxZ); }

  /// draw SACDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param interval which will be drawn
  /// \param compression compression of Delta SACs. (setMaxCompressedSACDelta() should be called first in case of non standard compression parameter)
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACDeltaSector(const unsigned int sector, const unsigned int interval, const float minZ = 0, const float maxZ = -1, const SACDeltaCompression compression = SACDeltaCompression::NO, const std::string filename = "SACDeltaSector.pdf") const { drawSACDeltaHelper(false, Sector(sector), interval, compression, filename, minZ, maxZ); }

  /// draw SACs for one side for one integration interval
  /// \param side side which will be drawn
  /// \param interval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACsSide(const o2::tpc::Side side, const unsigned int interval, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACsSide.pdf") const { drawSACHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), interval, filename, minZ, maxZ); }

  /// draw SAC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroSide(const o2::tpc::Side side, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACZeroSide.pdf") const { drawSACZeroHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw SACDelta for one sector for one integration interval
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACZeroOutlierSide(const o2::tpc::Side side, const float minZ = 0, const float maxZ = -1, const std::string filename = "SACZeroOutlierSide.pdf") const { drawSACZeroOutlierHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw SACDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param interval which will be drawn
  /// \param compression compression of Delta SACs. (setMaxCompressedSACDelta() should be called first in case of non standard compression parameter)
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSACDeltaSide(const o2::tpc::Side side, const unsigned int interval, const float minZ = 0, const float maxZ = -1, const SACDeltaCompression compression = SACDeltaCompression::NO, const std::string filename = "SACDeltaSide.pdf") const { drawSACDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), interval, compression, filename, minZ, maxZ); }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "SACFactorized.root", const char* outName = "SACFactorized") const;

  /// \param intervals number of integration intervals which will be dumped to the tree (-1: all integration intervals)
  /// \param outFileName name of the output file
  void dumpToTree(int intervals = -1, const char* outFileName = "SACTree.root") const;

  /// resetting aggregated SACs
  void reset();

  /// \return returns side for given GEM stack
  static Side getSide(const unsigned int gemStack) { return (gemStack < GEMSTACKSPERSIDE) ? Side::A : Side::C; }

  /// \return returns stack for given sector and stack
  static unsigned int getStack(const unsigned int sector, const unsigned int stack) { return static_cast<unsigned int>(stack + sector * GEMSTACKSPERSECTOR); }

  /// \return returns stack (starts at 0 for each side)
  static unsigned int getStackInSide(const unsigned int sector, const unsigned int stack) { return getStack(sector, stack) % (GEMSTACKS / 2); }

  void setSACZero(const SACZero& sacZero) { mSACZero = sacZero; }

 private:
  const unsigned int mTimeFrames{};                             ///< number of timeframes which are stored
  std::array<std::vector<int32_t>, o2::tpc::GEMSTACKS> mSACs{}; ///< SACs aggregated over mTimeFrames
  SACZero mSACZero{};                                           ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  SACOne mSACOne{};                                             ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  std::array<int, GEMSTACKS> mOutlierMap{};                     ///< map containing the outliers for the SAC0
  SACDelta<float> mSACDelta{};                                  ///< uncompressed: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  inline static int sNThreads{1};                               ///< number of threads which are used during the calculations

  /// helper function for drawing SACDelta
  void drawSACDeltaHelper(const bool type, const Sector sector, const unsigned int interval, const SACDeltaCompression compression, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing SACs
  void drawSACHelper(const bool type, const Sector sector, const unsigned int interval, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing SACZero
  void drawSACZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing SACZero
  void drawSACZeroOutlierHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;

  ClassDefNV(SACFactorization, 1)
};

} // namespace o2::tpc

#endif
