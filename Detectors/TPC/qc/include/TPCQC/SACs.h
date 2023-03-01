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

///
/// @file   SACs.h
/// @author
///

#ifndef AliceO2_TPC_SACS_H
#define AliceO2_TPC_SACS_H

// root includes
#include "TCanvas.h"

// o2 includes
#include "TPCCalibration/IDCContainer.h"
#include "DataFormatsTPC/Defs.h"

namespace o2::tpc::qc
{

/// Keep QC information for SAC related observables
///
class SACs
{
 public:
  SACs() = default;

  /// \return returns the stored SAC value
  /// \param stack stack
  /// \param interval integration interval
  auto getSACValue(const unsigned int stack, const unsigned int interval) const { return mSACs[stack][interval]; }

  /// \return returns the stored SAC0 value
  /// \param stack stack
  float getSACZeroVal(const unsigned int stack) const { return mSACZero->getValueIDCZero(getSide(stack), stack % GEMSTACKSPERSIDE); }

  /// \return returns SAC1 value
  /// \param Side TPC side
  /// \param interval integration interval
  float getSACOneVal(const Side side, unsigned int integrationInterval) const;

  /// \return returns the stored DeltaSAC value
  /// \param stack stack
  /// \param interval integration interval
  float getSACDeltaVal(const unsigned int stack, unsigned int interval) const { return mSACDelta->getValue(getSide(stack), getSACDeltaIndex(stack, interval)); }

  /// \return returns index for SAC delta
  /// \param stack stack
  /// \param interval local integration interval
  unsigned int getSACDeltaIndex(const unsigned int stack, unsigned int interval) const { return stack % GEMSTACKSPERSIDE + GEMSTACKSPERSIDE * interval; }

  void setSACZero(SACZero* sacZero) { mSACZero = sacZero; }
  void setSACOne(SACOne* sacOne, const Side side = Side::A) { mSACOne[side] = sacOne; }

  template <typename T>
  void setSACDelta(SACDelta<T>* sacDelta)
  {
    mSACDelta = sacDelta;
  }

  /// setting the fourier coefficients
  void setFourierCoeffSAC(FourierCoeffSAC* fourier) { mFourierSAC = fourier; }

  TCanvas* drawSACTypeSides(const SACType type, const unsigned int integrationInterval, const int minZ = 0, const int maxZ = -1, TCanvas* canv = nullptr);
  TCanvas* drawSACOneCanvas(int nbins1D, float xMin1D, float xMax1D, int integrationIntervals = -1, TCanvas* outputCanvas = nullptr) const;
  TCanvas* drawFourierCoeffSAC(Side side, int nbins1D, float xMin1D, float xMax1, TCanvas* outputCanvas = nullptr) const;

  void dumpToFile(std::string filename, int type = 0);

 private:
  std::array<std::vector<int32_t>, o2::tpc::GEMSTACKS> mSACs{};
  SACZero* mSACZero = nullptr;
  std::array<SACOne*, SIDES> mSACOne{}; ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  SACDelta<unsigned char>* mSACDelta = nullptr;
  FourierCoeffSAC* mFourierSAC = nullptr; ///< fourier coefficients of SACOne

  /// \return returns side for given GEM stack
  Side getSide(const unsigned int gemStack) const { return (gemStack < GEMSTACKSPERSIDE) ? Side::A : Side::C; }

  /// \return returns stack for given sector and stack
  unsigned int getStack(const unsigned int sector, const unsigned int stack) const { return static_cast<unsigned int>(stack + sector * GEMSTACKSPERSECTOR); }

  ClassDefNV(SACs, 1)
};
} // namespace o2::tpc::qc
#endif
