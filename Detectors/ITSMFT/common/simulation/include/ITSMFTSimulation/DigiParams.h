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

/// \file DigiParams.h
/// \brief Simulation parameters for the ALIPIDE chip

#ifndef ALICEO2_ITSMFT_DIGIPARAMS_H
#define ALICEO2_ITSMFT_DIGIPARAMS_H

#include <vector>
#include <algorithm>
#include <Rtypes.h>
#include "ITSMFTSimulation/AlpideSignalTrapezoid.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"

////////////////////////////////////////////////////////////
//                                                        //
// Simulation params for Alpide chip                      //
//                                                        //
// This is a provisionary implementation, until proper    //
// microscopic simulation and its configuration will      //
// be implemented                                         //
//                                                        //
////////////////////////////////////////////////////////////

namespace o2
{
namespace itsmft
{

class AlpideSimResponse;

class DigiParams
{

  using SignalShape = o2::itsmft::AlpideSignalTrapezoid;

 public:
  DigiParams();
  ~DigiParams() = default;

  void setNoisePerPixel(float v) { mNoisePerPixel = v; }
  float getNoisePerPixel() const { return mNoisePerPixel; }

  void setContinuous(bool v) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }

  int getROFrameLengthInBC(int layer = -1) const { return layer < 0 ? mROFrameLengthInBC : mROFrameLayerLengthInBC[layer]; }
  void setROFrameLengthInBC(int n, int layer = -1) { layer < 0 ? mROFrameLengthInBC = n : mROFrameLayerLengthInBC[layer] = n; }

  void setROFrameLength(float ns, int layer = -1);
  float getROFrameLength(int layer = -1) const { return layer < 0 ? mROFrameLength : mROFrameLayerLength[layer]; }
  float getROFrameLengthInv(int layer = -1) const { return layer < 0 ? mROFrameLengthInv : mROFrameLayerLengthInv[layer]; }

  void setStrobeDelay(float ns) { mStrobeDelay = ns; }
  float getStrobeDelay(int layer = -1) const { return layer < 0 ? mStrobeDelay : mStrobeLayerDelay[layer]; }

  void setStrobeLength(float ns) { mStrobeLength = ns; }
  float getStrobeLength(int layer = -1) const { return layer < 0 ? mStrobeLength : mStrobeLayerLength[layer]; }

  void setTimeOffset(double sec) { mTimeOffset = sec; }
  double getTimeOffset() const { return mTimeOffset; }

  void setROFrameBiasInBC(int n, int layer = -1) { layer < 0 ? mROFrameBiasInBC = n : mROFrameLayerBiasInBC[layer] = n; }
  int getROFrameBiasInBC(int layer = -1) const { return layer < 0 ? mROFrameBiasInBC : mROFrameLayerBiasInBC[layer]; }

  void setChargeThreshold(int v, float frac2Account = 0.1);
  void setNSimSteps(int v);
  void setEnergyToNElectrons(float v) { mEnergyToNElectrons = v; }

  void setVbb(float v) { mVbb = v; }
  void setIBVbb(float v) { mIBVbb = v; }
  void setOBVbb(float v) { mOBVbb = v; }

  int getChargeThreshold() const { return mChargeThreshold; }
  int getMinChargeToAccount() const { return mMinChargeToAccount; }
  int getNSimSteps() const { return mNSimSteps; }
  float getNSimStepsInv() const { return mNSimStepsInv; }
  float getEnergyToNElectrons() const { return mEnergyToNElectrons; }

  float getVbb() const { return mVbb; }
  float getIBVbb() const { return mIBVbb; }
  float getOBVbb() const { return mOBVbb; }

  bool isTimeOffsetSet() const { return mTimeOffset > -infTime; }

  const o2::itsmft::AlpideSimResponse* getAlpSimResponse() const { return mAlpSimResponse; }
  void setAlpSimResponse(const o2::itsmft::AlpideSimResponse* par) { mAlpSimResponse = par; }

  const SignalShape& getSignalShape() const { return mSignalShape; }
  SignalShape& getSignalShape() { return (SignalShape&)mSignalShape; }

  bool withStaggering() const noexcept { return !mROFrameLayerLength.empty(); }
  void addROFrameLayerLengthInBC(int len) { mROFrameLayerLengthInBC.push_back(len); }
  void addROFrameLayerBiasInBC(int len) { mROFrameLayerBiasInBC.push_back(len); }
  void addStrobeLength(float ns) { mStrobeLayerLength.push_back(ns); }
  void addStrobeDelay(float ns) { mStrobeLayerDelay.push_back(ns); }

  virtual void print() const;

 private:
  static constexpr double infTime = 1e99;
  bool mIsContinuous = false;              ///< flag for continuous simulation
  float mNoisePerPixel = 1.e-8;            ///< ALPIDE Noise per chip
  int mROFrameLengthInBC = 0;              ///< ROF length in BC for continuous mode
  float mROFrameLength = 0;                ///< length of RO frame in ns
  float mStrobeDelay = 0.;                 ///< strobe start (in ns) wrt ROF start
  float mStrobeLength = 0;                 ///< length of the strobe in ns (sig. over threshold checked in this window only)
  double mTimeOffset = -2 * infTime;       ///< time offset (in seconds!) to calculate ROFrame from hit time
  int mROFrameBiasInBC = 0;                ///< misalignment of the ROF start in BC
  int mChargeThreshold = 150;              ///< charge threshold in Nelectrons
  int mMinChargeToAccount = 15;            ///< minimum charge contribution to account
  int mNSimSteps = 7;                      ///< number of steps in response simulation
  float mEnergyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  float mVbb = 0.0;   ///< back bias absolute value for MFT (in Volt)
  float mIBVbb = 0.0; ///< back bias absolute value for ITS Inner Barrel (in Volt)
  float mOBVbb = 0.0; ///< back bias absolute value for ITS Outer Barrel (in Volt)

  std::vector<int> mROFrameLayerLengthInBC; ///< staggering ROF length in BC for continuous mode per layer
  std::vector<int> mROFrameLayerBiasInBC;   ///< staggering ROF bias in BC for continuous mode per layer
  std::vector<float> mROFrameLayerLength;   ///< staggering ROF length in ns for continuous mode per layer
  std::vector<float> mStrobeLayerLength;    ///< staggering length of the strobe in ns (sig. over threshold checked in this window only)
  std::vector<float> mStrobeLayerDelay;     ///< staggering delay of the strobe in ns

  o2::itsmft::AlpideSignalTrapezoid mSignalShape; ///< signal timeshape parameterization

  const o2::itsmft::AlpideSimResponse* mAlpSimResponse = nullptr; //!< pointer on external response

  // auxiliary precalculated parameters
  float mROFrameLengthInv = 0;               ///< inverse length of RO frame in ns
  std::vector<float> mROFrameLayerLengthInv; // inverse length of RO frame in ns per layer
  float mNSimStepsInv = 0;                   ///< its inverse

  ClassDef(DigiParams, 3);
};
} // namespace itsmft
} // namespace o2

#endif
