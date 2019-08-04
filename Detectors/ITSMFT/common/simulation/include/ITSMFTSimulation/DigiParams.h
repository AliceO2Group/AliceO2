// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigiParams.h
/// \brief Simulation parameters for the ALIPIDE chip

#ifndef ALICEO2_ITSMFT_DIGIPARAMS_H
#define ALICEO2_ITSMFT_DIGIPARAMS_H

#include <Rtypes.h>
#include <ITSMFTSimulation/AlpideSignalTrapezoid.h>

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

  void setROFrameLength(float ns);
  float getROFrameLength() const { return mROFrameLength; }
  float getROFrameLengthInv() const { return mROFrameLengthInv; }

  void setStrobeDelay(float ns) { mStrobeDelay = ns; }
  float getStrobeDelay() const { return mStrobeDelay; }

  void setStrobeLength(float ns) { mStrobeLength = ns; }
  float getStrobeLength() const { return mStrobeLength; }

  void setTimeOffset(double sec) { mTimeOffset = sec; }
  double getTimeOffset() const { return mTimeOffset; }

  void setChargeThreshold(int v, float frac2Account = 0.1);
  void setNSimSteps(int v);
  void setEnergyToNElectrons(float v) { mEnergyToNElectrons = v; }

  int getChargeThreshold() const { return mChargeThreshold; }
  int getMinChargeToAccount() const { return mMinChargeToAccount; }
  int getNSimSteps() const { return mNSimSteps; }
  float getNSimStepsInv() const { return mNSimStepsInv; }
  float getEnergyToNElectrons() const { return mEnergyToNElectrons; }

  bool isTimeOffsetSet() const { return mTimeOffset > -infTime; }

  const o2::itsmft::AlpideSimResponse* getAlpSimResponse() const { return mAlpSimResponse; }
  void setAlpSimResponse(const o2::itsmft::AlpideSimResponse* par) { mAlpSimResponse = par; }

  const SignalShape& getSignalShape() const { return mSignalShape; }
  SignalShape& getSignalShape() { return (SignalShape&)mSignalShape; }

  void print() const;

 private:
  static constexpr double infTime = 1e99;
  bool mIsContinuous = false;        ///< flag for continuous simulation
  float mNoisePerPixel = 1.e-7;      ///< ALPIDE Noise per chip
  float mROFrameLength = 6000.;      ///< length of RO frame in ns
  float mStrobeDelay = 6000.;        ///< strobe start (in ns) wrt ROF start
  float mStrobeLength = 100.;        ///< length of the strobe in ns (sig. over threshold checked in this window only)
  double mTimeOffset = -2 * infTime; ///< time offset (in seconds!) to calculate ROFrame from hit time

  int mChargeThreshold = 150;              ///< charge threshold in Nelectrons
  int mMinChargeToAccount = 15;            ///< minimum charge contribution to account
  int mNSimSteps = 7;                      ///< number of steps in response simulation
  float mEnergyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  o2::itsmft::AlpideSignalTrapezoid mSignalShape; ///< signal timeshape parameterization

  const o2::itsmft::AlpideSimResponse* mAlpSimResponse = nullptr; //!< pointer on external response

  // auxiliary precalculated parameters
  float mROFrameLengthInv = 0; ///< inverse length of RO frame in ns
  float mNSimStepsInv = 0;     ///< its inverse

  ClassDefNV(DigiParams, 1);
};
} // namespace itsmft
} // namespace o2

#endif
