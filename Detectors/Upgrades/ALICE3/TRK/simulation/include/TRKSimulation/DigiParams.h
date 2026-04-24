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
/// \brief Simulation parameters for the TRK digitizer. Based on the ITS2 and ITS3 digitizer parameters

#ifndef ALICEO2_TRK_DIGIPARAMS_H
#define ALICEO2_TRK_DIGIPARAMS_H

#include <array>

#include <Rtypes.h>
#include "ITSMFTSimulation/AlpideSignalTrapezoid.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "TRKBase/AlmiraParam.h"
#include "TRKBase/TRKBaseParam.h"
#include "TRKBase/GeometryTGeo.h"

////////////////////////////////////////////////////////////
//                                                        //
// Simulation params for the TRK digitizer                //
//                                                        //
// This is a provisionary implementation, until proper    //
// microscopic simulation and its configuration will      //
// be implemented                                         //
//                                                        //
////////////////////////////////////////////////////////////

namespace o2
{
namespace trk
{

class ChipSimResponse;

class DigiParams
{

  using SignalShape = o2::itsmft::AlpideSignalTrapezoid;

 public:
  DigiParams();
  ~DigiParams() = default;

  void setNoisePerPixel(float v) { mNoisePerPixel = v; }
  float getNoisePerPixel() const { return mNoisePerPixel; }

  int getROFrameLengthInBC(int layer) const { return mROFrameLayerLengthInBC[layer]; }
  void setROFrameLengthInBC(int n, int layer) { mROFrameLayerLengthInBC[layer] = n; }

  void setROFrameLength(float ns, int layer);
  float getROFrameLength(int layer) const { return mROFrameLayerLength[layer]; }
  float getROFrameLengthInv(int layer) const { return mROFrameLayerLengthInv[layer]; }

  void setStrobeDelay(float ns, int layer) { mStrobeLayerDelay[layer] = ns; }
  float getStrobeDelay(int layer) const { return mStrobeLayerDelay[layer]; }

  void setStrobeLength(float ns, int layer) { mStrobeLayerLength[layer] = ns; }
  float getStrobeLength(int layer) const { return mStrobeLayerLength[layer]; }

  void setTimeOffset(double sec) { mTimeOffset = sec; }
  double getTimeOffset() const { return mTimeOffset; }

  void setROFrameBiasInBC(int n, int layer) { mROFrameLayerBiasInBC[layer] = n; }
  int getROFrameBiasInBC(int layer) const { return mROFrameLayerBiasInBC[layer]; }

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

  const o2::trk::ChipSimResponse* getResponse() const { return mResponse.get(); }
  void setResponse(const o2::itsmft::AlpideSimResponse*);

  const SignalShape& getSignalShape() const { return mSignalShape; }
  SignalShape& getSignalShape() { return (SignalShape&)mSignalShape; }

  virtual void print() const;

 private:
  static constexpr double infTime = 1e99;
  float mNoisePerPixel = 1.e-7;          ///< Noise per chip
  double mTimeOffset = -2 * infTime;     ///< time offset (in seconds!) to calculate ROFrame from hit time
  int mChargeThreshold = 75;             ///< charge threshold in Nelectrons
  int mMinChargeToAccount = 7;           ///< minimum charge contribution to account
  int mNSimSteps = 475;                  ///< number of steps in response simulation
  float mNSimStepsInv = 1. / mNSimSteps; ///< its inverse

  float mEnergyToNElectrons = 1. / 3.6e-9; // conversion of eloss to Nelectrons

  float mVbb = 0.0;   ///< back bias absolute value for MFT (in Volt)
  float mIBVbb = 0.0; ///< back bias absolute value for ITS Inner Barrel (in Volt)
  float mOBVbb = 0.0; ///< back bias absolute value for ITS Outter Barrel (in Volt)

  std::array<int, o2::trk::AlmiraParam::getNLayers()> mROFrameLayerLengthInBC; ///< staggering ROF length in BC for continuous mode per layer
  std::array<int, o2::trk::AlmiraParam::getNLayers()> mROFrameLayerBiasInBC;   ///< staggering ROF bias in BC for continuous mode per layer
  std::array<float, o2::trk::AlmiraParam::getNLayers()> mROFrameLayerLength;   ///< staggering ROF length in ns for continuous mode per layer
  std::array<float, o2::trk::AlmiraParam::getNLayers()> mStrobeLayerLength;    ///< staggering strobe length in ns per layer
  std::array<float, o2::trk::AlmiraParam::getNLayers()> mStrobeLayerDelay;     ///< staggering strobe delay in ns per layer

  o2::itsmft::AlpideSignalTrapezoid mSignalShape; ///< signal timeshape parameterization

  std::unique_ptr<o2::trk::ChipSimResponse> mResponse; //!< pointer on external response

  // auxiliary precalculated parameters
  std::array<float, o2::trk::AlmiraParam::getNLayers()> mROFrameLayerLengthInv; ///< inverse length of RO frame in ns per layer

  //   ClassDef(DigiParams, 2);
};
} // namespace trk
} // namespace o2

#endif
