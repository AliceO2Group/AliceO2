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
#ifndef ALICEO2_EMCAL_SIMPARAM_H_
#define ALICEO2_EMCAL_SIMPARAM_H_

#include <iosfwd>
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "Rtypes.h"

namespace o2
{
namespace emcal
{
/// \class SimParam
/// \brief EMCal simulation parameters
/// \ingroup EMCALsimulation

class SimParam : public o2::conf::ConfigurableParamHelper<SimParam>
{
 public:
  ~SimParam() override = default;

  // Parameters used in Digitizer
  Int_t getDigitThreshold() const { return mDigitThreshold; }
  Float_t getPinNoise() const { return mPinNoise; }
  Float_t getPinNoiseLG() const { return mPinNoiseLG; }
  Float_t getTimeResolutionPar0() const { return mTimeResolutionPar0; }
  Float_t getTimeResolutionPar1() const { return mTimeResolutionPar1; }
  Double_t getTimeResolution(Double_t energy) const;
  Int_t getNADCEC() const { return mNADCEC; }
  Int_t getMeanPhotonElectron() const { return mMeanPhotonElectron; }
  Float_t getGainFluctuations() const { return mGainFluctuations; }
  Float_t getTimeResponseTau() const { return mTimeResponseTau; }
  Float_t getTimeResponsePower() const { return mTimeResponsePower; }
  Float_t getTimeResponseThreshold() const { return mTimeResponseThreshold; }
  Int_t getBCPhaseSwap() const { return mSwapPhase; }

  // Parameters used in SDigitizer
  Float_t getA() const { return mA; }
  Float_t getB() const { return mB; }
  Float_t getECPrimaryThreshold() const { return mECPrimThreshold; }

  Float_t getSignalDelay() const { return mSignalDelay; }
  unsigned int getTimeBinOffset() const { return mTimeWindowStart; }
  Float_t getLiveTime() const { return mLiveTime; }
  Float_t getBusyTime() const { return mBusyTime; }
  Float_t getPreTriggerTime() const { return mPreTriggerTime; }

  Bool_t doSmearEnergy() const { return mSmearEnergy; }
  Bool_t doSimulateTimeResponse() const { return mSimulateTimeResponse; }
  Bool_t doRemoveDigitsBelowThreshold() const { return mRemoveDigitsBelowThreshold; }
  Bool_t doSimulateNoiseDigits() const { return mSimulateNoiseDigits; }
  Bool_t doSimulateL1Phase() const { return mSimulateL1Phase; }

  Bool_t isDisablePileup() const { return mDisablePileup; }

  void PrintStream(std::ostream& stream) const;

  // Digitizer
  Int_t mDigitThreshold{3};              ///< Threshold for storing digits in EMC
  Int_t mMeanPhotonElectron{4400};       ///< number of photon electrons per GeV deposited energy
  Float_t mGainFluctuations{15.};        ///< correct fMeanPhotonElectron by the gain fluctuations
  Float_t mPinNoise{0.012};              ///< Electronics noise in EMC, APD
  Float_t mPinNoiseLG{0.1};              ///< Electronics noise in EMC, APD, Low Gain
  Float_t mTimeResolutionPar0{0.26666};  ///< Time resolution of FEE electronics
  Float_t mTimeResolutionPar1{1.4586};   ///< Time resolution of FEE electronics
  Int_t mNADCEC{0x10000};                ///< number of channels in EC section ADC
  Float_t mTimeResponseTau{2.35};        ///< Raw time response function tau parameter
  Float_t mTimeResponsePower{2};         ///< Raw time response function power parameter
  Float_t mTimeResponseThreshold{0.001}; ///< Raw time response function energy threshold
  Int_t mSwapPhase{0};                   ///< BC phase swap similar to data

  // SDigitizer
  Float_t mA{0.};                 ///< Pedestal parameter
  Float_t mB{1.e+6};              ///< Slope Digitizition parameters
  Float_t mECPrimThreshold{0.05}; ///< To store primary if EC Shower Elos > threshold

  // Timing
  Float_t mSignalDelay{600};          ///< Signal delay time (ns)
  unsigned int mTimeWindowStart{400}; ///< The start of the time window
  Float_t mLiveTime{1500};            ///< EMCal live time (ns)
  Float_t mBusyTime{35000};           ///< EMCal busy time (ns)
  Float_t mPreTriggerTime{600};       ///< EMCal pre-trigger time (ns)

  // Processing
  Bool_t mSmearEnergy{true};                ///< do time and energy smearing
  Bool_t mSimulateTimeResponse{true};       ///< simulate time response
  Bool_t mRemoveDigitsBelowThreshold{true}; ///< remove digits below threshold
  Bool_t mSimulateNoiseDigits{true};        ///< simulate noise digits
  bool mSimulateL1Phase{true};              ///< Simulate L1 phase

  // DigitizerSpec
  Bool_t mDisablePileup{false}; ///< disable pileup simulation

  O2ParamDef(SimParam, "EMCSimParam");
};

std::ostream& operator<<(std::ostream& stream, const SimParam& s);

} // namespace emcal

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::emcal::SimParam> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
