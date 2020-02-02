// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_SIMPARAM_H_
#define ALICEO2_EMCAL_SIMPARAM_H_

#include <iosfwd>
#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"
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
  Float_t getTimeNoise() const { return mTimeNoise; }
  Float_t getTimeDelay() const { return mTimeDelay; }
  Bool_t isTimeDelayFromOCDB() const { return mTimeDelayFromCDB; }
  Float_t getTimeResolutionPar0() const { return mTimeResolutionPar0; }
  Float_t getTimeResolutionPar1() const { return mTimeResolutionPar1; }
  Double_t getTimeResolution(Double_t energy) const;
  Int_t getNADCEC() const { return mNADCEC; }
  Int_t getMeanPhotonElectron() const { return mMeanPhotonElectron; }
  Float_t getGainFluctuations() const { return mGainFluctuations; }
  Int_t getTimeResponseTau() const { return mTimeResponseTau; }
  Float_t getTimeResponsePower() const { return mTimeResponsePower; }
  Float_t getTimeResponseThreshold() const { return mTimeResponsePower; }

  // Parameters used in SDigitizer
  Float_t getA() const { return mA; }
  Float_t getB() const { return mB; }
  Float_t getECPrimaryThreshold() const { return mECPrimThreshold; }

  Float_t getSignalDelay() const { return mSignalDelay; }
  Float_t getLiveTime() const { return mLiveTime; }
  Float_t getBusyTime() const { return mBusyTime; }
  Bool_t isDisablePileup() const { return mDisablePileup; }

  void PrintStream(std::ostream& stream) const;

 private:

  // Digitizer
  Int_t mDigitThreshold{3};              ///< Threshold for storing digits in EMC
  Int_t mMeanPhotonElectron{4400};       ///< number of photon electrons per GeV deposited energy
  Float_t mGainFluctuations{15.};        ///< correct fMeanPhotonElectron by the gain fluctuations
  Float_t mPinNoise{0.012};              ///< Electronics noise in EMC, APD
  Float_t mTimeNoise{1.28e-5};           ///< Electronics noise in EMC, time
  Float_t mTimeDelay{600e-9};            ///< Simple time delay to mimick roughly delay in data
  Bool_t mTimeDelayFromCDB{false};       ///< Get time delay from OCDB
  Float_t mTimeResolutionPar0{0.26666};  ///< Time resolution of FEE electronics
  Float_t mTimeResolutionPar1{1.4586};   ///< Time resolution of FEE electronics
  Int_t mNADCEC{0x10000};                ///< number of channels in EC section ADC
  Float_t mTimeResponseTau{2.35};        ///< Raw time response function tau parameter
  Float_t mTimeResponsePower{2};         ///< Raw time response function power parameter
  Float_t mTimeResponseThreshold{0.001}; ///< Raw time response function energy threshold

  // SDigitizer
  Float_t mA{0.};                 ///< Pedestal parameter
  Float_t mB{1.e+6};              ///< Slope Digitizition parameters
  Float_t mECPrimThreshold{0.05}; ///< To store primary if EC Shower Elos > threshold

  // Timing
  Float_t mSignalDelay{700}; ///< Signal delay time (ns)
  Float_t mLiveTime{1500};   ///< EMCal live time (ns)
  Float_t mBusyTime{0};      ///< EMCal busy time (ns)

  // DigitizerSpec
  Bool_t mDisablePileup{false}; ///< disable pileup simulation

  O2ParamDef(SimParam, "EMCSimParam");
};

std::ostream& operator<<(std::ostream& stream, const SimParam& s);

} // namespace emcal
} // namespace o2

#endif
