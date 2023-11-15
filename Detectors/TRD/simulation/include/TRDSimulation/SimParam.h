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

/// \file SimParam.h

#ifndef O2_TRD_SIMPARAM_H
#define O2_TRD_SIMPARAM_H

#include <array>
#include "Rtypes.h" // for ClassDef

namespace o2
{
namespace trd
{

/// \brief Constant parameters for the TRD simulation
///
class SimParam
{
 public:
  enum class GasMixture { Xenon,
                          Argon };

  SimParam();
  SimParam(const SimParam&) = delete;
  SimParam& operator=(const SimParam&) = delete;

  /// initialization based on configured gas mixture in TRDSimParams
  void init();

  // Cached magnetic field, to be called by the user before using DiffusionAndTimeStructEstimator::GetDiffCoeff
  void cacheMagField();

  // Setters
  void setGasGain(float gasgain) { mGasGain = gasgain; }
  void setNoise(float noise) { mNoise = noise; }
  void setChipGain(float chipgain) { mChipGain = chipgain; }
  void setADCoutRange(float range) { mADCoutRange = range; }
  void setADCinRange(float range) { mADCinRange = range; }
  void setADCbaseline(int basel) { mADCbaseline = basel; }
  void setDiffusion(bool flag = true) { mDiffusionOn = flag; }
  void setElAttach(bool flag = true) { mElAttachOn = flag; }
  void setElAttachProp(float prop) { mElAttachProp = prop; }
  void setTimeResponse(bool flag = true) { mTRFOn = flag; }
  void setCrossTalk(bool flag = true) { mCTOn = flag; }
  void setPadCoupling(float v) { mPadCoupling = v; }
  void setTimeCoupling(float v) { mTimeCoupling = v; }
  void setTimeStruct(bool flag = true) { mTimeStructOn = flag; }
  void setPadResponse(bool flag = true) { mPRFOn = flag; }
  void setExB(bool flag = true) { mExBOn = flag; }
  void setSamplingFrequency(float freq) { mSamplingFrequency = freq; }
  void setTRF(int trf, float mu = 0., float sigma = 0.3)
  {
    mTRF = trf;
    mMu = mu;
    mSigma = sigma;
  }

  // Getters
  float getGasGain() const { return mGasGain; }
  float getNoise() const { return mNoise; }
  float getChipGain() const { return mChipGain; }
  float getADCoutRange() const { return mADCoutRange; }
  float getADCinRange() const { return mADCinRange; }
  int getADCbaseline() const { return mADCbaseline; }
  float getTRFlo() const { return mTRFlo; }
  float getTRFhi() const { return mTRFhi; }
  float getPadCoupling() const { return mPadCoupling; }
  float getTimeCoupling() const { return mTimeCoupling; }
  bool diffusionOn() const { return mDiffusionOn; }
  bool elAttachOn() const { return mElAttachOn; }
  float getElAttachProp() const { return mElAttachProp; }
  bool trfOn() const { return mTRFOn; }
  bool ctOn() const { return mCTOn; }
  bool timeStructOn() const { return mTimeStructOn; }
  bool prfOn() const { return mPRFOn; }
  int getNumberOfPadsInPadResponse() const { return mNPadsInPadResponse; }
  double timeResponse(double) const;
  double crossTalk(double) const;
  bool isExBOn() const { return mExBOn; }
  bool isXenon() const { return (mGasMixture == GasMixture::Xenon); }
  bool isArgon() const { return (mGasMixture == GasMixture::Argon); }
  GasMixture getGasMixture() const { return mGasMixture; }
  float getSamplingFrequency() const { return mSamplingFrequency; }
  float getCachedField() const;

 private:
  /// Fill the arrays mTRDsmp and mCTsmp for the given gas mixture
  void sampleTRF();

  float mNoise{1250.f};                 ///< Electronics noise
  float mChipGain{12.4f};               ///< Electronics gain
  float mADCoutRange{1023.f};           ///< ADC output range (number of channels for the 10 bit ADC)
  float mADCinRange{2000.f};            ///< ADC input range (2V)
  int mADCbaseline{10};                 ///< ADC intrinsic baseline in ADC channel
  bool mDiffusionOn{true};              ///< Switch for the diffusion
  bool mElAttachOn{false};              ///< Switch for the electron attachment
  float mElAttachProp{0.f};             ///< Propability for electron attachment (for 1m)
  bool mTRFOn{true};                    ///< Switch for the time response
  bool mCTOn{true};                     ///< Switch for cross talk
  bool mPRFOn{true};                    ///< Switch for the pad response
  int mNPadsInPadResponse{3};           ///< Number of pads included in the pad response
  static constexpr int mNBinsMax = 200; ///< Maximum number of bins for integrated time response and cross talk

  // From CommonParam
  GasMixture mGasMixture{GasMixture::Xenon};
  bool mExBOn{true};              ///< Switch for the ExB effects
  bool mFieldCached{false};       ///< flag if B-field has been cached already
  float mField{0.};               ///< Cached magnetic field
  float mSamplingFrequency{10.f}; ///< Sampling Frequency in MHz

  // Use 0.46, instead of the theroetical value 0.3, since it reproduces better
  // the test beam data, even tough it is not understood why.
  float mPadCoupling{.46f}; ///< The pad coupling factor
  float mTimeCoupling{.4f}; ///< Time coupling factor (image charge of moving ions); same number as for the TPC
  bool mTimeStructOn{true}; ///< Switch for cell time structure

  /// The parameters below depend on the gas mixture and are re-evaluated upon a change
  std::array<float, mNBinsMax> mTRFsmp{};                            ///< Integrated time response
  std::array<float, mNBinsMax> mCTsmp{};                             ///< Integrated cross talk
  int mTRF{0};                                                       ///< Sampled TRF function. 0: default TRF, 1: TRF described in TRF TDR, 2: No TRF, 3: Landau dist as TRF (parameters specified below)
  float mMu{0.};                                                     ///< Mu of the Landau distribution used to describe TRF
  float mSigma{0.03};                                                ///< Sigma of the Landau distribution used to describe TRF
  int mTRFbin{200};                                                  ///<  Number of bins for the TRF and x-talk
  float mTRFlo{-.4f};                                                ///<  Lower boundary of the TRF and x-talk
  float mTRFhi{3.58f};                                               ///<  Higher boundary of the TRF and x-talk
  float mInvTRFwid{static_cast<float>(mTRFbin) / (mTRFhi - mTRFlo)}; ///<  Inverse of the bin width of the integrated TRF and x-talk
  float mGasGain{4000.f};                                            ///< Gas gain

  ClassDefNV(SimParam, 2); // The TRD simulation parameters
};

} // namespace trd
} // namespace o2
#endif
