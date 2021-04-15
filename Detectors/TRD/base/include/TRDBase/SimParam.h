// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  enum {
    kNPadsInPadResponse = 3 ///< Number of pads included in the pad response
  };

  /// For a singleton class copying and assigning is not allowed
  SimParam(const SimParam&) = delete;
  SimParam& operator=(const SimParam&) = delete;

  /// Returns an instance of this class. A SimParam object is created upon the first call of this function
  static SimParam* instance();

  /// After setting a new gas mixture the parameters need to be re-evaluated
  void reInit();

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
  const int getNumberOfPadsInPadResponse() const { return kNPadsInPadResponse; }
  inline double timeResponse(double) const;
  inline double crossTalk(double) const;

 protected:
  static SimParam* mgInstance; ///<  Instance of this class (singleton implementation)

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
  bool mPRFOn{true};                    ///<  Switch for the pad response
  static constexpr int mNBinsMax = 200; ///< Maximum number of bins for integrated time response and cross talk

  // Use 0.46, instead of the theroetical value 0.3, since it reproduces better
  // the test beam data, even tough it is not understood why.
  float mPadCoupling{.46f}; ///< The pad coupling factor
  float mTimeCoupling{.4f}; ///< Time coupling factor (image charge of moving ions); same number as for the TPC
  bool mTimeStructOn{true}; ///< Switch for cell time structure

  /// The parameters below depend on the gas mixture and are re-evaluated upon a change
  std::array<float, mNBinsMax> mTRFsmp{};                            ///< Integrated time response
  std::array<float, mNBinsMax> mCTsmp{};                             ///< Integrated cross talk
  int mTRFbin{200};                                                  ///<  Number of bins for the TRF and x-talk
  float mTRFlo{-.4f};                                                ///<  Lower boundary of the TRF and x-talk
  float mTRFhi{3.58f};                                               ///<  Higher boundary of the TRF and x-talk
  float mInvTRFwid{static_cast<float>(mTRFbin) / (mTRFhi - mTRFlo)}; ///<  Inverse of the bin width of the integrated TRF and x-talk
  float mGasGain{4000.f};                                            ///< Gas gain

 private:
  /// This is a singleton, constructor is private!
  SimParam();
  ~SimParam() = default;

  /// Fill the arrays mTRDsmp and mCTsmp for the given gas mixture
  void sampleTRF();

  ClassDefNV(SimParam, 1); // The TRD simulation parameters
};

inline double SimParam::timeResponse(double time) const
{
  //
  // Applies the preamp shaper time response
  // (We assume a signal rise time of 0.2us = fTRFlo/2.
  //

  double rt = (time - .5 * mTRFlo) * mInvTRFwid;
  int iBin = (int)rt;
  double dt = rt - iBin;
  if ((iBin >= 0) && (iBin + 1 < mTRFbin)) {
    return mTRFsmp[iBin] + (mTRFsmp[iBin + 1] - mTRFsmp[iBin]) * dt;
  } else {
    return 0.0;
  }
}

inline double SimParam::crossTalk(double time) const
{
  //
  // Applies the pad-pad capacitive cross talk
  //

  double rt = (time - mTRFlo) * mInvTRFwid;
  int iBin = (int)rt;
  double dt = rt - iBin;
  if ((iBin >= 0) && (iBin + 1 < mTRFbin)) {
    return mCTsmp[iBin] + (mCTsmp[iBin + 1] - mCTsmp[iBin]) * dt;
  } else {
    return 0.0;
  }
}

} // namespace trd
} // namespace o2
#endif
