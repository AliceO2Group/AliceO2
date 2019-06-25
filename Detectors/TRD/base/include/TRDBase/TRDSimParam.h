// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDSIMPARAM_H
#define O2_TRDSIMPARAM_H

//Forwards to standard header with protection for GPU compilation
#include "GPUCommonRtypes.h" // for ClassDef

namespace o2
{
namespace trd
{
////////////////////////////////////////////////////////////////////////////
//                                                                        //
// Class containing constant simulation parameters                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
class TRDSimParam
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540,
         kNPadsInPadResponse = 3 // Number of pads included in the pad response
  };

  static TRDSimParam* Instance();
  static void Terminate();

  void SetGasGain(float gasgain) { mGasGain = gasgain; }
  void SetNoise(float noise) { mNoise = noise; }
  void SetChipGain(float chipgain) { mChipGain = chipgain; }
  void SetADCoutRange(float range) { mADCoutRange = range; }
  void SetADCinRange(float range) { mADCinRange = range; }
  void SetADCbaseline(int basel) { mADCbaseline = basel; }
  void SetDiffusion(int diffOn = 1) { mDiffusionOn = diffOn; }
  void SetElAttach(int elOn = 1) { mElAttachOn = elOn; }
  void SetElAttachProp(float prop) { mElAttachProp = prop; }
  void SetTimeResponse(int trfOn = 1)
  {
    mTRFOn = trfOn;
    ReInit();
  }
  void SetCrossTalk(int ctOn = 1)
  {
    mCTOn = ctOn;
    ReInit();
  }
  void SetPadCoupling(float v) { mPadCoupling = v; }
  void SetTimeCoupling(float v) { mTimeCoupling = v; }
  void SetTimeStruct(bool tsOn = 1) { mTimeStructOn = tsOn; }
  void SetPadResponse(int prfOn = 1) { mPRFOn = prfOn; }
  void SetNTimeBins(int ntb) { mNTimeBins = ntb; }
  void SetNTBoverwriteOCDB(bool over = true) { mNTBoverwriteOCDB = over; }
  float GetGasGain() const { return mGasGain; }
  float GetNoise() const { return mNoise; }
  float GetChipGain() const { return mChipGain; }
  float GetADCoutRange() const { return mADCoutRange; }
  float GetADCinRange() const { return mADCinRange; }
  int GetADCbaseline() const { return mADCbaseline; }
  float GetTRFlo() const { return mTRFlo; }
  float GetTRFhi() const { return mTRFhi; }
  float GetPadCoupling() const { return mPadCoupling; }
  float GetTimeCoupling() const { return mTimeCoupling; }
  int GetNTimeBins() const { return mNTimeBins; }
  bool GetNTBoverwriteOCDB() const { return mNTBoverwriteOCDB; }
  bool DiffusionOn() const { return mDiffusionOn; }
  bool ElAttachOn() const { return mElAttachOn; }
  float GetElAttachProp() const { return mElAttachProp; }
  bool TRFOn() const { return mTRFOn; }
  bool CTOn() const { return mCTOn; }
  bool TimeStructOn() const { return mTimeStructOn; }
  bool PRFOn() const { return mPRFOn; }
  inline double TimeResponse(double time) const
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
  inline double CrossTalk(double time) const
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

  const int getNumberOfPadsInPadResponse() const { return kNPadsInPadResponse; }

  void ReInit();

 protected:
  static TRDSimParam* fgInstance; //  Instance of this class (singleton implementation)
  static bool fgTerminated;       //  Defines if this class has already been terminated and
                                  //  therefore does not return instances in GetInstance anymore

  // Digitization parameter
  float mGasGain;  //  Gas gain
  float mNoise;    //  Electronics noise
  float mChipGain; //  Electronics gain

  float mADCoutRange; //  ADC output range (number of channels)
  float mADCinRange;  //  ADC input range (input charge)
  int mADCbaseline;   //  ADC intrinsic baseline in ADC channel

  int mDiffusionOn; //  Switch for the diffusion

  int mElAttachOn;     //  Switch for the electron attachment
  float mElAttachProp; //  Propability for electron attachment (for 1m)

  int mTRFOn;       //  Switch for the time response
  float* mTRFsmp;   //! Integrated time response
  int mTRFbin;      //  Number of bins for the TRF
  float mTRFlo;     //  Lower boundary of the TRF
  float mTRFhi;     //  Higher boundary of the TRF
  float mInvTRFwid; //  Inverse of the bin width of the integrated TRF

  int mCTOn;     //  Switch for cross talk
  float* mCTsmp; //! Integrated cross talk

  float mPadCoupling;  //  Pad coupling factor
  float mTimeCoupling; //  Time coupling factor (image charge of moving ions)
  int mTimeStructOn;   //  Switch for cell time structure

  int mPRFOn; //  Switch for the pad response

  int mNTimeBins;         //  Number of time bins (only used it fNTBoverwriteOCDB = true)
  bool mNTBoverwriteOCDB; //  Switch to overwrite number of time bins from PCDB

 private:
  // This is a singleton, constructor is private!
  TRDSimParam();
  ~TRDSimParam();

  void Init();
  void SampleTRF();

  ClassDefNV(TRDSimParam, 1) // The TRD simulation parameters
};
} // namespace trd
} // namespace o2
#endif
