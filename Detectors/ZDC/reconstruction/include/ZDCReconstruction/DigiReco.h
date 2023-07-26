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

#include <map>
#include <deque>
#include <gsl/span>
#include <TFile.h>
#include <TTree.h>
#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCReconstruction/BaselineParam.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/RecEventAux.h"

#ifndef ALICEO2_ZDC_DIGI_RECO_H
#define ALICEO2_ZDC_DIGI_RECO_H

//#define ALICEO2_ZDC_DIGI_RECO_DEBUG
#ifdef O2_ZDC_DEBUG
#ifndef ALICEO2_ZDC_DIGI_RECO_DEBUG
#define ALICEO2_ZDC_DIGI_RECO_DEBUG
#endif
#endif

namespace o2
{
namespace zdc
{
using O2_ZDC_DIGIRECO_FLT = float;

struct DigiRecoTDC {
  DigiRecoTDC(uint16_t myval, uint16_t myamp, o2::InteractionRecord myir)
  {
    val = myval;
    amp = myamp;
    ir = myir;
  }
  uint16_t val;
  uint16_t amp;
  o2::InteractionRecord ir;
};

class DigiReco
{
 public:
  DigiReco() = default;
  ~DigiReco() = default;
  void init();
  void prepareInterpolation();
  int process(const gsl::span<const o2::zdc::OrbitData>& orbitdata,
              const gsl::span<const o2::zdc::BCData>& bcdata,
              const gsl::span<const o2::zdc::ChannelData>& chdata);
  int write();
  void setVerbosity(int v)
  {
    mVerbosity = v;
  }
  int getVerbosity() const { return mVerbosity; }
  void setDebugOutput(bool state = true)
  {
    mTreeDbg = state;
  }
  void eor();
  uint8_t getTriggerCondition() { return mTriggerCondition; }
  void setTripleTrigger() { mTriggerCondition = 0x7; }
  void setDoubleTrigger() { mTriggerCondition = 0x3; }
  void setSingleTrigger() { mTriggerCondition = 0x1; }

  void setAlpha(double v) { mAlpha = v; };
  double getAlpha() { return mAlpha; };

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  void setTDCParam(const ZDCTDCParam* param) { mTDCParam = param; };
  const ZDCTDCParam* getTDCParam() { return mTDCParam; };
  void setTDCCorr(const ZDCTDCCorr* param) { mTDCCorr = param; };
  const ZDCTDCCorr* getTDCCorr() { return mTDCCorr; };
  void setEnergyParam(const ZDCEnergyParam* param) { mEnergyParam = param; };
  const ZDCEnergyParam* getEnergyParam() { return mEnergyParam; };
  void setTowerParam(const ZDCTowerParam* param) { mTowerParam = param; };
  const ZDCTowerParam* getTowerParam() { return mTowerParam; };
  void setBaselineParam(const BaselineParam* param) { mPedParam = param; };
  const BaselineParam* getBaselineParam() { return mPedParam; };
  void setRecoConfigZDC(const RecoConfigZDC* cfg) { mRecoConfigZDC = cfg; };
  const RecoConfigZDC* getRecoConfigZDC() { return mRecoConfigZDC; };
  // Enable or disable low pass filtering
  void setLowPassFilter(bool val = true)
  {
    mLowPassFilter = val;
    mLowPassFilterSet = true;
    LOG(warn) << __func__ << " Configuration of low pass filtering: " << (mLowPassFilter ? "enabled" : "disabled");
  };
  bool getLowPassFilter() { return mLowPassFilter; };
  void setFullInterpolation(bool val = true)
  {
    mFullInterpolation = val;
    mFullInterpolationSet = true;
    LOG(warn) << __func__ << " Full waveform interpolation: " << (mFullInterpolation ? "enabled" : "disabled");
  };
  bool getFullInterpolation() { return mFullInterpolation; };
  // Enable or disable TDC corrections
  void setCorrSignal(bool val = true)
  {
    mCorrSignal = val;
    mCorrSignalSet = true;
    LOG(warn) << __func__ << " Configuration of TDC signal correction: " << (mCorrSignal ? "enabled" : "disabled");
  };
  bool getCorrSignal() { return mCorrSignal; };
  void setCorrBackground(bool val = true)
  {
    mCorrBackground = val;
    mCorrBackgroundSet = true;
    LOG(warn) << __func__ << " Configuration of TDC pile-up correction: " << (mCorrBackground ? "enabled" : "disabled");
  };
  bool getCorrBackground() { return mCorrBackground; };
  bool inError()
  {
    return mInError;
  }

  const uint32_t* getTDCMask() const { return mTDCMask; }
  const uint32_t* getChMask() const { return mChMask; }
  const std::vector<o2::zdc::RecEventAux>& getReco() { return mReco; }

 private:
  const ModuleConfig* mModuleConfig = nullptr;              /// Trigger/readout configuration object
  void updateOffsets(int ibun);                             /// Update offsets to process current bunch
  void lowPassFilter();                                     /// low-pass filtering of digitized data
  int reconstructTDC(int seq_beg, int seq_end);             /// Reconstruction of uncorrected TDCs
  int reconstruct(int seq_beg, int seq_end);                /// Main method for data reconstruction
  int processTrigger(int itdc, int ibeg, int iend);         /// Replay of trigger algorithm on acquired data
  int processTriggerExtended(int itdc, int ibeg, int iend); /// Replay of trigger algorithm on acquired data
  int interpolate(int itdc, int ibeg, int iend);            /// Interpolation of samples to evaluate signal amplitude and arrival time
  int fullInterpolation(int itdc, int ibeg, int iend);      /// Interpolation of samples
  void correctTDCPile();                                    /// Correction of pile-up in TDC
  bool mLowPassFilter = true;                               /// Enable low pass filtering
  bool mLowPassFilterSet = false;                           /// Low pass filtering set via function call
  bool mFullInterpolation = false;                          /// Full waveform interpolation
  bool mFullInterpolationSet = false;                       /// Full waveform interpolation set via function call
  int mInterpolationStep = 25;                              /// Coarse interpolation step
  bool mCorrSignal = true;                                  /// Enable TDC signal correction
  bool mCorrSignalSet = false;                              /// TDC signal correction set via function call
  bool mCorrBackground = true;                              /// Enable TDC pile-up correction
  bool mCorrBackgroundSet = false;                          /// TDC pile-up correction set via function call
  bool mInError = false;                                    /// ZDC reconstruction ends in error
  int mAssignedTDC[NTDCChannels] = {0};                     /// Number of assigned TDCs in sequence (debugging)

  int correctTDCSignal(int itdc, int16_t TDCVal, float TDCAmp, float& fTDCVal, float& fTDCAmp, bool isbeg, bool isend); /// Correct TDC single signal
  int correctTDCBackground(int ibc, int itdc, std::deque<DigiRecoTDC>& tdc);                                            /// TDC amplitude and time corrections due to pile-up from previous bunches

  O2_ZDC_DIGIRECO_FLT getPoint(int itdc, int ibeg, int iend, int i); /// Interpolation for current TDC
  void setPoint(int itdc, int ibeg, int iend, int i);                /// Interpolation for current TDC

  void assignTDC(int ibun, int ibeg, int iend, int itdc, int tdc, float amp); /// Set reconstructed TDC values
  void findSignals(int ibeg, int iend);                                       /// Find signals around main-main that satisfy condition on TDC
  const RecoParamZDC* mRopt = nullptr;
  bool mIsContinuous = true;                     /// continuous (self-triggered) or externally-triggered readout
  uint8_t mTriggerCondition = 0x7;               /// Trigger condition: 0x1 single, 0x3 double and 0x7 triple
  int mNBCAHead = 0;                             /// when storing triggered BC, store also mNBCAHead BCs
  const ZDCTDCParam* mTDCParam = nullptr;        /// TDC calibration object
  const ZDCTDCCorr* mTDCCorr = nullptr;          /// TDC correction coefficients
  const ZDCEnergyParam* mEnergyParam = nullptr;  /// Energy calibration object
  const ZDCTowerParam* mTowerParam = nullptr;    /// Tower calibration object
  const BaselineParam* mPedParam = nullptr;      /// Tower calibration object
  uint32_t mTriggerMask = 0;                     /// Mask of triggering channels
  uint32_t mTDCMask[NTDCChannels] = {0};         /// Identify TDC channels in trigger pattern
  uint32_t mChMask[NChannels] = {0};             /// Identify all channels in readout pattern
  const RecoConfigZDC* mRecoConfigZDC = nullptr; /// CCDB configuration parameters
  int32_t mVerbosity = DbgMinimal;
  O2_ZDC_DIGIRECO_FLT mTS[NTS];                     /// Tapered sinc function
  bool mTreeDbg = false;                            /// Write reconstructed data in debug output file
  std::unique_ptr<TFile> mDbg = nullptr;            /// Debug output file
  std::unique_ptr<TTree> mTDbg = nullptr;           /// Debug tree
  gsl::span<const o2::zdc::OrbitData> mOrbitData;   /// Reconstructed data
  gsl::span<const o2::zdc::BCData> mBCData;         /// BC info
  gsl::span<const o2::zdc::ChannelData> mChData;    /// Payload
  std::vector<o2::zdc::RecEventAux> mReco;          /// Reconstructed data
  std::map<uint32_t, int> mOrbit;                   /// Information about orbit
  float mOffset[NChannels];                         /// Offset in current orbit
  uint32_t mOffsetOrbit = 0xffffffff;               /// Current orbit
  uint8_t mSource[NChannels];                       /// Source of pedestal
  static constexpr int mNSB = TSN * NTimeBinsPerBC; /// Total number of interpolated points per bunch crossing
  RecEventAux mRec;                                 /// Debug reconstruction event
  int mNBC = 0;
  int mNLonely = 0;
  int mLonely[o2::constants::lhc::LHCMaxBunches] = {0};
  int mLonelyTrig[o2::constants::lhc::LHCMaxBunches] = {0};
  uint32_t mMissingPed[NChannels] = {0};
  int16_t tdc_shift[NTDCChannels] = {0};                          /// TDC correction (units of 1/96 ns)
  float tdc_calib[NTDCChannels] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; /// TDC correction factor
  constexpr static uint16_t mMask[NTimeBinsPerBC] = {0x0001, 0x002, 0x004, 0x008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800};
  O2_ZDC_DIGIRECO_FLT mAlpha = 3; // Parameter of interpolation function
  // Configuration of interpolation for current TDC
  int mNbun;  // Number of adjacent bunches
  int mNsam;  // Number of acquired samples
  int mNtot;  // Total number of points in the interpolated arrays
  int mIlast; // Index of last acquired sample
  int mNint;  // Total points in the interpolation region (-1)
  O2_ZDC_DIGIRECO_FLT mFirstSample;
  O2_ZDC_DIGIRECO_FLT mLastSample;
};
} // namespace zdc
} // namespace o2
#endif
