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
#include <gsl/span>
#include <TFile.h>
#include <TTree.h>
#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/RecEventAux.h"

#ifndef ALICEO2_ZDC_DIGI_RECO_H
#define ALICEO2_ZDC_DIGI_RECO_H
namespace o2
{
namespace zdc
{
class DigiReco
{
 public:
  DigiReco() = default;
  ~DigiReco() = default;
  void init();
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
  void eor()
  {
    if (mTreeDbg) {
      LOG(INFO) << "ZDC DigiReco: closing debug output";
      mTDbg->Write();
      mTDbg.reset();
      mDbg->Close();
      mDbg.reset();
    }
    LOG(INFO) << "Detected " << mNLonely << " lonely bunches";
    LOG(INFO) << "Detected " << mNLastLonely << " lonely bunches at end of orbit";
  }

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  void setTDCParam(const ZDCTDCParam* param) { mTDCParam = param; };
  const ZDCTDCParam* getTDCParam() { return mTDCParam; };
  void setEnergyParam(const ZDCEnergyParam* param) { mEnergyParam = param; };
  const ZDCEnergyParam* getEnergyParam() { return mEnergyParam; };
  void setTowerParam(const ZDCTowerParam* param) { mTowerParam = param; };
  const ZDCTowerParam* getTowerParam() { return mTowerParam; };
  void setRecoConfigZDC(const RecoConfigZDC* cfg) { mRecoConfigZDC = cfg; };
  const RecoConfigZDC* getRecoConfigZDC() { return mRecoConfigZDC; };

  const std::vector<o2::zdc::RecEventAux>& getReco() { return mReco; }

 private:
  const ModuleConfig* mModuleConfig = nullptr;                                /// Trigger/readout configuration object
  int reconstruct(int seq_beg, int seq_end);                                  /// Main method for data reconstruction
  void processTrigger(int itdc, int ibeg, int iend);                          /// Replay of trigger algorithm on acquired data
  void interpolate(int itdc, int ibeg, int iend);                             /// Interpolation of samples to evaluate signal amplitude and arrival time
  void assignTDC(int ibun, int ibeg, int iend, int itdc, int tdc, float amp); /// Set reconstructed TDC values
  bool mIsContinuous = true;                                                  /// continuous (self-triggered) or externally-triggered readout
  int mNBCAHead = 0;                                                          /// when storing triggered BC, store also mNBCAHead BCs
  const ZDCTDCParam* mTDCParam = nullptr;                                     /// TDC calibration object
  const ZDCEnergyParam* mEnergyParam = nullptr;                               /// Energy calibration object
  const ZDCTowerParam* mTowerParam = nullptr;                                 /// Tower calibration object
  uint32_t mTDCMask[NTDCChannels] = {0};                                      /// Identify TDC channels in trigger mask
  const RecoConfigZDC* mRecoConfigZDC = nullptr;                              /// CCDB configuration parameters
  int32_t mVerbosity = DbgMinimal;
  Double_t mTS[NTS];                                /// Tapered sinc function
  bool mTreeDbg = false;                            /// Write reconstructed data in debug output file
  std::unique_ptr<TFile> mDbg = nullptr;            /// Debug output file
  std::unique_ptr<TTree> mTDbg = nullptr;           /// Debug tree
  gsl::span<const o2::zdc::OrbitData> mOrbitData;   /// Reconstructed data
  gsl::span<const o2::zdc::BCData> mBCData;         /// BC info
  gsl::span<const o2::zdc::ChannelData> mChData;    /// Payload
  std::vector<o2::zdc::RecEventAux> mReco;          /// Reconstructed data
  std::map<uint32_t, int> mOrbit;                   /// Information about orbit
  static constexpr int mNSB = TSN * NTimeBinsPerBC; /// Total number of interpolated points per bunch crossing
  RecEventAux mRec;                                 /// Debug reconstruction event
  int mNBC = 0;
  int mNLonely = 0;
  int mNLastLonely = 0;
  int16_t tdc_shift[NTDCChannels] = {0}; /// TDC correction (units of 1/96 ns)
  constexpr static uint16_t mMask[NTimeBinsPerBC] = {0x0001, 0x002, 0x004, 0x008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800};
};
} // namespace zdc
} // namespace o2
#endif
