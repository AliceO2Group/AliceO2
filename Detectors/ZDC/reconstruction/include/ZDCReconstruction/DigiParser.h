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

#ifndef ALICEO2_ZDC_DIGI_PARSER_H
#define ALICEO2_ZDC_DIGI_PARSER_H

namespace o2
{
namespace zdc
{

class DigiParser
{
 public:
  DigiParser() = default;
  ~DigiParser() = default;
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
  void eor();

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

private:
  const ModuleConfig* mModuleConfig = nullptr;              /// Trigger/readout configuration object

  const RecoParamZDC* mRopt = nullptr;
  int mNBCAHead = 0;                             /// when storing triggered BC, store also mNBCAHead BCs
  uint32_t mTriggerMask = 0;                     /// Mask of triggering channels
  const RecoConfigZDC* mRecoConfigZDC = nullptr; /// CCDB configuration parameters
  int32_t mVerbosity = DbgMinimal;
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
  constexpr static uint16_t mMask[NTimeBinsPerBC] = {0x0001, 0x002, 0x004, 0x008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800};
  // Configuration of interpolation for current TDC
  int mNbun;  // Number of adjacent bunches
  int mNsam;  // Number of acquired samples
  int mNtot;  // Total number of points in the interpolated arrays
  int mIlast; // Index of last acquired sample
  int mNint;  // Total points in the interpolation region (-1)
};
} // namespace zdc
} // namespace o2
#endif
