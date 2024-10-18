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
#include <TH1.h>
#include <TH2.h>
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
  void setVerbosity(int v)
  {
    mVerbosity = v;
  }
  void setOutput(std::string output){
    mOutput = output;
  }
  int getVerbosity() const { return mVerbosity; }
  void eor();

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

private:
  const ModuleConfig* mModuleConfig = nullptr;              /// Trigger/readout configuration object
  const RecoParamZDC* mRopt = nullptr;

  void setStat(TH1* h);
  void setModuleLabel(TH1* h);

  int32_t mVerbosity = DbgMinimal;
  bool mRejectPileUp = true;
  std::string mOutput = "ZDCDigiParser.root";
  uint32_t mTriggerMask = 0;                     /// Mask of triggering channels
  uint32_t mTDCMask[NTDCChannels] = {0};         /// Identify TDC channels in trigger pattern
  uint32_t mChMask[NChannels] = {0};             /// Identify all channels in readout pattern

  std::unique_ptr<TH1> mTransmitted = nullptr;
  std::unique_ptr<TH1> mFired = nullptr;
  std::unique_ptr<TH1> mBaseline[NChannels] = {nullptr};
  std::unique_ptr<TH1> mCounts[NChannels] = {nullptr};
  std::unique_ptr<TH2> mSignalTH[NChannels] = {nullptr};
  std::unique_ptr<TH2> mBunchH[NChannels] = {nullptr}; // Bunch pattern Hit

  int mNBC = 0;
};
} // namespace zdc
} // namespace o2
#endif
