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

#include <memory>
#include <gsl/span>
#include "ZDCBase/Constants.h"
#include "CommonDataFormat/FlatHisto1D.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCCalib/NoiseCalibData.h"
#ifndef ALICEO2_ZDC_NOISECALIBEPN_H
#define ALICEO2_ZDC_NOISECALIBEPN_H
namespace o2
{
namespace zdc
{
class NoiseCalibEPN
{
 public:
  NoiseCalibEPN() = default;
  int init();

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata);
  int endOfRun();
  int saveDebugHistos(const std::string fn = "ZDCNoiseCalibEPN.root");
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }
  void setVerbosity(int val) { mVerbosity = val; }
  NoiseCalibData mData;
  NoiseCalibData& getData() { return mData; }
  std::array<o2::dataformats::FlatHisto1D<double>*, NChannels> mH{};

 private:
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  const ModuleConfig* mModuleConfig = nullptr; /// Trigger/readout configuration object
  const RecoParamZDC* mRopt = nullptr;
  uint32_t mChMask[NChannels] = {0}; ///< Identify all channels in readout pattern
  int32_t mVerbosity = DbgMinimal;
};
} // namespace zdc
} // namespace o2

#endif
