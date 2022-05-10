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

#include <mutex>
#include <memory>
#include <TH1.h>
#include <TH2.h>
#include <THnBase.h>
#include <THnSparse.h>
#include <TMinuit.h>
#include "ZDCBase/Constants.h"
#include "CommonDataFormat/FlatHisto1D.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/WaveformCalibData.h"
#include "DataFormatsZDC/ZDCWaveform.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include "ZDCCalib/WaveformCalibQueue.h"
#ifndef ALICEO2_ZDC_WAVEFORMCALIBEPN_H
#define ALICEO2_ZDC_WAVEFORMCALIBEPN_H
namespace o2
{
namespace zdc
{
class WaveformCalibEPN
{
 public:
  WaveformCalibEPN() = default;
  int init();
  static constexpr int NH = WaveformCalibConfig::NH;
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info,
              const gsl::span<const o2::zdc::ZDCWaveform>& wave);
  int endOfRun();
  int write(const std::string fn = "ZDCWaveformCalibEPN.root");
  void setConfig(const WaveformCalibConfig* param) { mWaveformCalibConfig = param; };
  const WaveformCalibConfig* getConfig() const { return mWaveformCalibConfig; };
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }
  std::array<o2::dataformats::FlatHisto1D<float>*, NH> mH{};
  WaveformCalibData mData;
  WaveformCalibData& getData() { return mData; }

 private:
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mNBin = 0;
  int32_t mVerbosity = DbgMinimal;
  const WaveformCalibConfig* mWaveformCalibConfig = nullptr; /// Configuration of intercalibration

  int mFirst=0;
  int mLast=0;
  int mN=1;

  void configure(int ifirst, int ilast){
    if(ifirst>0 || ilast<0 || ilast<ifirst){
      LOGF(fatal,"WaveformCalibEPN configure error with ifirst=%d ilast=%d", ifirst, ilast);
    }
    mFirst=ifirst;
    mLast=ilast;
    mN=ilast-ifirst+1;
  }

  
  
    WaveformCalibQueue mQueue;
};
} // namespace zdc
} // namespace o2

#endif
