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
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibData.h"
#include "ZDCCalib/InterCalibConfig.h"
#ifndef ALICEO2_ZDC_INTERCALIBEPN_H_
#define ALICEO2_ZDC_INTERCALIBEPN_H_
namespace o2
{
namespace zdc
{
class InterCalibEPN
{
 public:
  InterCalibEPN() = default;
  int init();
  static constexpr int HidZNA = 0;
  static constexpr int HidZPA = 1;
  static constexpr int HidZNC = 2;
  static constexpr int HidZPC = 3;
  static constexpr int HidZEM = 4;
  static constexpr int HidZNI = 5;
  static constexpr int HidZPI = 6;
  static constexpr int NH = InterCalibData::NH;
  static constexpr int NPAR = InterCalibData::NPAR;
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info); // Calibration of RUN3 data
  int endOfRun();                                     // Perform minimization
  int process(const char* hname, int ic);             // Calibration of RUN2 data
  int saveDebugHistos(const std::string fn = "ZDCInterCalibEPN.root");
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }
  void cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w);
  void setInterCalibConfig(const InterCalibConfig* param) { mInterCalibConfig = param; };
  const InterCalibConfig* getInterCalibConfig() const { return mInterCalibConfig; };
  void setVerbosity(int val) { mVerbosity = val; }
  InterCalibData mData;
  InterCalibData& getData() { return mData; }
  std::array<o2::dataformats::FlatHisto1D<float>*, 2 * NH> mH{};
  std::array<o2::dataformats::FlatHisto2D<float>*, NH> mC{};

 private:
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mVerbosity = DbgMinimal;
  const InterCalibConfig* mInterCalibConfig = nullptr; /// Configuration of intercalibration
};
} // namespace zdc
} // namespace o2

#endif
