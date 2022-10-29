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
#include "ZDCReconstruction/ZDCTDCParam.h" //added by me
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/TDCCalibConfig.h"

#ifndef ALICEO2_ZDC_TDCCALIBEPN_H_
#define ALICEO2_ZDC_TDCCALIBEPN_H_
namespace o2
{
namespace zdc
{
class TDCCalibEPN
{
 public:
  TDCCalibEPN() = default;
  int init();
  static constexpr int HtdcZNAC = 0;
  static constexpr int HtdcZNAS = 1;
  static constexpr int HtdcZPAC = 2;
  static constexpr int HtdcZPAS = 3;
  static constexpr int HtdcZEM1 = 4;
  static constexpr int HtdcZEM2 = 5;
  static constexpr int HtdcZNCC = 6;
  static constexpr int HtdcZNCS = 7;
  static constexpr int HtdcZPCC = 8;
  static constexpr int HtdcZPCS = 9;
  static constexpr int NTDC = TDCCalibData::NTDC;
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info); // Calibration of RUN3 data
  int endOfRun();                                     // End of TDCCalib
  int write(const std::string fn = "ZDCTDCCalibEPN.root");

  void fill1D(int iTDC, int nHits, o2::zdc::RecEventFlat ev); //function to fill histograms;

  void setTDCCalibConfig(const TDCCalibConfig* param) { mTDCCalibConfig = param; };
  const TDCCalibConfig* getTDCCalibConfig() const { return mTDCCalibConfig; };
  void setSaveDebugHistos() { mSaveDebugHistos = true; }
  void setDontSaveDebugHistos() { mSaveDebugHistos = false; }
  void setVerbosity(int val) { mVerbosity = val; }
  TDCCalibData mData;
  TDCCalibData& getData() { return mData; }
  std::array<o2::dataformats::FlatHisto1D<float>*, NTDC> mTDC{};

 private:
  bool mInitDone = false;
  bool mSaveDebugHistos = false;
  int32_t mVerbosity = DbgMinimal;
  const TDCCalibConfig* mTDCCalibConfig = nullptr; /// Configuration of intercalibration
};
} // namespace zdc
} // namespace o2

#endif
