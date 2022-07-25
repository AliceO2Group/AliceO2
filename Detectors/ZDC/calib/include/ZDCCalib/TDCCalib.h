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
#include <TMath.h>
#include "ZDCBase/Constants.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCCalib/TDCCalibConfig.h"
#include "CCDB/CcdbObjectInfo.h" //added by me
#include "CommonDataFormat/FlatHisto1D.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCReconstruction/ZDCTowerParam.h"

#ifndef ALICEO2_ZDC_TDCCALIB_H_
#define ALICEO2_ZDC_TDCCALIB_H_
namespace o2
{
namespace zdc
{

class TDCCalib //after
{

  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo; //added by me

 public:
  TDCCalib() = default;
  int init();
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info); // Calibration of RUN3 data
  int process(const TDCCalibData& data);              // Calibration of RUN3 data - aggregator node
  int endOfRun();                                     // Perform minimization
  double extractShift(int ih);
  void add(int ih, o2::dataformats::FlatHisto1D<float>& h1);
  int write(const std::string fn = "ZDCTDCCalib.root");

  const ZDCTDCParam& getTDCParamUpd() const { return mTDCParamUpd; };
  CcdbObjectInfo& getCcdbObjectInfo() { return mInfo; }

  void setTDCParam(const ZDCTDCParam* param) { mTDCParam = param; };
  const ZDCTDCParam* getTDCParam() const { return mTDCParam; };
  void setTDCCalibConfig(const TDCCalibConfig* param) { mTDCCalibConfig = param; };
  const TDCCalibConfig* getTDCCalibConfig() const { return mTDCCalibConfig; };

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

 private:
  std::array<o2::dataformats::FlatHisto1D<float>*, NTDCChannels> mCTDC{}; //array of FlatHisto1D, number of elements = NTDCChannles (= 10), defined in constants.h {} means defined but not initialized
  std::array<std::unique_ptr<TH1>, NTDCChannels> mHCTDC{};                //copy of flat histo 1D in TH1F to use root functions
  bool mInitDone = false;
  bool mSaveDebugHistos = true;
  const TDCCalibConfig* mTDCCalibConfig = nullptr; /// Configuration of TDC calibration, this line has been swapped with the following one to be consistent with intercalibration
  const ZDCTDCParam* mTDCParam = nullptr;          /// TDC calibration object
  int32_t mVerbosity = DbgMinimal;

  TDCCalibData mData;
  ZDCTDCParam mTDCParamUpd;        /// Updated TDC calibration object, added by me
  CcdbObjectInfo mInfo;            /// CCDB Info, added by me
  void assign(int ih, bool ismod); /// Assign updated calibration object, added by me
};
} // namespace zdc
} // namespace o2

#endif
