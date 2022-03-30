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
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/InterCalibData.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibConfig.h"
#ifndef ALICEO2_ZDC_INTERCALIB_H_
#define ALICEO2_ZDC_INTERCALIB_H_
namespace o2
{
namespace zdc
{
class InterCalib
{
 public:
  InterCalib() = default;
  int init();
  static constexpr int HidZNA = 0;
  static constexpr int HidZPA = 1;
  static constexpr int HidZNC = 2;
  static constexpr int HidZPC = 3;
  static constexpr int HidZEM = 4;
  static constexpr int NH = InterCalibData::NH;
  static constexpr int NPAR = InterCalibData::NPAR;
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info); // Calibration of RUN3 data - direct
  int process(const InterCalibData &data);            // Calibration of RUN3 data - aggregator node
  int endOfRun();                                     // Perform minimization
  int process(const char* hname, int ic);             // Calibration of RUN2 data
  void replay(int ih, THnSparse* hs, int ic);         // Test of calibration using RUN2 data
  int mini(int ih);
  int write(const std::string fn = "ZDCInterCalib.root");

  static double mAdd[NPAR][NPAR]; /// Temporary copy of cumulated sums
  static void fcn(int& npar, double* gin, double& chi, double* par, int iflag);
  void cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w);

  void setEnergyParam(const ZDCEnergyParam* param) { mEnergyParam = param; };
  const ZDCEnergyParam* getEnergyParam() const { return mEnergyParam; };
  void setTowerParam(const ZDCTowerParam* param) { mTowerParam = param; };
  const ZDCTowerParam* getTowerParam() const { return mTowerParam; };
  void setInterCalibConfig(const InterCalibConfig* param) { mInterCalibConfig = param; };
  const InterCalibConfig* getInterCalibConfig() const { return mInterCalibConfig; };

 private:
  std::array<std::unique_ptr<TH1>, 2 * NH> mHUnc{};
  std::array<std::unique_ptr<TH2>, NH> mCUnc{};
  std::array<std::unique_ptr<TH1>, NH> mHCorr{};
  std::array<std::unique_ptr<TH2>, NH> mCCorr{};
  std::array<std::unique_ptr<TMinuit>, NH> mMn{};
  InterCalibData mData;
  bool mInitDone = false;
  static std::mutex mMtx; /// mutex for critical section
  double mPar[NH][NPAR] = {0};
  double mErr[NH][NPAR] = {0};
  const ZDCEnergyParam* mEnergyParam = nullptr;        /// Energy calibration object
  const ZDCTowerParam* mTowerParam = nullptr;          /// Tower calibration object
  const InterCalibConfig* mInterCalibConfig = nullptr; /// Configuration of intercalibration
};
} // namespace zdc
} // namespace o2

#endif
