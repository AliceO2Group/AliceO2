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
#include <TH1.h>
#include <TH2.h>
#include <TMinuit.h>
#include "ZDCBase/Constants.h"
#include "DataFormatsZDC/RecEvent.h"
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
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info);
  // Test of calibration using RUN2 data
  int process(const char* hname, int ic);
  int mini(int ih);
  static constexpr int NPAR = 6; /// Dimension of matrix (1 + 4 coefficients + offset)
  static constexpr int NH = 5;   /// ZNA, ZPA, ZNC, ZPC, ZEM
  static double add[NPAR][NPAR]; /// Temporary copy of cumulated sums
  static void fcn(int& npar, double* gin, double& chi, double* par, int iflag);
  void cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w);

  void setEnergyParam(const ZDCEnergyParam* param) { mEnergyParam = param; };
  const ZDCEnergyParam* getEnergyParam() { return mEnergyParam; };
  void setTowerParam(const ZDCTowerParam* param) { mTowerParam = param; };
  const ZDCTowerParam* getTowerParam() { return mTowerParam; };
  void setInterCalibConfig(const InterCalibConfig* param) { mInterCalibConfig = param; };
  const InterCalibConfig* getInterCalibConfig() { return mInterCalibConfig; };

 private:
  TH1* h[2*NH] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  TH2* hc[NH] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  bool mInitDone = false;
  static std::mutex mtx; /// mutex for critical section
  double sum[NH][NPAR][NPAR] = {0};
  double par[NH][NPAR] = {0};
  double err[NH][NPAR] = {0};
  const ZDCEnergyParam* mEnergyParam = nullptr;        /// Energy calibration object
  const ZDCTowerParam* mTowerParam = nullptr;          /// Tower calibration object
  const InterCalibConfig* mInterCalibConfig = nullptr; /// Configuration of intercalibration
  std::vector<float> store[5];
};
} // namespace zdc
} // namespace o2

#endif
