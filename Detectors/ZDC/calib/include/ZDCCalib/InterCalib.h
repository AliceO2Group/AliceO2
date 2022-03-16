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
  void clear();
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info);
  // Test of calibration using RUN2 data
  int process(const char* hname, int ic);
  int mini();
  static constexpr int NPAR = 6; /// Dimension of matrix (1 + 4 coefficients + offset)
  static constexpr int NH = 5;   /// ZNA, ZPA, ZNC, ZPC, ZEM
  static double add[NPAR][NPAR];
  static void fcn(int& npar, double* gin, double& chi, double* par, int iflag);
  void cumulate(double tc, double t1, double t2, double t3, double t4, double w);

  void setEnergyParam(const ZDCEnergyParam* param) { mEnergyParam = param; };
  const ZDCEnergyParam* getEnergyParam() { return mEnergyParam; };
  void setTowerParam(const ZDCTowerParam* param) { mTowerParam = param; };
  const ZDCTowerParam* getTowerParam() { return mTowerParam; };

 private:
  TH1* hb[NH] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  TH1* ha[NH] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  static std::mutex mtx; /// mutex for critical section
  double sum[NPAR][NPAR] = {0};
  double par[NPAR] = {0};
  double err[NPAR] = {0};
  double mCutLow = -std::numeric_limits<float>::infinity();
  double mCutHigh = std::numeric_limits<float>::infinity();
  const ZDCEnergyParam* mEnergyParam = nullptr; /// Energy calibration object
  const ZDCTowerParam* mTowerParam = nullptr;   /// Tower calibration object
  std::vector<float> store[5];
};
} // namespace zdc
} // namespace o2

#endif
