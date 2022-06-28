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
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCCalib/InterCalibConfig.h"
#ifndef ALICEO2_ZDC_TDCCALIB_H_
#define ALICEO2_ZDC_TDCCALIB_H_
namespace o2
{
namespace zdc
{
class InterCalib
{
 public:
  TDCCalib() = default;
  int init();
  void clear(int ih = -1);
  int process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info); // Calibration of RUN3 data
  int endOfRun();                                     // Perform minimization
  int process(const char* hname, int ic);             // Calibration of RUN2 data
  void replay(int ih, THnSparse* hs, int ic);         // Test of calibration using RUN2 data
  int mini(int ih);
  int write(const std::string fn = "ZDCTDCCalib.root");

  static void fcn(int& npar, double* gin, double& chi, double* par, int iflag);

  void setTDCParam(const ZDCTDCParam* param) { mTDCParam = param; };
  const ZDCTDCParam* getTDCParam() const { return mTDCParam; };
  void setTDCCalibConfig(const TDCCalibConfig* param) { mTDCCalibConfig = param; };
  const TDCCalibConfig* getTDCCalibConfig() const { return mTDCCalibConfig; };

 private:
  std::array<std::unique_ptr<TH1>, NTDCChannels> mHTDC{};
  std::array<std::unique_ptr<TMinuit>, NTDCChannels> mMn{};
  bool mInitDone = false;
  static std::mutex mMtx;                          /// mutex for critical section
  const ZDCTDCParam* mTDCParam = nullptr;          /// TDC calibration object
  const TDCCalibConfig* mTDCCalibConfig = nullptr; /// Configuration of TDC calibration
};
} // namespace zdc
} // namespace o2

#endif
