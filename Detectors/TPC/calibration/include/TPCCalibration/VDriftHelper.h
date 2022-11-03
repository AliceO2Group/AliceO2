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

/// \file VDriftHelper.h
/// \brief Helper class to extract VDrift from different sources
/// \author ruben.shahoian@cern.ch

#ifndef TPC_VDRIFT_HELPER_H_
#define TPC_VDRIFT_HELPER_H_

#include "GPUCommonRtypes.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include <array>
#include <vector>
#include <string_view>

namespace o2::framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
} // namespace o2::framework

namespace o2::tpc
{
class LtrCalibData;

class VDriftHelper
{
 public:
  enum Source : int { GasParam,
                      Laser,
                      ITSTPCTgl,
                      NSources
  };
  static constexpr std::array<std::string_view, NSources> SourceNames = {
    "GasParam",
    "Laser",
    "TPCITSTgl"};

  VDriftHelper();
  void accountLaserCalibration(const LtrCalibData* calib, long fallBackTimeStamp = 2);
  void accountDriftCorrectionITSTPCTgl(const VDriftCorrFact* calib);
  bool isUpdated() const { return mUpdated; }
  void acknowledgeUpdate() { mUpdated = false; }

  const VDriftCorrFact& getVDriftObject() const { return mVD; }
  Source getSource() const { return mSource; }
  std::string_view getSourceName() const { return SourceNames[mSource]; }

  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);
  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs, bool laser = true, bool itstpcTgl = true);
  static void extractCCDBInputs(o2::framework::ProcessingContext& pc, bool laser = true, bool itstpcTgl = true);

 protected:
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);

  VDriftCorrFact mVD{};
  Source mSource{};       // update source
  bool mUpdated = false;  // signal update, must be reset once new value is fetched
  uint32_t mMayRenormSrc = 0xffffffff; // if starting VDrift correction != 1, we will renorm reference in such a way that initial correction is 1.0, flag per source

  ClassDefNV(VDriftHelper, 1);
};
} // namespace o2::tpc
#endif
