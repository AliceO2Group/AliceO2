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

/// \file CorrectionMapsHelper.h
/// \brief Helper class to access correction maps from CCDB
/// \author ruben.shahoian@cern.ch

#ifndef TPC_CORRECTION_MAPS_HELPER_H_
#define TPC_CORRECTION_MAPS_HELPER_H_

#include <memory>
#include <vector>

namespace o2
{
namespace framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
} // namespace framework

namespace gpu
{
class TPCFastTransform;
}

namespace tpc
{

class CorrectionMapsHelper
{
 public:
  o2::gpu::TPCFastTransform* getCorrMap() { return mCorrMap.get(); }
  o2::gpu::TPCFastTransform* getCorrMapRef() { return mCorrMapRef.get(); }

  void setCorrMap(std::unique_ptr<o2::gpu::TPCFastTransform> m) { mCorrMap = std::move(m); }
  void setCorrMapRef(std::unique_ptr<o2::gpu::TPCFastTransform> m) { mCorrMapRef = std::move(m); }

  void adoptCorrMap(o2::gpu::TPCFastTransform* m) { mCorrMap.reset(m); }
  void adoptCorrMapRef(o2::gpu::TPCFastTransform* m) { mCorrMapRef.reset(m); }

  bool isUpdated() const { return mUpdated; }
  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);
  void acknowledgeUpdate() { mUpdated = false; }

  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs);
  static void extractCCDBInputs(o2::framework::ProcessingContext& pc);

 protected:
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);
  bool mUpdated = false;
  std::unique_ptr<o2::gpu::TPCFastTransform> mCorrMap{};    // current transform
  std::unique_ptr<o2::gpu::TPCFastTransform> mCorrMapRef{}; // reference transform
};

} // namespace tpc

} // namespace o2

#endif
