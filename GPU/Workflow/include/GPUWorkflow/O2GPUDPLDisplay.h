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

#ifndef O2_GPU_DPL_DISPLAY_H
#define O2_GPU_DPL_DISPLAY_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/Task.h"
#include <memory>

namespace o2::trd
{
class GeometryFlat;
}
namespace o2::globaltracking
{
struct DataRequest;
}
namespace o2::itsmft
{
class TopologyDictionary;
}

namespace o2::gpu
{
class GPUO2InterfaceDisplay;
struct GPUO2InterfaceConfiguration;
class TPCFastTransform;
struct GPUSettingsTF;

class O2GPUDPLDisplaySpec : public o2::framework::Task
{
 public:
  O2GPUDPLDisplaySpec(bool useMC, o2::dataformats::GlobalTrackID::mask_t trkMask, o2::dataformats::GlobalTrackID::mask_t clMask, std::shared_ptr<o2::globaltracking::DataRequest> dataRequest) : mUseMC(useMC), mTrkMask(trkMask), mClMask(clMask), mDataRequest(dataRequest) {}
  ~O2GPUDPLDisplaySpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  bool mUseMC = false;
  bool mUpdateCalib = false;
  bool mDisplayShutDown = false;
  bool mFirst = false;
  o2::dataformats::GlobalTrackID::mask_t mTrkMask;
  o2::dataformats::GlobalTrackID::mask_t mClMask;
  std::unique_ptr<GPUO2InterfaceDisplay> mDisplay;
  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
  std::unique_ptr<TPCFastTransform> mFastTransform;
  std::unique_ptr<o2::trd::GeometryFlat> mTrdGeo;
  std::unique_ptr<o2::itsmft::TopologyDictionary> mITSDict;
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
  std::unique_ptr<o2::gpu::GPUSettingsTF> mTFSettings;
};

} // namespace o2::gpu

#endif
