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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

#include "TRDBase/TrackletTransformer.h"

namespace o2
{
namespace trd
{

class TRDTrackletTransformerSpec : public o2::framework::Task
{
 public:
  TRDTrackletTransformerSpec(std::shared_ptr<o2::globaltracking::DataRequest> dataRequest, bool trigRecFilterActive, bool applyXOR) : mDataRequest(dataRequest), mTrigRecFilterActive(trigRecFilterActive), mApplyXOR(applyXOR){};
  ~TRDTrackletTransformerSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
  bool mTrigRecFilterActive; ///< if true, transform only TRD tracklets for which ITS data is available
  bool mApplyXOR;            ///< if true, the 8-th bit of position and slope will be inverted before transformation in chamber coordinates
  TrackletTransformer mTransformer;
};

o2::framework::DataProcessorSpec getTRDTrackletTransformerSpec(bool trigRecFilterActive, bool applyXOR);

} // end namespace trd
} // end namespace o2
