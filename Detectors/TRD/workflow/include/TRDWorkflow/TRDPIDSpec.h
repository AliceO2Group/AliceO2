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

/// \file TRDPIDSpec.h
/// \brief This file provides the specification for the trd pid workflow.
/// \author Felix Schlepper

#ifndef O2_TRD_PIDSPEC_H
#define O2_TRD_PIDSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GRPGeomHelper.h"

namespace o2
{
namespace trd
{

class TRDPIDSpec : public o2::framework::Task
{
 public:
  TRDPIDSpec(std::shared_ptr<o2::globaltracking::DataRequest> dataRequest, std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dataRequest), mGGCCDBRequest(gr){};
  ~TRDPIDSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

o2::framework::DataProcessorSpec getTRDPIDSpec(o2::dataformats::GlobalTrackID::mask_t src);

} // end namespace trd
} // end namespace o2

#endif // O2_TRD_PIDSPEC_H
