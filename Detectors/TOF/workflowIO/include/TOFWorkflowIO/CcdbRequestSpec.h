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

/// @file   CcdbRequestSpec.h

#ifndef O2_TOF_CCDBREQUEST
#define O2_TOF_CCDBREQUEST

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CcdbRequest : public Task
{
 public:
  CcdbRequest(std::shared_ptr<o2::base::GRPGeomRequest> gr) : mGGCCDBRequest(gr) {}
  ~CcdbRequest() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

/// create a processor spec
framework::DataProcessorSpec getCcdbRequestSpec();

} // namespace tof
} // namespace o2

#endif /* O2_TOF_CCDBREQUEST */
