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

/// \file   MID/Workflow/src/FilteringBCSpec.cxx
/// \brief  MID filtering spec
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 March 2022

#include "MIDWorkflow/FilteringBCSpec.h"

#include <vector>
#include <gsl/gsl>
#include <fmt/format.h>
#include "Framework/CCDBParamSpec.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDFiltering/FiltererBC.h"
#include "MIDFiltering/FiltererBCParam.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDWorkflow/ColumnDataSpecsUtils.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class FilteringBCDeviceDPL
{
 public:
  FilteringBCDeviceDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, std::vector<of::OutputSpec> outputSpecs) : mGGCCDBRequest(gr), mUseMC(useMC)
  {
    mOutputs = specs::buildOutputs(outputSpecs);
  }

  void init(of::InitContext& ic)
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    mFiltererBC.setBCDiffLow(FiltererBCParam::Instance().maxBCDiffLow);
    mFiltererBC.setBCDiffHigh(FiltererBCParam::Instance().maxBCDiffHigh);
    mFiltererBC.setSelectOnly(FiltererBCParam::Instance().selectOnly);
  }

  void finaliseCCDB(of::ConcreteDataMatcher matcher, void* obj)
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      LOG(info) << "Update Bunch Filling for MID";
      mFiltererBC.setBunchFilling(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling());
    }
  }

  void run(of::ProcessingContext& pc)
  {
    updateTimeDependentParams(pc);

    auto data = specs::getData(pc, "mid_filter_BC_in", EventType::Standard);
    auto inROFRecords = specs::getRofs(pc, "mid_filter_BC_in", EventType::Standard);

    auto inMCContainer = mUseMC ? specs::getLabels(pc, "mid_filter_BC_in") : nullptr;

    auto filteredROFs = mFiltererBC.process(inROFRecords);
    mDigitsMerger.process(data, filteredROFs, inMCContainer.get());

    pc.outputs().snapshot(mOutputs[0], mDigitsMerger.getColumnData());
    pc.outputs().snapshot(mOutputs[1], mDigitsMerger.getROFRecords());
    if (mUseMC) {
      pc.outputs().snapshot(mOutputs[2], mDigitsMerger.getMCContainer());
    }
  }

 private:
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc)
  {
    // Triggers finalizeCCDB
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  }

  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC{false};
  FiltererBC mFiltererBC;
  DigitsMerger mDigitsMerger;
  std::vector<of::Output> mOutputs;
};

of::DataProcessorSpec getFilteringBCSpec(bool useMC, std::string_view inDesc)
{

  auto inputSpecs = specs::buildInputSpecs("mid_filter_BC_in", inDesc, useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              true,                           // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputSpecs,
                                                              false);

  auto outputSpecs = specs::buildStandardOutputSpecs("mid_filter_BC_out", "BDATA", useMC);

  return of::DataProcessorSpec{
    "MIDFilteringBC",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::FilteringBCDeviceDPL>(ggRequest, useMC, outputSpecs)}};
}
} // namespace mid
} // namespace o2