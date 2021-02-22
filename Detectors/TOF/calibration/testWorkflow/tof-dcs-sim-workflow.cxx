// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// // we need to add workflow options before including Framework/runDataProcessing
// void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
// {
//   // option allowing to set parameters
// }

// ------------------------------------------------------------------

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;
  // for TOF
  // for test, we use less DPs that official ones
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_vp_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_vn_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_ip_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_in_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"TOF_FEACSTATUS_[00..01]", 0, 255});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"TOF_HVSTATUS_SM[00..01]MOD[0..1]", 0, 524287});
  // for TOF, official list
  //mTOFDataPointHints.emplace_back(DataPointHint<double>{"tof_hv_vp_[00..89]", 0, 50.});
  //mTOFDataPointHints.emplace_back(DataPointHint<double>{"tof_hv_vn_[00..89]", 0, 50.});
  //mTOFDataPointHints.emplace_back(DataPointHint<double>{"tof_hv_ip_[00..89]", 0, 50.});
  //mTOFDataPointHints.emplace_back(DataPointHint<double>{"tof_hv_in_[00..89]", 0, 50.});
  //mTOFDataPointHints.emplace_back(DataPointHint<int32_t>{"TOF_FEACSTATUS_[00..71]", 0, 255});
  //mTOFDataPointHints.emplace_back(DataPointHint<int32_t>{"TOF_HVSTATUS_SM[00..17]MOD[0..4]", 0, 524287});
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "TOF"));
  return specs;
}
