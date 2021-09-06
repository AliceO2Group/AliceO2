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
  // EMC aliases and values for sim

  //DOUBLE type
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"EMC_PT_[00..83]/Temperature", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"EMC_PT_[88..91]/Temperature", 100, 150.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"EMC_PT_[96..159]/Temperature", 200, 250.});

  // UINT type
  //FEE CFG aliases
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_DDL_LIST0", 0x55555555, 0x55555555});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_DDL_LIST1", 0x2AAA, 0x2AAA});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_SRU[00..09]_CFG", 1, 1});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_SRU[10..19]_CFG", 2, 2});
  //TRU aliases
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_SRU[00..19]_FMVER", 0xF0F0F0F0, 0xF0F0F0F0});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_PEAKFINDER", 4, 4});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_L0ALGSEL", 5, 5});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_COSMTHRESH", 6, 6});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_GLOBALTHRESH", 7, 7});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK0", 8, 8});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK1", 9, 9});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK2", 10, 10});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK3", 11, 11});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK4", 12, 12});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_TRU[00..45]_MASK5", 13, 13});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"EMC_STU_ERROR_COUNT_TRU[0..67]", 1000, 1000}); // not implemented in EMC DCS processor yet
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint32_t>{"DMC_STU_ERROR_COUNT_TRU[0..55]", 2000, 2000}); // not implemented in EMC DCS processor yet

  //INT type
  // EMCAL STU aliases
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_PATCHSIZE", 101, 101});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GETRAW", 102, 102});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_REGION", 0x103, 0x103});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_MEDIAN", 104, 104});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_FWVERS", 0x105, 0x105});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GA0", 106, 106});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GB0", 107, 107});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GC0", 108, 108});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GA1", 109, 109});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GB1", 110, 110});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_GC1", 111, 111});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JA0", 112, 112});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JB0", 113, 113});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JC0", 114, 114});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JA1", 115, 115});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JB1", 116, 116});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"EMC_STU_JC1", 117, 117});
  // DCAL STU aliases
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_PATCHSIZE", 201, 201});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GETRAW", 202, 202});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_REGION", 0x203, 0x203});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_MEDIAN", 204, 204});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_FWVERS", 0x205, 0x205});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GA0", 206, 206});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GB0", 207, 207});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GC0", 208, 208});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GA1", 209, 209});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GB1", 210, 210});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_GC1", 211, 211});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JA0", 212, 212});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JB0", 213, 213});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JC0", 214, 214});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JA1", 215, 215});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JB1", 216, 216});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_JC1", 217, 217});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"DMC_STU_PHOS_scale[0..3]", 218, 218});

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "EMC"));
  return specs;
}
