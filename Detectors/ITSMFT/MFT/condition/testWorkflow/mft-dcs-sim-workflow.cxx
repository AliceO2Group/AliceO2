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
  // for MFT
  // for test, we use less DPs that official ones

  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.Analog", 0.05, 0.3});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.Digital", 0.1, 2.0});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.BackBias", 0.0, 1.0});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Voltage.BackBias", 0, 3.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel008.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel009.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel010.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard00/channel011.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard05/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel008.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel009.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel010.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard10/channel011.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate0/easyBoard15/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel008.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel009.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel010.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard00/channel011.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard05/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel007.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel008.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel009.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel010.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard10/channel011.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel000.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel001.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel002.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel003.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel004.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel005.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel006.actual.iMon", 1.8, 2.2});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"mft_infra:CAEN/alimftcae001/branchController11/easyCrate1/easyBoard15/channel007.actual.iMon", 1.8, 2.2});

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "MFT"));
  return specs;
}
