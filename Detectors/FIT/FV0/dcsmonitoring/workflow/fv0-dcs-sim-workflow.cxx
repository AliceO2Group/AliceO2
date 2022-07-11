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

/// \file fv0-dcs-sim-workflow.cxx
/// \brief Simulate DCS data for FV0
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;
  // for testing, we use less DPs than the official ones
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FV0/HV/SA[1..2]/actual/iMon", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FV0/PM/SA[1..2]/actual/ADC0_BASELINE", 30, 150});
  // Official list
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FV0/HV/S[A,B,C,D,E,F,G,H][1..4]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FV0/HV/S[A,B,C,D,E,F,G,H][51,52]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FV0/HV/SREF/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FV0/PM/S[A,B,C,D,E,F,G,H][1..4]/actual/ADC[0,1]_BASELINE", 30, 150});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FV0/PM/S[A,B,C,D,E,F,G,H][51,52]/actual/ADC[0,1]_BASELINE", 30, 150});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FV0/PM/SREF/actual/ADC[0,1]_BASELINE", 30, 150});

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "FV0"));
  return specs;
}
