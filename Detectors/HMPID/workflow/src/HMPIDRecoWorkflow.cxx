// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDWorkflow/DigitReaderSpec.h"
#include "HMPIDWorkflow/ClusterizerSpec.h"

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// The standalone workflow executable for HMPID reconstruction workflow
/// - digit reader
/// - clusterer

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  bool useMC = true;
  specs.emplace_back(o2::hmpid::getDigitReaderSpec(useMC));
  specs.emplace_back(o2::hmpid::getHMPIDClusterizerSpec(useMC));

  return std::move(specs);
}
