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

#include "FITWorkflow/FITIntegrateClusterReaderSpec.h"
#include "DataFormatsFDD/RecPoint.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  wf.emplace_back(o2::fit::getFITIntegrateClusterReaderSpec<o2::fdd::RecPoint>());
  return wf;
}
