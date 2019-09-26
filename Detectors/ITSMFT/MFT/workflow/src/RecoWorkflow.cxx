// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecoWorkflow.cxx

#include "MFTWorkflow/RecoWorkflow.h"

#include "MFTWorkflow/DigitReaderSpec.h"
#include "MFTWorkflow/ClustererSpec.h"
#include "MFTWorkflow/ClusterWriterSpec.h"
#include "MFTWorkflow/ClusterReaderSpec.h"
#include "MFTWorkflow/TrackerSpec.h"
#include "MFTWorkflow/TrackWriterSpec.h"

namespace o2
{
namespace mft
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC)
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::mft::getDigitReaderSpec(useMC));
  specs.emplace_back(o2::mft::getClustererSpec(useMC));
  specs.emplace_back(o2::mft::getClusterWriterSpec(useMC));

  //specs.emplace_back(o2::mft::getClusterReaderSpec(useMC));
  specs.emplace_back(o2::mft::getTrackerSpec(useMC));
  specs.emplace_back(o2::mft::getTrackWriterSpec(useMC));

  return specs;
}

} // namespace RecoWorkflow

} // namespace mft
} // namespace o2
