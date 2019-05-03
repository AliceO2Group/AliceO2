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

#include "ITSWorkflow/RecoWorkflow.h"

#include "ITSWorkflow/DigitReaderSpec.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"

namespace o2
{
namespace ITS
{

namespace RecoWorkflow
{

framework::WorkflowSpec getWorkflow(bool useMC)
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::ITS::getDigitReaderSpec(useMC));
  specs.emplace_back(o2::ITS::getClustererSpec(useMC));
  specs.emplace_back(o2::ITS::getClusterWriterSpec(useMC));
  //specs.emplace_back(o2::ITS::getTrackerSpec());
  specs.emplace_back(o2::ITS::getCookedTrackerSpec(useMC));
  specs.emplace_back(o2::ITS::getTrackWriterSpec(useMC));

  return specs;
}

} // namespace RecoWorkflow

} // namespace ITS
} // namespace o2
