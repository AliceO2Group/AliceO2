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
namespace MFT
{

namespace RecoWorkflow
{

framework::WorkflowSpec getWorkflow()
{
  framework::WorkflowSpec specs;

  //specs.emplace_back(o2::MFT::getDigitReaderSpec());
  //specs.emplace_back(o2::MFT::getClustererSpec());
  //specs.emplace_back(o2::MFT::getClusterWriterSpec());

  specs.emplace_back(o2::MFT::getClusterReaderSpec());
  specs.emplace_back(o2::MFT::getTrackerSpec());
  //specs.emplace_back(o2::MFT::getTrackWriterSpec());

  return specs;
}

} // namespace RecoWorkflow

} // namespace MFT
} // namespace o2
