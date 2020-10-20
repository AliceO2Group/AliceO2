// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  AODProducerWorkflow.cxx

#include "Algorithm/RangeTokenizer.h"
#include "AODProducerWorkflow/AODProducerWorkflow.h"
#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "DataFormatsTPC/Constants.h"
#include "FT0Workflow/DigitReaderSpec.h"
#include "FT0Workflow/ReconstructionSpec.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "GlobalTrackingWorkflow/MatchTPCITSWorkflow.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/PrimaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/PublisherSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"

namespace o2
{
namespace aodproducer
{

framework::WorkflowSpec getAODProducerWorkflow()
{
  framework::WorkflowSpec specs;

  // TODO:
  // switch to configurable parameters (?)
  bool useFT0 = true;
  bool useMC = false;

  specs.emplace_back(o2::vertexing::getPrimaryVertexingSpec(useFT0, useMC));
  specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
  specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
  specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));

  // FIXME:
  // switch (?) to RecPointReader (which does not return RECCHDATA at the moment)
  specs.emplace_back(o2::ft0::getDigitReaderSpec(useMC));
  specs.emplace_back(o2::ft0::getReconstructionSpec(useMC));

  specs.emplace_back(o2::aodproducer::getAODProducerWorkflowSpec());
  return specs;
}

} // namespace aodproducer
} // namespace o2
