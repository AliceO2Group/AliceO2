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
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/PublisherSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"

namespace o2::aodproducer
{

framework::WorkflowSpec getAODProducerWorkflow(int ignoreWriter)
{
  // TODO:
  // switch to configurable parameters (?)
  bool useMC = true;

  // FIXME:
  // switch (?) from o2::ft0::getReconstructionSpec to RecPointReader
  // (which does not return RECCHDATA at the moment)
  framework::WorkflowSpec specs{
    o2::vertexing::getPrimaryVertexReaderSpec(useMC),
    o2::globaltracking::getTrackTPCITSReaderSpec(useMC),
    o2::its::getITSTrackReaderSpec(useMC),
    o2::tpc::getTPCTrackReaderSpec(useMC),
    o2::ft0::getDigitReaderSpec(useMC),
    o2::ft0::getReconstructionSpec(useMC),
    o2::aodproducer::getAODProducerWorkflowSpec(ignoreWriter)};

  return specs;
}

} // namespace o2::aodproducer
