// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AODProducerWorkflowSpec.h

#ifndef O2_AODPRODUCER_WORKFLOW_SPEC
#define O2_AODPRODUCER_WORKFLOW_SPEC

#include "DataFormatsFT0/RecPoints.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "TStopwatch.h"
#include <CCDB/BasicCCDBManager.h>
#include <string>
#include <vector>

using namespace o2::framework;

namespace o2::aodproducer
{

using TracksTable = o2::soa::Table<o2::aod::track::CollisionId,
                                   o2::aod::track::TrackType,
                                   o2::aod::track::X,
                                   o2::aod::track::Alpha,
                                   o2::aod::track::Y,
                                   o2::aod::track::Z,
                                   o2::aod::track::Snp,
                                   o2::aod::track::Tgl,
                                   o2::aod::track::Signed1Pt,
                                   o2::aod::track::SigmaY,
                                   o2::aod::track::SigmaZ,
                                   o2::aod::track::SigmaSnp,
                                   o2::aod::track::SigmaTgl,
                                   o2::aod::track::Sigma1Pt,
                                   o2::aod::track::RhoZY,
                                   o2::aod::track::RhoSnpY,
                                   o2::aod::track::RhoSnpZ,
                                   o2::aod::track::RhoTglY,
                                   o2::aod::track::RhoTglZ,
                                   o2::aod::track::RhoTglSnp,
                                   o2::aod::track::Rho1PtY,
                                   o2::aod::track::Rho1PtZ,
                                   o2::aod::track::Rho1PtSnp,
                                   o2::aod::track::Rho1PtTgl,
                                   o2::aod::track::TPCInnerParam,
                                   o2::aod::track::Flags,
                                   o2::aod::track::ITSClusterMap,
                                   o2::aod::track::TPCNClsFindable,
                                   o2::aod::track::TPCNClsFindableMinusFound,
                                   o2::aod::track::TPCNClsFindableMinusCrossedRows,
                                   o2::aod::track::TPCNClsShared,
                                   o2::aod::track::TRDPattern,
                                   o2::aod::track::ITSChi2NCl,
                                   o2::aod::track::TPCChi2NCl,
                                   o2::aod::track::TRDChi2,
                                   o2::aod::track::TOFChi2,
                                   o2::aod::track::TPCSignal,
                                   o2::aod::track::TRDSignal,
                                   o2::aod::track::TOFSignal,
                                   o2::aod::track::Length,
                                   o2::aod::track::TOFExpMom,
                                   o2::aod::track::TrackEtaEMCAL,
                                   o2::aod::track::TrackPhiEMCAL>;

class AODProducerWorkflowDPL : public Task
{
 public:
  AODProducerWorkflowDPL() = default;
  ~AODProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  int mFillTracksITS = 1;
  int mFillTracksTPC = 1;
  int mFillTracksITSTPC = 1;
  TStopwatch mTimer;

  uint64_t maxGlBC = 0;
  uint64_t minGlBC = INT64_MAX;

  void findMinMaxBc(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints, gsl::span<const o2::dataformats::TrackTPCITS>& tracksITSTPC, const std::vector<o2::InteractionTimeRecord>& mcRecords);
  int64_t getTFNumber(uint64_t firstVtxGlBC, int runNumber);

  template <typename TracksType, typename TracksCursorType>
  void fillTracksTable(const TracksType& tracks, std::vector<int>& vCollRefs, const TracksCursorType& tracksCursor, int trackType);
};

/// create a processor spec
framework::DataProcessorSpec getAODProducerWorkflowSpec();

} // namespace o2::aodproducer

#endif /* O2_AODPRODUCER_WORKFLOW_SPEC */
