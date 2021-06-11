// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TRDTrackletReaderSpec.h

#ifndef O2_TRD_TRACKLETREADER
#define O2_TRD_TRACKLETREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace trd
{

class TRDTrackletReader : public o2::framework::Task
{
 public:
  TRDTrackletReader(bool useMC, bool useTrkltTransf) : mUseMC(useMC), mUseTrackletTransform(useTrkltTransf) {}
  ~TRDTrackletReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree();
  void connectTreeCTracklet();
  bool mUseMC{false};
  bool mUseTrackletTransform{false};
  std::unique_ptr<TFile> mFileTrklt;
  std::unique_ptr<TTree> mTreeTrklt;
  std::unique_ptr<TFile> mFileCTrklt;
  std::unique_ptr<TTree> mTreeCTrklt;
  std::string mInFileNameTrklt{"trdtracklets.root"};
  std::string mInTreeNameTrklt{"o2sim"};
  std::vector<o2::trd::CalibratedTracklet> mTrackletsCal, *mTrackletsCalPtr = &mTrackletsCal;
  std::vector<o2::trd::Tracklet64> mTracklets, *mTrackletsPtr = &mTracklets;
  std::vector<o2::trd::TriggerRecord> mTriggerRecords, *mTriggerRecordsPtr = &mTriggerRecords;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
};

/// create a processor spec
/// read TRD tracklets from a root file
framework::DataProcessorSpec getTRDTrackletReaderSpec(bool useMC, bool useCalibratedTracklets);

} // namespace trd
} // namespace o2

#endif /* O2_TRD_TRACKLETREADER */
