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
#include "DataFormatsTRD/TriggerRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace trd
{

class TRDTrackletReader : public o2::framework::Task
{
 public:
  TRDTrackletReader(bool useMC) : mUseMC(useMC) {}
  ~TRDTrackletReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC{false};
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInFileName{"trdtracklets.root"};
  std::string mInTreeName{"o2sim"};
  std::vector<o2::trd::Tracklet64> mTracklets, *mTrackletsPtr = &mTracklets;
  std::vector<o2::trd::TriggerRecord> mTriggerRecords, *mTriggerRecordsPtr = &mTriggerRecords;
  std::vector<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
};

/// create a processor spec
/// read TRD tracklets from a root file
framework::DataProcessorSpec getTRDTrackletReaderSpec(bool useMC);

} // namespace trd
} // namespace o2

#endif /* O2_TRD_TRACKLETREADER */
