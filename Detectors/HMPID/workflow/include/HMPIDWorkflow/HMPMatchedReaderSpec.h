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

/// @file   HMPMatchedReaderSpec.h

#ifndef O2_HMP_MATCHINFOREADER
#define O2_HMP_MATCHINFOREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace hmpid
{

class HMPMatchedReader : public o2::framework::Task
{
 public:
  HMPMatchedReader(bool useMC) : mUseMC(useMC) {}
  ~HMPMatchedReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  bool mUseMC = false;
  //  int mMode = 1;

  std::string mInFileName{"o2match_hmp.root"};
  std::string mInTreeName{"matchHMP"};
  std::unique_ptr<TFile> mFile = nullptr;
  std::unique_ptr<TTree> mTree = nullptr;
  std::vector<o2::dataformats::MatchInfoHMP> mMatches, *mMatchesPtr = &mMatches;
  // std::vector<o2::dataformats::MatchInfoHMP> *mMatchesPtr = nullptr;
  std::vector<o2::MCCompLabel> mLabelHMP, *mLabelHMPPtr = &mLabelHMP;
};

/// create a processor spec
/// read matched HMP clusters from a ROOT file
framework::DataProcessorSpec getHMPMatchedReaderSpec(bool useMC);

} // namespace hmpid
} // namespace o2

#endif /* O2_HMP_MATCHINFOREADER */
