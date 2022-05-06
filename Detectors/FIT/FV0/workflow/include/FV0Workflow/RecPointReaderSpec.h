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

/// @file   RecPointReaderSpec.h

#ifndef O2_FV0_RECPOINTREADER
#define O2_FV0_RECPOINTREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/RootSerializationSupport.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFV0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

class RecPointReader : public Task
{
 public:
  RecPointReader(bool useMC = false);
  ~RecPointReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = false; // use MC truth
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFV0;

  std::vector<o2::fv0::RecPoints>* mRecPoints = nullptr;
  std::vector<o2::fv0::ChannelDataFloat>* mChannelData = nullptr;

  std::string mInputFileName = "o2reco_fv0.root";
  std::string mRecPointTreeName = "o2sim";
  std::string mRecPointBranchName = "FV0Cluster";
  std::string mChannelDataBranchName = "FV0RecChData";
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getRecPointReaderSpec(bool useMC);

} // namespace fv0
} // namespace o2

#endif /* O2_FV0_RECPOINTREADER */
