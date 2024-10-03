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

#ifndef O2_FDD_RECPOINTREADER
#define O2_FDD_RECPOINTREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFDD/RecPoint.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
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
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFDD;

  std::vector<o2::fdd::RecPoint>* mRecPoints = nullptr;
  std::vector<o2::fdd::ChannelDataFloat>* mChannelData = nullptr;

  std::string mInputFileName = "o2reco_fdd.root";
  std::string mRecPointTreeName = "o2sim";
  std::string mRecPointBranchName = "FDDCluster";
  std::string mChannelDataBranchName = "FDDRecChData";
};

/// create a processor spec
/// read simulated FDD digits from a root file
framework::DataProcessorSpec getFDDRecPointReaderSpec(bool useMC);

} // namespace fdd
} // namespace o2

#endif /* O2_FDD_RECPOINTREADER */
