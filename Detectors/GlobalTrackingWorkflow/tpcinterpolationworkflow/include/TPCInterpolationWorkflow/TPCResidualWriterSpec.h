// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RESIDUAL_WRITER_H
#define O2_TPC_RESIDUAL_WRITER_H

/// @file   TPCResidualWriterSpec.h

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>

namespace o2
{
namespace tpc
{

class ResidualWriterTPC : public o2::framework::Task
{
 public:
  ResidualWriterTPC(bool useMC = false) : mUseMC(useMC) {}
  ~ResidualWriterTPC() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  bool mUseMC = false;                               ///< MC flag
  std::string mOutFileName = "o2residuals_tpc.root"; ///< name of output file
  std::string mTreeName = "residualsTPC";            ///< name of tree containing output
  std::string mOutTracksBranchName = "tracks";       ///< name of branch containing output used tracks
  std::string mOutResidualsBranchName = "residuals"; ///< name of branch containing output used residuals
  std::unique_ptr<TFile> mFile = nullptr;
  std::unique_ptr<TTree> mTree = nullptr;
};

/// create a processor spec
framework::DataProcessorSpec getTPCResidualWriterSpec(bool useMC);

} // namespace tpc
} // namespace o2

#endif
