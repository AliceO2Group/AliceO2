// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EntropyEncoderSpec.cxx
/// @author Michael Lettrich, Matthias Richter
/// @since  2020-01-16
/// @brief  ProcessorSpec for the TPC cluster entropy encoding

#include "TPCWorkflow/EntropyEncoderSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCEntropyCoding/EncodedClusters.h"
#include "TPCEntropyCoding/TPCEntropyEncoder.h"
#include "librans/rans.h"

//#include "TPCClusterDecompressor.cxx"
#include <memory> // for make_shared
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <Compression.h>

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

DataProcessorSpec getEntropyEncoderSpec()
{
  struct ProcessAttributes {
    int verbosity = 1;
  };

  auto initFunction = [](InitContext& ic) {
    auto processAttributes = std::make_shared<ProcessAttributes>();

    auto processingFct = [processAttributes](ProcessingContext& pc) {
      auto clusters = pc.inputs().get<CompressedClustersROOT*>("input");
      if (clusters == nullptr) {
        LOG(ERROR) << "invalid input";
        return;
      }

      auto encodedClusters = o2::tpc::TPCEntropyEncoder::encode(*clusters);

      const char* outFileName = "tpc-encoded-clusters.root";

      // Fixme (lettrich): no TPC specific files and  no TPC specific workflows
      // create the tree
      TFile f(outFileName, "recreate");
      TTree tree("EncodedClusters", "");
      o2::tpc::TPCEntropyEncoder::appendToTTree(tree, *encodedClusters);
      LOG(INFO) << "writing compressed clusters into " << outFileName;
    };

    return processingFct;
  };

  return DataProcessorSpec{"tpc-entropy-encoder", // process id
                           {{"input", "TPC", "COMPCLUSTERS", 0, Lifetime::Timeframe}},
                           {},
                           AlgorithmSpec(initFunction)};
}

} // namespace tpc
} // namespace o2
