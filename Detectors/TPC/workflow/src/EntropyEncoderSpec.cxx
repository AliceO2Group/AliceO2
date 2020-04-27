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
//#include "TPCClusterDecompressor.cxx"
#include <memory> // for make_shared
#include <vector>

// TEMP: as a test, the processor simply writes the data to file, remove this
#include <TFile.h>
#include <TTree.h>

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
      auto compressed = pc.inputs().get<CompressedClusters*>("input");
      if (compressed == nullptr) {
        LOG(ERROR) << "invalid input";
        return;
      }
      LOG(INFO) << "input data with " << compressed->nTracks << " track(s) and " << compressed->nAttachedClusters << " attached clusters";

      std::unique_ptr<TFile> testFile(TFile::Open("tpc-cluster-encoder.root", "RECREATE"));
      testFile->WriteObject(compressed.get(), "TPCCompressedClusters");
      testFile->Write();
      testFile->Close();

      //o2::tpc::ClusterNativeAccess clustersNativeDecoded; // Cluster native access structure as used by the tracker
      //std::vector<o2::tpc::ClusterNative> clusterBuffer; // std::vector that will hold the actual clusters, clustersNativeDecoded will point inside here
      //mDecoder.decompress(clustersCompressed, clustersNativeDecoded, clusterBuffer, param); // Run decompressor
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
