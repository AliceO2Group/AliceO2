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
#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCReconstruction/CTFCoder.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  CompressedClusters clusters;

  if (mFromFile) {
    auto tmp = pc.inputs().get<CompressedClustersROOT*>("input");
    if (tmp == nullptr) {
      LOG(ERROR) << "invalid input";
      return;
    }
    clusters = *tmp;
  } else {
    auto tmp = pc.inputs().get<CompressedClustersFlat*>("input");
    if (tmp == nullptr) {
      LOG(ERROR) << "invalid input";
      return;
    }
    clusters = *tmp;
  }

  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"TPC", "CTFDATA", 0, Lifetime::Timeframe});
  CTFCoder::encode(buffer, clusters);
  auto eeb = CTF::get(buffer.data()); // cast to container pointer
  eeb->compactify();                  // eliminate unnecessary padding
  buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  // eeb->print();
  LOG(INFO) << "Created encoded data of size " << eeb->size() << " for TPC";
}

DataProcessorSpec getEntropyEncoderSpec(bool inputFromFile)
{
  std::vector<InputSpec> inputs;
  header::DataDescription inputType = inputFromFile ? header::DataDescription("COMPCLUSTERS") : header::DataDescription("COMPCLUSTERSFLAT");
  return DataProcessorSpec{
    "tpc-entropy-encoder", // process id
    {{"input", "TPC", inputType, 0, Lifetime::Timeframe}},
    Outputs{{"TPC", "CTFDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec(adaptFromTask<EntropyEncoderSpec>(inputFromFile))};
}

} // namespace tpc
} // namespace o2
