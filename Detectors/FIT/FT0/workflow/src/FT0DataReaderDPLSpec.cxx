// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DataReaderDPLSpec.cxx

#include "FT0Workflow/FT0DataReaderDPLSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{
using namespace std;
template <typename RawReader>
void FT0DataReaderDPLSpec<RawReader>::init(InitContext& ic)

{
}
template <typename RawReader>
void FT0DataReaderDPLSpec<RawReader>::run(ProcessingContext& pc)

{
  DPLRawParser parser(pc.inputs());
  mRawReader.clear();
  LOG(INFO) << "FT0DataReaderDPLSpec";
  uint64_t count = 0;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    //Proccessing each page
    count++;
    auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
    gsl::span<const uint8_t> payload(it.data(), it.size());
    mRawReader.process(rdhPtr->linkID, payload);
  }
  LOG(INFO)<<"Pages: "<<count;
  mRawReader.print();
  mRawReader.makeSnapshot(pc.outputs());
}

template<typename RawReader>
DataProcessorSpec getFT0DataReaderDPLSpec(bool dumpReader)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderFT0ext";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  return DataProcessorSpec{
    "ft0-datareader-dpl",
    o2::framework::select("TF:FT0/RAWDATA"),
    outputSpec,
    adaptFromTask<FT0DataReaderDPLSpec<RawReader>>(dumpReader),
    Options{}};
}

} // namespace ft0
} // namespace o2
