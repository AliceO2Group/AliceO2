// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fmt/format.h>
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "input-spec", VariantType::String, "A:FLP/RAWDATA", {"selection string input specs"}});
}

#include "Framework/runDataProcessing.h"

using RDH = o2::header::RAWDataHeader;

void printHeader()
{
  fmt::print("{:>5} {:>4} {:>4} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1}\n", "PkC", "pCnt", "fId", "Mem", "CRU", "GLID", "HBOrbit", "HB BC", "s");
}

void printRDH(const RDH& rdh)
{
  const int globalLinkID = int(rdh.linkID) + (((rdh.word1 >> 32) >> 28) * 12);

  fmt::print("{:>5} {:>4} {:>4} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1}\n", (uint64_t)rdh.packetCounter, (uint64_t)rdh.pageCnt, (uint64_t)rdh.feeId, (uint64_t)(rdh.memorySize), (uint64_t)rdh.cruID, (uint64_t)globalLinkID, (uint64_t)rdh.heartbeatOrbit, (uint64_t)rdh.heartbeatBC, (uint64_t)rdh.stop);
}

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  WorkflowSpec workflow;
  workflow.emplace_back(DataProcessorSpec{
    "calib-pedestal",
    select(config.options().get<std::string>("input-spec").c_str()),
    Outputs{},
    AlgorithmSpec{[](InitContext& setup) { //
      return adaptStateless([](InputRecord& inputs, DataAllocator& outputs) {
        for (auto& input : inputs) {
          const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
          LOG(INFO) << dh->dataOrigin.as<std::string>() << "/" << dh->dataDescription.as<std::string>() << "/"
                    << dh->subSpecification << " payload size " << dh->payloadSize;

          // there is a bug in InpuRecord::get for vectors of simple types, not catched in
          // DataAllocator unit test
          //auto data = inputs.get<std::vector<char>>(input.spec->binding.c_str());
          //LOG(INFO) << "data size " << data.size();
          printHeader();
          try {
            const char* pos = input.payload;
            const char* last = pos + dh->payloadSize;

            o2::framework::RawParser parser(input.payload, dh->payloadSize);

            //while (pos < last) {
            for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
              auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV4>();
              if (!rdhPtr) {
                break;
              }
              //const auto& rdh = *((RDH*)pos);
              const auto& rdh = *rdhPtr;
              //const auto payloadSize = rdh.memorySize - rdh.headerSize;
              //const auto dataWrapperID = rdh.endPointID;
              //const auto linkID = rdh.linkID;
              //const auto globalLinkID = linkID + dataWrapperID * 12;
              printRDH(rdh);

              //pos += rdh.offsetToNext;
            }
          } catch (const std::runtime_error& e) {
            LOG(ERROR) << "can not create raw parser form input data";
            o2::header::hexDump("payload", input.payload, dh->payloadSize, 64);
            LOG(ERROR) << e.what();
          }
        }
      });
    }}});
  return workflow;
}
