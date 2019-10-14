// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonDataProcessors.h"

#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Variant.h"

#include "TFile.h"

#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

using namespace o2::framework::data_matcher;

namespace o2
{
namespace framework
{

struct InputObjectRoute {
  std::string uniqueId;
  std::string directory;
  bool operator<(InputObjectRoute const& other) const
  {
    return this->uniqueId < other.uniqueId;
  }
};

struct InputObject {
  TClass* kind = nullptr;
  void* obj = nullptr;
  std::string name;
};

std::string outputROOTfilename()
{
  return "AnalysisResults.root";
}

DataProcessorSpec CommonDataProcessors::getOutputObjSink(outputObjMap const& outMap)
{
  auto writerFunction = [outMap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto outputObjects = std::make_shared<std::map<InputObjectRoute, InputObject>>();

    auto endofdatacb = [outputObjects](EndOfStreamContext& context) {
      LOG(INFO) << "Writing merged objects to file";
      std::string currentDirectory = "";
      TFile* f = TFile::Open(outputROOTfilename().c_str(), "RECREATE");
      for (auto& [route, entry] : *outputObjects) {
        std::string nextDirectory = route.directory;
        if (nextDirectory != currentDirectory) {
          LOGP(INFO, "Now writing in {}", nextDirectory);
          if (!f->FindKey(nextDirectory.c_str())) {
            f->mkdir(nextDirectory.c_str());
          }
          currentDirectory = nextDirectory;
        }
        LOGP(INFO, "Now writing {}", entry.name);
        (f->GetDirectory(currentDirectory.c_str()))->WriteObjectAny(entry.obj, entry.kind, entry.name.c_str());
      }
      if (f) {
        f->Close();
      }
      LOG(INFO) << "All outputs merged in their respective target files";
      context.services().get<ControlService>().readyToQuit(QuitRequest::All);
    };

    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);
    return [outputObjects, outMap](ProcessingContext& pc) mutable -> void {
      auto const& ref = pc.inputs().get("x");
      if (!ref.header) {
        LOG(ERROR) << "Header not found";
        return;
      }
      if (!ref.payload) {
        LOG(ERROR) << "Payload not found";
        return;
      }
      auto dh = o2::header::get<o2::header::DataHeader*>(ref.header);
      if (!dh) {
        LOG(ERROR) << "DataHeader not found";
        return;
      }
      FairTMessage tm(const_cast<char*>(ref.payload), dh->payloadSize);
      InputObject obj;
      obj.kind = tm.GetClass();
      if (obj.kind == nullptr) {
        LOGP(error, "Cannot read class info from buffer.");
        return;
      }

      obj.obj = tm.ReadObjectAny(obj.kind);
      TNamed* named = static_cast<TNamed*>(obj.obj);
      LOGP(info, "Object name {}", named->GetName());
      obj.name = named->GetName();
      auto lookup = outMap.find(obj.name);
      std::string directory{"VariousObjects"};
      if (lookup != outMap.end()) {
        directory = lookup->second;
      }
      InputObjectRoute key{obj.name, directory};
      auto existing = outputObjects->find(key);
      if (existing == outputObjects->end()) {
        outputObjects->insert(std::make_pair(key, obj));
        return;
      }
      auto merger = existing->second.kind->GetMerge();
      if (!merger) {
        LOGP(error, "Already one object found for {}.", obj.name);
        return;
      }

      TList coll;
      coll.Add(static_cast<TObject*>(obj.obj));
      merger(existing->second.obj, &coll, nullptr);
    };
  };

  DataProcessorSpec spec{
    "internal-dpl-global-analysis-file-sink",
    {InputSpec("x", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"ATSK"}))},
    Outputs{},
    AlgorithmSpec(writerFunction),
    {}};

  return spec;
}

DataProcessorSpec
  CommonDataProcessors::getGlobalFileSink(std::vector<InputSpec> const& danglingOutputInputs,
                                          std::vector<InputSpec>& unmatched)
{
  auto writerFunction = [danglingOutputInputs](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto filename = ic.options().get<std::string>("outfile");
    auto keepString = ic.options().get<std::string>("keep");

    if (filename.empty()) {
      throw std::runtime_error("output file missing");
    }

    bool hasOutputsToWrite = false;
    auto [variables, outputMatcher] = DataDescriptorQueryBuilder::buildFromKeepConfig(keepString);
    VariableContext context;
    for (auto& spec : danglingOutputInputs) {
      auto concrete = DataSpecUtils::asConcreteDataMatcher(spec);
      if (outputMatcher->match(concrete, context)) {
        hasOutputsToWrite = true;
      }
    }
    if (hasOutputsToWrite == false) {
      return std::move([](ProcessingContext& pc) mutable -> void {
        static bool once = false;
        /// We do it like this until we can use the interruptible sleep
        /// provided by recent FairMQ releases.
        if (!once) {
          LOG(INFO) << "No dangling output to be dumped.";
          once = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      });
    }
    auto output = std::make_shared<std::ofstream>(filename.c_str(), std::ios_base::binary);
    return std::move([output, matcher = outputMatcher](ProcessingContext& pc) mutable -> void {
      VariableContext matchingContext;
      LOG(INFO) << "processing data set with " << pc.inputs().size() << " entries";
      for (const auto& entry : pc.inputs()) {
        LOG(INFO) << "  " << *(entry.spec);
        auto header = DataRefUtils::getHeader<header::DataHeader*>(entry);
        auto dataProcessingHeader = DataRefUtils::getHeader<DataProcessingHeader*>(entry);
        if (matcher->match(*header, matchingContext) == false) {
          continue;
        }
        output->write(reinterpret_cast<char const*>(header), sizeof(header::DataHeader));
        output->write(reinterpret_cast<char const*>(dataProcessingHeader), sizeof(DataProcessingHeader));
        output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
        LOG(INFO) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
      }
    });
  };

  std::vector<InputSpec> validBinaryInputs;
  auto onlyTimeframe = [](InputSpec const& input) {
    return input.lifetime == Lifetime::Timeframe;
  };

  auto noTimeframe = [](InputSpec const& input) {
    return input.lifetime != Lifetime::Timeframe;
  };

  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(validBinaryInputs), onlyTimeframe);
  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(unmatched), noTimeframe);

  DataProcessorSpec spec{
    "internal-dpl-global-binary-file-sink",
    validBinaryInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"outfile", VariantType::String, "dpl-out.bin", {"Name of the output file"}},
     {"keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION to save in outfile"}}}};

  return spec;
}

DataProcessorSpec CommonDataProcessors::getDummySink(std::vector<InputSpec> const& danglingOutputInputs)
{
  return DataProcessorSpec{
    "internal-dpl-dummy-sink",
    danglingOutputInputs,
    Outputs{},
  };
}

} // namespace framework
} // namespace o2
