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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataOutputDirector.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/TableBuilder.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Variant.h"
#include "../../../Algorithm/include/Algorithm/HeaderStack.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/StringHelpers.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/RuntimeError.h"

#include "TFile.h"
#include "TTree.h"

#include <ROOT/RSnapshotOptions.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <ROOT/RVec.hxx>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>

template class std::vector<o2::framework::OutputObjectInfo>;
template class std::vector<o2::framework::OutputTaskInfo>;
using namespace o2::framework::data_matcher;

namespace o2
{
namespace framework
{

struct InputObjectRoute {
  std::string name;
  uint32_t uniqueId;
  std::string directory;
  uint32_t taskHash;
  OutputObjHandlingPolicy policy;
  OutputObjSourceType sourceType;
};

struct InputObject {
  TClass* kind = nullptr;
  void* obj = nullptr;
  std::string name;
};

const static std::unordered_map<OutputObjHandlingPolicy, std::string> ROOTfileNames = {{OutputObjHandlingPolicy::AnalysisObject, "AnalysisResults.root"},
                                                                                       {OutputObjHandlingPolicy::QAObject, "QAResults.root"}};

// =============================================================================
DataProcessorSpec CommonDataProcessors::getOutputObjHistSink(std::vector<OutputObjectInfo> const& objmap, std::vector<OutputTaskInfo> const& tskmap)
{
  auto writerFunction = [objmap, tskmap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto inputObjects = std::make_shared<std::vector<std::pair<InputObjectRoute, InputObject>>>();

    auto endofdatacb = [inputObjects](EndOfStreamContext& context) {
      LOG(DEBUG) << "Writing merged objects and histograms to file";
      if (inputObjects->empty()) {
        LOG(ERROR) << "Output object map is empty!";
        context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        return;
      }
      std::string currentDirectory = "";
      std::string currentFile = "";
      TFile* f[OutputObjHandlingPolicy::numPolicies];
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        f[i] = nullptr;
      }
      for (auto& [route, entry] : *inputObjects) {
        auto file = ROOTfileNames.find(route.policy);
        if (file != ROOTfileNames.end()) {
          auto filename = file->second;
          if (f[route.policy] == nullptr) {
            f[route.policy] = TFile::Open(filename.c_str(), "RECREATE");
          }
          auto nextDirectory = route.directory;
          if ((nextDirectory != currentDirectory) || (filename != currentFile)) {
            if (!f[route.policy]->FindKey(nextDirectory.c_str())) {
              f[route.policy]->mkdir(nextDirectory.c_str());
            }
            currentDirectory = nextDirectory;
            currentFile = filename;
          }

          // translate the list-structure created by the registry into a directory structure within the file
          std::function<void(TList*, TDirectory*)> writeListToFile;
          writeListToFile = [&](TList* list, TDirectory* parentDir) {
            TIter next(list);
            TNamed* object = nullptr;
            while ((object = (TNamed*)next())) {
              if (object->InheritsFrom(TList::Class())) {
                writeListToFile((TList*)object, parentDir->mkdir(object->GetName(), object->GetName(), true));
              } else {
                parentDir->WriteObjectAny(object, object->Class(), object->GetName());
                list->Remove(object);
              }
            }
          };

          TDirectory* currentDir = f[route.policy]->GetDirectory(currentDirectory.c_str());
          if (route.sourceType == OutputObjSourceType::HistogramRegistrySource) {
            TList* outputList = (TList*)entry.obj;
            outputList->SetOwner(false);

            // if registry should live in dedicated folder a TNamed object is appended to the list
            if (outputList->Last()->IsA() == TNamed::Class()) {
              delete outputList->Last();
              outputList->RemoveLast();
              currentDir = currentDir->mkdir(outputList->GetName(), outputList->GetName(), true);
            }

            writeListToFile(outputList, currentDir);
            outputList->SetOwner();
            delete outputList;
            entry.obj = nullptr;
          } else {
            currentDir->WriteObjectAny(entry.obj, entry.kind, entry.name.c_str());
          }
        }
      }
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        if (f[i] != nullptr) {
          f[i]->Close();
        }
      }
      LOG(DEBUG) << "All outputs merged in their respective target files";
      context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);
    return [inputObjects, objmap, tskmap](ProcessingContext& pc) mutable -> void {
      auto const& ref = pc.inputs().get("x");
      if (!ref.header) {
        LOG(ERROR) << "Header not found";
        return;
      }
      if (!ref.payload) {
        LOG(ERROR) << "Payload not found";
        return;
      }
      auto datah = o2::header::get<o2::header::DataHeader*>(ref.header);
      if (!datah) {
        LOG(ERROR) << "No data header in stack";
        return;
      }

      auto objh = o2::header::get<o2::framework::OutputObjHeader*>(ref.header);
      if (!objh) {
        LOG(ERROR) << "No output object header in stack";
        return;
      }

      FairTMessage tm(const_cast<char*>(ref.payload), static_cast<int>(datah->payloadSize));
      InputObject obj;
      obj.kind = tm.GetClass();
      if (obj.kind == nullptr) {
        LOG(error) << "Cannot read class info from buffer.";
        return;
      }

      auto policy = objh->mPolicy;
      auto sourceType = objh->mSourceType;
      auto hash = objh->mTaskHash;

      obj.obj = tm.ReadObjectAny(obj.kind);
      TNamed* named = static_cast<TNamed*>(obj.obj);
      obj.name = named->GetName();
      auto hpos = std::find_if(tskmap.begin(), tskmap.end(), [&](auto&& x) { return x.id == hash; });
      if (hpos == tskmap.end()) {
        LOG(ERROR) << "No task found for hash " << hash;
        return;
      }
      auto taskname = hpos->name;
      auto opos = std::find_if(objmap.begin(), objmap.end(), [&](auto&& x) { return x.id == hash; });
      if (opos == objmap.end()) {
        LOG(ERROR) << "No object list found for task " << taskname << " (hash=" << hash << ")";
        return;
      }
      auto objects = opos->bindings;
      if (std::find(objects.begin(), objects.end(), obj.name) == objects.end()) {
        LOG(ERROR) << "No object " << obj.name << " in map for task " << taskname;
        return;
      }
      auto nameHash = compile_time_hash(obj.name.c_str());
      InputObjectRoute key{obj.name, nameHash, taskname, hash, policy, sourceType};
      auto existing = std::find_if(inputObjects->begin(), inputObjects->end(), [&](auto&& x) { return (x.first.uniqueId == nameHash) && (x.first.taskHash == hash); });
      if (existing == inputObjects->end()) {
        inputObjects->push_back(std::make_pair(key, obj));
        return;
      }
      auto merger = existing->second.kind->GetMerge();
      if (!merger) {
        LOG(ERROR) << "Already one unmergeable object found for " << obj.name;
        return;
      }

      TList coll;
      coll.Add(static_cast<TObject*>(obj.obj));
      merger(existing->second.obj, &coll, nullptr);
    };
  };

  DataProcessorSpec spec{
    "internal-dpl-aod-global-analysis-file-sink",
    {InputSpec("x", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"ATSK"}))},
    Outputs{},
    AlgorithmSpec(writerFunction),
    {}};

  return spec;
}

enum FileType : int {
  AOD,
  DANGLING
};

// add sink for the AODs
DataProcessorSpec
  CommonDataProcessors::getGlobalAODSink(std::shared_ptr<DataOutputDirector> dod,
                                         std::vector<InputSpec> const& outputInputs)
{

  auto writerFunction = [dod, outputInputs](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    LOGP(DEBUG, "======== getGlobalAODSink::Init ==========");

    // find out if any table needs to be saved
    bool hasOutputsToWrite = false;
    for (auto& outobj : outputInputs) {
      auto ds = dod->getDataOutputDescriptors(outobj);
      if (ds.size() > 0) {
        hasOutputsToWrite = true;
        break;
      }
    }

    // if nothing needs to be saved then return a trivial functor
    // this happens when nothing needs to be saved but there are dangling outputs
    if (!hasOutputsToWrite) {
      return [](ProcessingContext&) mutable -> void {
        static bool once = false;
        if (!once) {
          LOG(INFO) << "No AODs to be saved.";
          once = true;
        }
      };
    }

    // end of data functor is called at the end of the data stream
    auto endofdatacb = [dod](EndOfStreamContext& context) {
      dod->closeDataFiles();
      context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    auto& callbacks = ic.services().get<CallbackService>();
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

    // prepare map<uint64_t, uint64_t>(startTime, tfNumber)
    std::map<uint64_t, uint64_t> tfNumbers;

    // this functor is called once per time frame
    return std::move([dod, tfNumbers](ProcessingContext& pc) mutable -> void {
      LOGP(DEBUG, "======== getGlobalAODSink::processing ==========");
      LOGP(DEBUG, " processing data set with {} entries", pc.inputs().size());

      // return immediately if pc.inputs() is empty. This should never happen!
      if (pc.inputs().size() == 0) {
        LOGP(INFO, "No inputs available!");
        return;
      }

      // update tfNumbers
      uint64_t startTime = 0;
      uint64_t tfNumber = 0;
      auto ref = pc.inputs().get("tfn");
      if (ref.spec && ref.payload) {
        startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->startTime;
        tfNumber = pc.inputs().get<uint64_t>("tfn");
        tfNumbers.insert(std::pair<uint64_t, uint64_t>(startTime, tfNumber));
      }

      // loop over the DataRefs which are contained in pc.inputs()
      for (const auto& ref : pc.inputs()) {
        if (!ref.spec) {
          LOGP(DEBUG, "The input \"{}\" is not valid and will be skipped!", ref.spec->binding);
          continue;
        }

        // skip non-AOD refs
        if (!DataSpecUtils::partialMatch(*ref.spec, header::DataOrigin("AOD"))) {
          continue;
        }
        startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->startTime;

        // does this need to be saved?
        auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
        auto tableName = std::string(dh->dataDescription.str);
        auto ds = dod->getDataOutputDescriptors(*dh);
        if (ds.size() <= 0) {
          continue;
        }

        // get TF number fro startTime
        auto it = tfNumbers.find(startTime);
        if (it != tfNumbers.end()) {
          tfNumber = (it->second / dod->getNumberTimeFramesToMerge()) * dod->getNumberTimeFramesToMerge();
        } else {
          LOGP(FATAL, "No time frame number found for output with start time {}", startTime);
          throw std::runtime_error("Processing is stopped!");
        }

        // get the TableConsumer and corresponding arrow table
        auto s = pc.inputs().get<TableConsumer>(ref.spec->binding);
        auto table = s->asArrowTable();
        if (!table->Validate().ok()) {
          LOGP(WARNING, "The table \"{}\" is not valid and will not be saved!", tableName);
          continue;
        } else if (table->schema()->fields().empty() == true) {
          LOGP(DEBUG, "The table \"{}\" is empty but will be saved anyway!", tableName);
        }

        // loop over all DataOutputDescriptors
        // a table can be saved in multiple ways
        // e.g. different selections of columns to different files
        for (auto d : ds) {
          auto fileAndFolder = dod->getFileFolder(d, tfNumber);
          auto treename = fileAndFolder.folderName + d->treename;
          TableToTree ta2tr(table,
                            fileAndFolder.file,
                            treename.c_str());

          if (d->colnames.size() > 0) {
            for (auto cn : d->colnames) {
              auto idx = table->schema()->GetFieldIndex(cn);
              auto col = table->column(idx);
              auto field = table->schema()->field(idx);
              if (idx != -1) {
                ta2tr.addBranch(col, field);
              }
            }
          } else {
            ta2tr.addAllBranches();
          }
          ta2tr.process();
        }
      }
    });
  }; // end of writerFunction

  // the command line options relevant for the writer are global
  // see runDataProcessing.h
  DataProcessorSpec spec{
    "internal-dpl-aod-writer",
    outputInputs,
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
      throw runtime_error("output file missing");
    }

    bool hasOutputsToWrite = false;
    auto [variables, outputMatcher] = DataDescriptorQueryBuilder::buildFromKeepConfig(keepString);
    VariableContext context;
    for (auto& spec : danglingOutputInputs) {
      auto concrete = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      if (outputMatcher->match(concrete, context)) {
        hasOutputsToWrite = true;
      }
    }
    if (hasOutputsToWrite == false) {
      return [](ProcessingContext&) mutable -> void {
        static bool once = false;
        if (!once) {
          LOG(DEBUG) << "No dangling output to be dumped.";
          once = true;
        }
      };
    }
    auto output = std::make_shared<std::ofstream>(filename.c_str(), std::ios_base::binary);
    return [output, matcher = outputMatcher](ProcessingContext& pc) mutable -> void {
      VariableContext matchingContext;
      LOG(DEBUG) << "processing data set with " << pc.inputs().size() << " entries";
      for (const auto& entry : pc.inputs()) {
        LOG(DEBUG) << "  " << *(entry.spec);
        auto header = DataRefUtils::getHeader<header::DataHeader*>(entry);
        auto dataProcessingHeader = DataRefUtils::getHeader<DataProcessingHeader*>(entry);
        if (matcher->match(*header, matchingContext) == false) {
          continue;
        }
        output->write(reinterpret_cast<char const*>(header), sizeof(header::DataHeader));
        output->write(reinterpret_cast<char const*>(dataProcessingHeader), sizeof(DataProcessingHeader));
        output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
        LOG(DEBUG) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
      }
    };
  };

  std::vector<InputSpec> validBinaryInputs;
  auto onlyTimeframe = [](InputSpec const& input) {
    return (DataSpecUtils::partialMatch(input, o2::header::DataOrigin("TFN")) == false) &&
           input.lifetime == Lifetime::Timeframe;
  };

  auto noTimeframe = [](InputSpec const& input) {
    return (DataSpecUtils::partialMatch(input, o2::header::DataOrigin("TFN")) == true) ||
           input.lifetime != Lifetime::Timeframe;
  };

  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(validBinaryInputs), onlyTimeframe);
  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(unmatched), noTimeframe);

  DataProcessorSpec spec{
    "internal-dpl-injected-global-binary-file-sink",
    validBinaryInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"outfile", VariantType::String, "dpl-out.bin", {"Name of the output file"}},
     {"keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION to save in outfile"}}}};

  return spec;
}

DataProcessorSpec CommonDataProcessors::getGlobalFairMQSink(std::vector<InputSpec> const& danglingOutputInputs)
{

  // we build the default channel configuration from the binding of the first input
  // in order to have more than one we would need to possibility to have support for
  // vectored options
  // use the OutputChannelSpec as a tool to create the default configuration for the out-of-band channel
  OutputChannelSpec externalChannelSpec;
  externalChannelSpec.name = "downstream";
  externalChannelSpec.type = ChannelType::Push;
  externalChannelSpec.method = ChannelMethod::Bind;
  externalChannelSpec.hostname = "localhost";
  externalChannelSpec.port = 0;
  externalChannelSpec.listeners = 0;
  // in principle, protocol and transport are two different things but fur simplicity
  // we use ipc when shared memory is selected and the normal tcp url whith zeromq,
  // this is for building the default configuration which can be simply changed from the
  // command line
  externalChannelSpec.protocol = ChannelProtocol::IPC;
  std::string defaultChannelConfig = formatExternalChannelConfiguration(externalChannelSpec);
  // at some point the formatting tool might add the transport as well so we have to check
  return specifyFairMQDeviceOutputProxy("internal-dpl-injected-output-proxy", danglingOutputInputs, defaultChannelConfig.c_str());
}

DataProcessorSpec CommonDataProcessors::getDummySink(std::vector<InputSpec> const& danglingOutputInputs)
{
  return DataProcessorSpec{
    "internal-dpl-injected-dummy-sink",
    danglingOutputInputs,
    Outputs{},
    AlgorithmSpec([](ProcessingContext& ctx) {})};
}

} // namespace framework
} // namespace o2
