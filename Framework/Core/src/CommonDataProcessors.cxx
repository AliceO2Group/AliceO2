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
};

struct InputObject {
  TClass* kind = nullptr;
  void* obj = nullptr;
  std::string name;
};

const static std::unordered_map<OutputObjHandlingPolicy, std::string> ROOTfileNames = {{OutputObjHandlingPolicy::AnalysisObject, "AnalysisResults.root"},
                                                                                       {OutputObjHandlingPolicy::QAObject, "QAResults.root"}};

// =============================================================================
DataProcessorSpec CommonDataProcessors::getHistogramRegistrySink(outputObjects const& objmap, const outputTasks& tskmap)
{
  auto writerFunction = [objmap, tskmap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto inputObjects = std::make_shared<std::vector<std::pair<InputObjectRoute, InputObject>>>();

    auto endofdatacb = [inputObjects](EndOfStreamContext& context) {
      LOG(DEBUG) << "Writing merged histograms to file";
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
          TList* outputList = (TList*)entry.obj;
          writeListToFile(outputList, f[route.policy]->GetDirectory(currentDirectory.c_str()));
          outputList->SetOwner(true);
          delete outputList; // properly remove the empty list and its sub-lists
        }
      }
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        if (f[i] != nullptr) {
          f[i]->Close();
        }
      }
      LOG(INFO) << "All outputs merged in their respective target files";
      context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);
    return [inputObjects, objmap, tskmap](ProcessingContext& pc) mutable -> void {
      auto const& ref = pc.inputs().get("y");
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
      auto hash = objh->mTaskHash;

      obj.obj = tm.ReadObjectAny(obj.kind);
      TNamed* named = static_cast<TNamed*>(obj.obj);
      obj.name = named->GetName();

      auto hpos = std::find_if(tskmap.begin(), tskmap.end(), [&](auto&& x) { return x.first == hash; });
      if (hpos == tskmap.end()) {
        LOG(ERROR) << "No task found for hash " << hash;
        return;
      }
      auto taskname = hpos->second;
      auto opos = std::find_if(objmap.begin(), objmap.end(), [&](auto&& x) { return x.first == hash; });
      if (opos == objmap.end()) {
        LOG(ERROR) << "No object list found for task " << taskname << " (hash=" << hash << ")";
        return;
      }
      auto objects = opos->second;
      if (std::find(objects.begin(), objects.end(), obj.name) == objects.end()) {
        LOG(ERROR) << "No object " << obj.name << " in map for task " << taskname;
        return;
      }
      auto nameHash = compile_time_hash(obj.name.c_str());
      InputObjectRoute key{obj.name, nameHash, taskname, hash, policy};
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
    "internal-dpl-global-analysis-file-sink",
    {InputSpec("y", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"HIST"}))},
    Outputs{},
    AlgorithmSpec(writerFunction),
    {}};

  return spec;
}

DataProcessorSpec CommonDataProcessors::getOutputObjSink(outputObjects const& objmap, outputTasks const& tskmap)
{
  auto writerFunction = [objmap, tskmap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto inputObjects = std::make_shared<std::vector<std::pair<InputObjectRoute, InputObject>>>();

    auto endofdatacb = [inputObjects](EndOfStreamContext& context) {
      LOG(DEBUG) << "Writing merged objects to file";
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
          (f[route.policy]->GetDirectory(currentDirectory.c_str()))->WriteObjectAny(entry.obj, entry.kind, entry.name.c_str());
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
      auto hash = objh->mTaskHash;

      obj.obj = tm.ReadObjectAny(obj.kind);
      TNamed* named = static_cast<TNamed*>(obj.obj);
      obj.name = named->GetName();
      auto hpos = std::find_if(tskmap.begin(), tskmap.end(), [&](auto&& x) { return x.first == hash; });
      if (hpos == tskmap.end()) {
        LOG(ERROR) << "No task found for hash " << hash;
        return;
      }
      auto taskname = hpos->second;
      auto opos = std::find_if(objmap.begin(), objmap.end(), [&](auto&& x) { return x.first == hash; });
      if (opos == objmap.end()) {
        LOG(ERROR) << "No object list found for task " << taskname << " (hash=" << hash << ")";
        return;
      }
      auto objects = opos->second;
      if (std::find(objects.begin(), objects.end(), obj.name) == objects.end()) {
        LOG(ERROR) << "No object " << obj.name << " in map for task " << taskname;
        return;
      }
      auto nameHash = compile_time_hash(obj.name.c_str());
      InputObjectRoute key{obj.name, nameHash, taskname, hash, policy};
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
    "internal-dpl-global-analysis-file-sink",
    {InputSpec("x", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"ATSK"}))},
    Outputs{},
    AlgorithmSpec(writerFunction),
    {}};

  return spec;
}

// add sink for the AODs
DataProcessorSpec
  CommonDataProcessors::getGlobalAODSink(std::vector<InputSpec> const& OutputInputs,
                                         std::vector<bool> const& isdangling)
{

  auto writerFunction = [OutputInputs, isdangling](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    LOG(DEBUG) << "======== getGlobalAODSink::Init ==========";

    auto dod = std::make_shared<DataOutputDirector>();

    // analyze ic and take actions accordingly
    // default values
    std::string fnbase("AnalysisResults");
    std::string filemode("RECREATE");
    int ntfmerge = 1;

    // values from json
    if (ic.options().isSet("json-file")) {
      auto fnjson = ic.options().get<std::string>("json-file");
      if (!fnjson.empty()) {
        auto [fnb, fmo, ntfm] = dod->readJson(fnjson);
        if (!fnb.empty()) {
          fnbase = fnb;
        }
        if (!fmo.empty()) {
          filemode = fmo;
        }
        if (ntfm > 0) {
          ntfmerge = ntfm;
        }
      }
    }

    // values from command line options, information from json is overwritten
    if (ic.options().isSet("res-file")) {
      fnbase = ic.options().get<std::string>("res-file");
    }
    if (ic.options().isSet("res-mode")) {
      filemode = ic.options().get<std::string>("res-mode");
    }
    if (ic.options().isSet("ntfmerge")) {
      auto ntfm = ic.options().get<int>("ntfmerge");
      if (ntfm > 0) {
        ntfmerge = ntfm;
      }
    }
    // parse the keepString
    if (ic.options().isSet("keep")) {
      dod->reset();
      auto keepString = ic.options().get<std::string>("keep");

      std::string d("dangling");
      if (d.find(keepString) == 0) {

        // use the dangling outputs
        std::vector<InputSpec> danglingOutputs;
        for (auto ii = 0; ii < OutputInputs.size(); ii++) {
          if (isdangling[ii]) {
            danglingOutputs.emplace_back(OutputInputs[ii]);
          }
        }
        dod->readSpecs(danglingOutputs);

      } else {

        // use the keep string
        dod->readString(keepString);
      }
    }
    dod->setFilenameBase(fnbase);

    // find out if any table needs to be saved
    bool hasOutputsToWrite = false;
    for (auto& outobj : OutputInputs) {
      auto ds = dod->getDataOutputDescriptors(outobj);
      if (ds.size() > 0) {
        hasOutputsToWrite = true;
        break;
      }
    }

    // if nothing needs to be saved then return a trivial functor
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

    // this functor is called once per time frame
    Int_t ntf = -1;
    return std::move([ntf, ntfmerge, filemode, dod](ProcessingContext& pc) mutable -> void {
      LOG(DEBUG) << "======== getGlobalAODSink::processing ==========";
      LOG(DEBUG) << " processing data set with " << pc.inputs().size() << " entries";

      // return immediately if pc.inputs() is empty
      auto ninputs = pc.inputs().size();
      if (ninputs == 0) {
        LOG(INFO) << "No inputs available!";
        return;
      }

      // increment the time frame counter ntf
      ntf++;

      // loop over the DataRefs which are contained in pc.inputs()
      for (const auto& ref : pc.inputs()) {

        // does this need to be saved?
        auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
        auto ds = dod->getDataOutputDescriptors(*dh);

        if (ds.size() > 0) {

          // get the TableConsumer and corresponding arrow table
          auto s = pc.inputs().get<TableConsumer>(ref.spec->binding);
          auto table = s->asArrowTable();
          if (!table->Validate().ok()) {
            LOGP(ERROR, "The table \"{}\" is not valid and will not be saved!", dh->description.str);
            continue;
          } else if (table->num_rows() <= 0) {
            LOGP(WARNING, "The table \"{}\" is empty and will not be saved!", dh->description.str);
            continue;
          }

          // loop over all DataOutputDescriptors
          // a table can be saved in multiple ways
          // e.g. different selections of columns to different files
          for (auto d : ds) {

            auto [file, directory] = dod->getFileFolder(d, ntf, ntfmerge, filemode);
            auto treename = directory + d->treename;
            TableToTree ta2tr(table,
                              file,
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
      }
    });
  }; // end of writerFunction

  DataProcessorSpec spec{
    "internal-dpl-aod-writer",
    OutputInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"json-file", VariantType::String, {"Name of the json configuration file"}},
     {"res-file", VariantType::String, {"Default name of the output file"}},
     {"res-mode", VariantType::String, {"Creation mode of the result files: NEW, CREATE, RECREATE, UPDATE"}},
     {"ntfmerge", VariantType::Int, {"Number of time frames to merge into one file"}},
     {"keep", VariantType::String, {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION:treename:col1/col2/..:filename"}}}};

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
  return specifyFairMQDeviceOutputProxy("internal-dpl-output-proxy", danglingOutputInputs, defaultChannelConfig.c_str());
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
