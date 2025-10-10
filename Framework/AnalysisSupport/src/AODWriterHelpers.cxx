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
#include "Framework/AnalysisContext.h"
#include "Framework/ConfigContext.h"
#include "Framework/ControlService.h"
#include "AODWriterHelpers.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InitContext.h"
#include "Framework/CallbackService.h"
#include "Framework/AnalysisSupportHelpers.h"
#include "Framework/TableConsumer.h"
#include "Framework/DataOutputDirector.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/Monitoring.h"
#include "Framework/Signpost.h"

#include <Monitoring/Monitoring.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TFile.h>
#include <TTree.h>
#include <TMap.h>
#include <TObjString.h>
#include <arrow/table.h>
#include <chrono>
#include <ios>

O2_DECLARE_DYNAMIC_LOG(histogram_registry);

namespace o2::framework::writers
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
  std::string container;
  std::string name;
  int count = -1;
};

const static std::unordered_map<OutputObjHandlingPolicy, std::string> ROOTfileNames = {{OutputObjHandlingPolicy::AnalysisObject, "AnalysisResults.root"},
                                                                                       {OutputObjHandlingPolicy::QAObject, "QAResults.root"}};

AlgorithmSpec AODWriterHelpers::getOutputTTreeWriter(ConfigContext const& ctx)
{
  auto& ac = ctx.services().get<AnalysisContext>();
  auto dod = AnalysisSupportHelpers::getDataOutputDirector(ctx);
  int compressionLevel = 505;
  if (ctx.options().hasOption("aod-writer-compression")) {
    compressionLevel = ctx.options().get<int>("aod-writer-compression");
  }
  return AlgorithmSpec{[dod, outputInputs = ac.outputsInputsAOD, compressionLevel](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    LOGP(debug, "======== getGlobalAODSink::Init ==========");

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
          LOG(info) << "No AODs to be saved.";
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
    callbacks.set<CallbackService::Id::EndOfStream>(endofdatacb);

    // prepare map<uint64_t, uint64_t>(startTime, tfNumber)
    std::map<uint64_t, uint64_t> tfNumbers;
    std::map<uint64_t, std::string> tfFilenames;

    std::vector<TString> aodMetaDataKeys;
    std::vector<TString> aodMetaDataVals;

    // this functor is called once per time frame
    return [dod, tfNumbers, tfFilenames, aodMetaDataKeys, aodMetaDataVals, compressionLevel](ProcessingContext& pc) mutable -> void {
      LOGP(debug, "======== getGlobalAODSink::processing ==========");
      LOGP(debug, " processing data set with {} entries", pc.inputs().size());

      // return immediately if pc.inputs() is empty. This should never happen!
      if (pc.inputs().size() == 0) {
        LOGP(info, "No inputs available!");
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
      // update tfFilenames
      std::string aodInputFile;
      auto ref2 = pc.inputs().get("tff");
      if (ref2.spec && ref2.payload) {
        startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref2)->startTime;
        aodInputFile = pc.inputs().get<std::string>("tff");
        tfFilenames.insert(std::pair<uint64_t, std::string>(startTime, aodInputFile));
      }

      // close all output files if one has reached size limit
      dod->checkFileSizes();

      // loop over the DataRefs which are contained in pc.inputs()
      for (const auto& ref : pc.inputs()) {
        if (!ref.spec) {
          LOGP(debug, "Invalid input will be skipped!");
          continue;
        }

        // get metadata
        if (DataSpecUtils::partialMatch(*ref.spec, header::DataDescription("AODMetadataKeys"))) {
          aodMetaDataKeys = pc.inputs().get<std::vector<TString>>(ref.spec->binding);
        }
        if (DataSpecUtils::partialMatch(*ref.spec, header::DataDescription("AODMetadataVals"))) {
          aodMetaDataVals = pc.inputs().get<std::vector<TString>>(ref.spec->binding);
        }

        // skip non-AOD refs
        if (!DataSpecUtils::partialMatch(*ref.spec, writableAODOrigins)) {
          continue;
        }
        startTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->startTime;

        // does this need to be saved?
        auto dh = DataRefUtils::getHeader<header::DataHeader*>(ref);
        auto tableName = dh->dataDescription.as<std::string>();
        auto ds = dod->getDataOutputDescriptors(*dh);
        if (ds.empty()) {
          continue;
        }

        // get TF number from startTime
        auto it = tfNumbers.find(startTime);
        if (it != tfNumbers.end()) {
          tfNumber = (it->second / dod->getNumberTimeFramesToMerge()) * dod->getNumberTimeFramesToMerge();
        } else {
          LOGP(fatal, "No time frame number found for output with start time {}", startTime);
          throw std::runtime_error("Processing is stopped!");
        }
        // get aod input file from startTime
        auto it2 = tfFilenames.find(startTime);
        if (it2 != tfFilenames.end()) {
          aodInputFile = it2->second;
        }

        // get the TableConsumer and corresponding arrow table
        auto msg = pc.inputs().get(ref.spec->binding);
        if (msg.header == nullptr) {
          LOGP(error, "No header for message {}:{}", ref.spec->binding, DataSpecUtils::describe(*ref.spec));
          continue;
        }
        auto s = pc.inputs().get<TableConsumer>(ref.spec->binding);
        auto table = s->asArrowTable();
        if (!table->Validate().ok()) {
          LOGP(warning, "The table \"{}\" is not valid and will not be saved!", tableName);
          continue;
        }
        if (table->schema()->fields().empty()) {
          LOGP(debug, "The table \"{}\" is empty but will be saved anyway!", tableName);
        }

        // loop over all DataOutputDescriptors
        // a table can be saved in multiple ways
        // e.g. different selections of columns to different files
        for (auto d : ds) {
          auto fileAndFolder = dod->getFileFolder(d, tfNumber, aodInputFile, compressionLevel);
          auto treename = fileAndFolder.folderName + "/" + d->treename;
          TableToTree ta2tr(table,
                            fileAndFolder.file,
                            treename.c_str());

          // update metadata
          if (fileAndFolder.file->FindObjectAny("metaData")) {
            LOGF(debug, "Metadata: target file %s already has metadata, preserving it", fileAndFolder.file->GetName());
          } else if (!aodMetaDataKeys.empty() && !aodMetaDataVals.empty()) {
            TMap aodMetaDataMap;
            for (uint32_t imd = 0; imd < aodMetaDataKeys.size(); imd++) {
              aodMetaDataMap.Add(new TObjString(aodMetaDataKeys[imd]), new TObjString(aodMetaDataVals[imd]));
            }
            fileAndFolder.file->WriteObject(&aodMetaDataMap, "metaData", "Overwrite");
          }

          if (!d->colnames.empty()) {
            for (auto& cn : d->colnames) {
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
    };
  }

  };
}

AlgorithmSpec AODWriterHelpers::getOutputObjHistWriter(ConfigContext const& ctx)
{
  using namespace monitoring;
  auto& ac = ctx.services().get<AnalysisContext>();
  auto tskmap = ac.outTskMap;
  auto objmap = ac.outObjHistMap;

  return AlgorithmSpec{[objmap, tskmap](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto& callbacks = ic.services().get<CallbackService>();
    auto inputObjects = std::make_shared<std::vector<std::pair<InputObjectRoute, InputObject>>>();

    static TFile* f[OutputObjHandlingPolicy::numPolicies];
    for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
      f[i] = nullptr;
    }

    static std::string currentDirectory = "";
    static std::string currentFile = "";

    auto endofdatacb = [inputObjects](EndOfStreamContext& context) {
      LOG(debug) << "Writing merged objects and histograms to file";
      if (inputObjects->empty()) {
        LOG(error) << "Output object map is empty!";
        context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        return;
      }
      for (auto i = 0u; i < OutputObjHandlingPolicy::numPolicies; ++i) {
        if (f[i] != nullptr) {
          f[i]->Close();
        }
      }
      LOG(debug) << "All outputs merged in their respective target files";
      context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

    callbacks.set<CallbackService::Id::EndOfStream>(endofdatacb);
    return [inputObjects, objmap, tskmap](ProcessingContext& pc) mutable -> void {
      auto mergePart = [&inputObjects, &objmap, &tskmap, &pc](DataRef const& ref) {
        O2_SIGNPOST_ID_GENERATE(hid, histogram_registry);
        O2_SIGNPOST_START(histogram_registry, hid, "mergePart", "Merging histogram");
        if (!ref.header) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "Header not found.");
          return;
        }
        auto datah = o2::header::get<o2::header::DataHeader*>(ref.header);
        if (!datah) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "No data header in stack");
          return;
        }

        if (!ref.payload) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "Payload not found for %{public}s/%{public}s/%d",
                                     datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                                     datah->subSpecification);
          return;
        }

        auto objh = o2::header::get<o2::framework::OutputObjHeader*>(ref.header);
        if (!objh) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "No output object header in stack of %{public}s/%{public}s/%d.",
                                     datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                                     datah->subSpecification);
          return;
        }

        InputObject obj;
        FairInputTBuffer tm(const_cast<char*>(ref.payload), static_cast<int>(datah->payloadSize));
        tm.InitMap();
        obj.kind = tm.ReadClass();
        tm.SetBufferOffset(0);
        tm.ResetMap();
        O2_SIGNPOST_ID_GENERATE(did, histogram_registry);
        O2_SIGNPOST_START(histogram_registry, did, "initialising root", "Starting deserialization of %{public}s/%{public}s/%d",
                          datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                          datah->subSpecification);
        if (obj.kind == nullptr) {
          O2_SIGNPOST_END(histogram_registry, did, "initialising root", "Failed to deserialise");
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "Cannot read class info from buffer of %{public}s/%{public}s/%d.",
                                     datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                                     datah->subSpecification);
          return;
        }
        O2_SIGNPOST_END(histogram_registry, did, "initialising root", "Done init.");

        auto policy = objh->mPolicy;
        auto sourceType = objh->mSourceType;
        auto hash = objh->mTaskHash;
        O2_SIGNPOST_START(histogram_registry, did, "deserialization", "Starting deserialization of %{public}s/%{public}s/%d",
                          datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                          datah->subSpecification);

        obj.obj = tm.ReadObjectAny(obj.kind);
        auto* named = static_cast<TNamed*>(obj.obj);
        obj.name = named->GetName();
        O2_SIGNPOST_END(histogram_registry, did, "deserialization", "Done deserialization.");
        // If we have a folder, we assume the first element of the path
        // to be the name of the registry.
        if (sourceType == HistogramRegistrySource) {
          obj.container = objh->containerName;
        } else {
          obj.container = obj.name;
        }
        auto hpos = std::find_if(tskmap.begin(), tskmap.end(), [&](auto&& x) { return x.id == hash; });
        if (hpos == tskmap.end()) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "No task found for hash %d.", hash);
          return;
        }
        auto taskname = hpos->name;
        auto opos = std::find_if(objmap.begin(), objmap.end(), [&](auto&& x) { return x.id == hash; });
        if (opos == objmap.end()) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "No object list found for task %{public}s (hash=%d).",
                                     taskname.c_str(), hash);
          return;
        }
        auto objects = opos->bindings;
        if (std::find(objects.begin(), objects.end(), obj.container) == objects.end()) {
          O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "mergePart", "No container %{public}s in map for task %{public}s.",
                                     obj.container.c_str(), taskname.c_str());
          return;
        }
        auto nameHash = runtime_hash(obj.name.c_str());
        InputObjectRoute key{obj.name, nameHash, taskname, hash, policy, sourceType};
        auto existing = std::find_if(inputObjects->begin(), inputObjects->end(), [&](auto&& x) { return (x.first.uniqueId == nameHash) && (x.first.taskHash == hash); });
        // If it's the first one, we just add it to the list.
        O2_SIGNPOST_START(histogram_registry, did, "merging", "Starting merging of %{public}s/%{public}s/%d",
                          datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                          datah->subSpecification);
        if (existing == inputObjects->end()) {
          obj.count = objh->mPipelineSize;
          inputObjects->emplace_back(key, obj);
          existing = inputObjects->end() - 1;
        } else {
          obj.count = existing->second.count;
          // Otherwise, we merge it with the existing one.
          auto merger = existing->second.kind->GetMerge();
          if (!merger) {
            O2_SIGNPOST_END(histogram_registry, did, "merging", "Unabled to merge");
            O2_SIGNPOST_END_WITH_ERROR(histogram_registry, hid, "merging", "Already one unmergeable object found for %{public}s", obj.name.c_str());
            return;
          }
          TList coll;
          coll.Add(static_cast<TObject*>(obj.obj));
          merger(existing->second.obj, &coll, nullptr);
        }
        // We expect as many objects as the pipeline size, for
        // a given object name and task hash.
        existing->second.count -= 1;

        if (existing->second.count != 0) {
          O2_SIGNPOST_END(histogram_registry, did, "merging", "Done partial merging.");
          O2_SIGNPOST_END(histogram_registry, hid, "mergePart", "Pipeline lanes still missing.");
          return;
        }
        O2_SIGNPOST_END(histogram_registry, did, "merging", "Done merging.");
        // Write the object here.
        auto route = existing->first;
        auto entry = existing->second;
        auto file = ROOTfileNames.find(route.policy);
        if (file == ROOTfileNames.end()) {
          O2_SIGNPOST_END(histogram_registry, hid, "mergePart", "Not matching any file.");
          return;
        }
        O2_SIGNPOST_START(histogram_registry, did, "writing", "Starting writing of %{public}s/%{public}s/%d",
                          datah->dataOrigin.as<std::string>().c_str(), datah->dataDescription.as<std::string>().c_str(),
                          datah->subSpecification);
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

        // FIXME: handle folders
        f[route.policy]->cd("/");
        auto* currentDir = f[route.policy]->GetDirectory(currentDirectory.c_str());
        // The name contains a path...
        int objSize = 0;
        if (sourceType == HistogramRegistrySource) {
          TDirectory* currentFolder = currentDir;
          O2_SIGNPOST_EVENT_EMIT(histogram_registry, hid, "mergePart", "Toplevel folder is %{public}s.",
                                 currentDir->GetName());
          std::string objName = entry.name;
          auto lastSlash = entry.name.rfind('/');

          if (lastSlash != std::string::npos) {
            auto dirname = entry.name.substr(0, lastSlash);
            objName = entry.name.substr(lastSlash + 1);
            currentFolder = currentDir->GetDirectory(dirname.c_str());
            if (!currentFolder) {
              O2_SIGNPOST_EVENT_EMIT(histogram_registry, hid, "mergePart", "Creating folder %{public}s",
                                     dirname.c_str());
              currentFolder = currentDir->mkdir(dirname.c_str(), "", kTRUE);
            } else {
              O2_SIGNPOST_EVENT_EMIT(histogram_registry, hid, "mergePart", "Folder %{public}s already there.",
                                     currentFolder->GetName());
            }
          }
          O2_SIGNPOST_EVENT_EMIT(histogram_registry, hid, "mergePart", "Writing %{public}s of kind %{public}s in %{public}s",
                                 entry.name.c_str(), entry.kind->GetName(), currentDir->GetName());
          objSize = currentFolder->WriteObjectAny(entry.obj, entry.kind, objName.c_str());
          O2_SIGNPOST_END(histogram_registry, did, "writing", "End writing %{public}s", entry.name.c_str());
          delete (TObject*)entry.obj;
          entry.obj = nullptr;
        } else {
          O2_SIGNPOST_EVENT_EMIT(histogram_registry, hid, "mergePart", "Writing %{public}s of kind %{public}s in %{public}s",
                                 entry.name.c_str(), entry.kind->GetName(), currentDir->GetName());
          objSize = currentDir->WriteObjectAny(entry.obj, entry.kind, entry.name.c_str());
          O2_SIGNPOST_END(histogram_registry, did, "writing", "End writing %{public}s", entry.name.c_str());
          delete (TObject*)entry.obj;
          entry.obj = nullptr;
        }
        O2_SIGNPOST_END(histogram_registry, hid, "mergePart", "Done merging object of %d bytes.", objSize);
      };
      O2_SIGNPOST_ID_GENERATE(rid, histogram_registry);
      O2_SIGNPOST_START(histogram_registry, rid, "processParts", "Start merging %zu parts received together.", pc.inputs().getNofParts(0));
      for (int pi = 0; pi < pc.inputs().getNofParts(0); ++pi) {
        mergePart(pc.inputs().get("x", pi));
      }
      O2_SIGNPOST_END(histogram_registry, rid, "processParts", "Done histograms in multipart message.");
    };
  }};
}
} // namespace o2::framework::writers
