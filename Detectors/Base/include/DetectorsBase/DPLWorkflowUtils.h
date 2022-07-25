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

// some user-space helpers/utilities for DPL workflowspecs

//
// Created by Sandro Wenzel on 19.04.22.
//

#ifndef O2_DPLWORKFLOWUTILS_H
#define O2_DPLWORKFLOWUTILS_H

#include "Framework/RootSerializationSupport.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include <vector>
#include <unordered_map>

namespace o2
{
namespace framework
{

// Finding out if the current process is the master DPL driver process,
// first setting up the topology. Might be important to know when we write
// files (to prevent that multiple processes write the same file)
bool isMasterWorkflowDefinition(ConfigContext const& configcontext)
{
  int argc = configcontext.argc();
  auto argv = configcontext.argv();
  bool ismaster = true;
  for (int argi = 0; argi < argc; ++argi) {
    // when channel-config is present it means that this is started as
    // as FairMQDevice which means it is already a forked process
    if (strcmp(argv[argi], "--channel-config") == 0) {
      ismaster = false;
      break;
    }
  }
  return ismaster;
}

// Finding out if we merely want to dump the DPL workflow json file.
// In this case we could avoid some computation/initialization, when
// this doesn't influence the topology.
bool isDumpWorkflowInvocation(ConfigContext const& configcontext)
{
  int argc = configcontext.argc();
  auto argv = configcontext.argv();
  bool isdump = false;
  for (int argi = 0; argi < argc; ++argi) {
    if (strcmp(argv[argi], "--dump-config") == 0) {
      isdump = true;
      break;
    }
  }
  return isdump;
}

// find out if we are an internal DPL device
// (given the device name)
bool isInternalDPL(std::string const& name)
{
  if (name.find("internal-dpl-") != std::string::npos) {
    return true;
  }
  return false;
}

// find out the name of THIS DPL device at runtime
// May be useful to know to prevent certain initializations for
// say internal DPL devices or writer devices
std::string whoAmI(ConfigContext const& configcontext)
{
  int argc = configcontext.argc();
  auto argv = configcontext.argv();
  // the name of this device is the string following the --id field in
  // the argv invocation
  for (int argi = 0; argi < argc; ++argi) {
    if (strcmp(argv[argi], "--id") == 0) {
      if (argi + 1 < argc) {
        return std::string(argv[argi + 1]);
      }
    }
  }
  return std::string("");
}

// a utility combining multiple specs into one
// (with some checking that it makes sense)
// spits out the combined spec (merged inputs/outputs and AlgorithmSpec)
// Can put policies later whether to multi-thread or serialize internally etc.
// fills a "remaining" spec container for things it can't simply merge
// (later on we could do a full topological sort / spec minimization approach)
o2::framework::DataProcessorSpec specCombiner(std::string const& name, std::vector<DataProcessorSpec> const& speccollection,
                                              std::vector<DataProcessorSpec>& remaining)
{
  std::vector<OutputSpec> combinedOutputSpec;
  std::vector<InputSpec> combinedInputSpec;
  std::vector<ConfigParamSpec> combinedOptions;

  // this maps an option key to one of multiple specs where this is used
  // we will need to introduce unambiguous options in case an option is present multiple times
  std::unordered_map<std::string, std::vector<std::pair<std::string, ConfigParamSpec>>> optionMerger;

  // maps a process name to a map holder conversion of "namespaced-keys" to original "keys"
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> optionsRemap;

  // keep track of which input bindings are already used
  // (should not have duplicates ... since devices may fetch data just based on the binding)
  std::unordered_map<std::string, bool> inputBindings;

  // we collect all outputs once ---> this is to check that none of the inputs matches
  // an output
  std::vector<OutputSpec> allOutputs;
  std::vector<DataProcessorSpec> mergableSpecs;
  for (auto& spec : speccollection) {
    // merge output specs
    for (auto& os : spec.outputs) {
      allOutputs.push_back(os);
    }
  }

  for (auto& spec : speccollection) {
    auto& procname = spec.name;
    optionsRemap[procname] = std::unordered_map<std::string, std::string>();

    // merge input specs ... but only after we have verified that the spec does
    // not depend in internal outputs
    bool inputCheckOk = true;
    for (auto& is : spec.inputs) {
      // let's see if input is part of outputs
      // ... in which case we can't easily merge the spec here
      // ... and just neglect it for the moment
      for (auto& o : allOutputs) {
        if (DataSpecUtils::match(is, o)) {
          std::cerr << "Found internal connection " << is << " ... will take out spec " << procname << " .. from merging process \n";
          inputCheckOk = false;
          break;
        }
      }
    }
    if (!inputCheckOk) {
      remaining.push_back(spec);
      // directly to next task
      continue;
    }
    for (auto& is : spec.inputs) {
      if (inputBindings.find(is.binding) != inputBindings.end()) {
        LOG(error) << "Found duplicate input binding " << is.binding;
      }
      combinedInputSpec.push_back(is);
      inputBindings[is.binding] = 1;
    }
    mergableSpecs.push_back(spec);

    // merge output specs
    for (auto& os : spec.outputs) {
      combinedOutputSpec.push_back(os);
    }
    // merge options (part 1)
    for (auto& opt : spec.options) {
      auto optkey = opt.name;
      auto iter = optionMerger.find(optkey);
      auto procconfigpair = std::pair<std::string, ConfigParamSpec>(procname, opt);
      if (iter == optionMerger.end()) {
        optionMerger[optkey] = std::vector<std::pair<std::string, ConfigParamSpec>>();
      }
      optionMerger[optkey].push_back(procconfigpair);
    }
  }
  // merge options (part 2)
  for (auto& iter : optionMerger) {
    if (iter.second.size() > 1) {
      // std::cerr << "Option " << iter.first << " duplicated in multiple procs --> applying namespacing \n";
      for (auto& part : iter.second) {
        auto procname = part.first;
        auto originalSpec = part.second;
        auto namespaced_name = procname + "." + originalSpec.name;
        combinedOptions.push_back(ConfigParamSpec{namespaced_name,
                                                  originalSpec.type, originalSpec.defaultValue, originalSpec.help, originalSpec.kind});
        optionsRemap[procname][namespaced_name] = originalSpec.name;
        // we need to back-apply
      }
    } else {
      // we can stay with original option
      for (auto& part : iter.second) {
        combinedOptions.push_back(part.second);
      }
    }
  }

  // logic for combined task processing function --> target is to run one only
  class CombinedTask
  {
   public:
    CombinedTask(std::vector<DataProcessorSpec> const& s, std::unordered_map<std::string, std::unordered_map<std::string, std::string>> optionsRemap) : tasks(s), optionsRemap(optionsRemap) {}

    void init(o2::framework::InitContext& ic)
    {
      std::cerr << "Init Combined\n";
      for (auto& t : tasks) {
        // the init function actually creates the onProcess function
        // which we have to do here (maybe some more stuff needed)
        auto& configRegistry = ic.mOptions;
        // we can get hold of the store because the store is the only data member of configRegistry
        static_assert(sizeof(o2::framework::ConfigParamRegistry) == sizeof(std::unique_ptr<o2::framework::ConfigParamStore>));
        auto store = reinterpret_cast<std::unique_ptr<ConfigParamStore>*>(&(configRegistry));
        auto& boost_store = (*store)->store(); // std::unique_ptr<boost::property_tree::ptree>
        auto& originalDeviceName = t.name;
        auto optionsiter = optionsRemap.find(originalDeviceName);
        if (optionsiter != optionsRemap.end()) {
          // we have options to remap
          for (auto& key : optionsiter->second) {
            //
            // LOG(info) << "Applying value " << boost_store.get<std::string>(key.first) << " to original key " << key.second;
            boost_store.put(key.second, boost_store.get<std::string>(key.first));
          }
        }
        t.algorithm.onProcess = t.algorithm.onInit(ic);
      }
    }

    void run(o2::framework::ProcessingContext& pc)
    {
      std::cerr << "Processing Combined\n";
      for (auto& t : tasks) {
        std::cerr << " Executing sub-device " << t.name << "\n";
        t.algorithm.onProcess(pc);
      }
    }

   private:
    std::vector<DataProcessorSpec> tasks;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> optionsRemap;
  };

  return DataProcessorSpec{
    name,
    combinedInputSpec,
    combinedOutputSpec,
    AlgorithmSpec{adaptFromTask<CombinedTask>(mergableSpecs, optionsRemap)},
    combinedOptions
    /* a couple of other fields can be set ... */
  };
};

} // namespace framework
} // namespace o2

#endif // O2_DPLWORKFLOWUTILS_H
