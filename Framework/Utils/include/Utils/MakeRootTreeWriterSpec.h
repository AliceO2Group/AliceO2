// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MAKEROOTTREEWRITERSPEC_H
#define FRAMEWORK_MAKEROOTTREEWRITERSPEC_H

/// @file   MakeRootTreeWriterSpec.h
/// @author Matthias Richter
/// @since  2018-05-15
/// @brief  Configurable generator for RootTreeWriter processor spec

#include "Utils/RootTreeWriter.h"
#include "Framework/InputSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>

namespace o2
{
namespace framework
{

/// @class MakeRootTreeWriterSpec
/// Generate a processor spec for the RootTreeWriter utility.
///
/// The generator is configured with the process name and a list of branch definitions.
/// Each branch definition holds the type of the data object to be extracted and stored,
/// InputSpec to describe the input route, and branchname to describe output branch.
/// Further optional arguments directly after the process name can change the behavior
/// of spec options defaults.
///
/// The processor spec is generated with the following options:
///   --outfile
///   --treename
///   --nevents
///
/// Default file name can be configured alone, tree name can only be specified after
/// file name. The default number of events can be specified at arbitrary place between
/// process name and branch configuration.
///
/// Processing spec is generated by invoking the class as functor;
///
/// Usage:
///   WorkflowSpec specs;
///   specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                      (
///                        "process_name",
///                        "default_file_name",
///                        "default_tree_name",
///                        1,                              // default number of events
///                        MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                        ...                             // further input and branch config
///                      )()                               // invocation of operator()
///                     );
///
///   // skipping defaults
///   specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                      (
///                        "process_name",
///                        MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                        ...                             // further input and branch config
///                      )()                               // invocation of operator()
///                     );
///
///   // only file name default
///   specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                      (
///                        "process_name",
///                        "default_file_name",
///                        MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                        ...                             // further input and branch config
///                      )()                               // invocation of operator()
///                     );
///
class MakeRootTreeWriterSpec
{
 public:
  using WriterType = RootTreeWriter;

  /// unary helper functor to extract the input key from the InputSpec
  struct KeyExtractor {
    static std::string asString(InputSpec const& arg) { return arg.binding; }
  };

  // branch definition structure uses InputSpec as key type
  template <typename T>
  using BranchDefinition = WriterType::BranchDef<T, InputSpec, KeyExtractor>;

  /// default constructor forbidden
  MakeRootTreeWriterSpec() = delete;

  template <typename... Args>
  MakeRootTreeWriterSpec(const char* processName, Args&&... args) : mProcessName(processName)
  {
    parseConstructorArgs<0>(std::forward<Args>(args)...);
  }

  DataProcessorSpec operator()()
  {
    auto initFct = [writer = mWriter](InitContext & ic)
    {
      auto filename = ic.options().get<std::string>("outfile");
      auto treename = ic.options().get<std::string>("treename");
      auto nEvents = ic.options().get<int>("nevents");
      auto counter = std::make_shared<int>();
      *counter = 0;
      if (filename.empty() || treename.empty()) {
        throw std::invalid_argument("output file name and tree name are mandatory options");
      }
      writer->init(filename.c_str(), treename.c_str());

      // the callback to be set as hook at stop of processing for the framework
      auto finishWriting = [writer]() { writer->close(); };
      ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

      auto processingFct = [writer, nEvents, counter](ProcessingContext& pc) {
        (*writer)(pc.inputs());
        *counter = *counter + 1;
        if (nEvents >= 0 && *counter >= nEvents) {
          pc.services().get<ControlService>().readyToQuit(true);
        }
      };

      return processingFct;
    };

    return DataProcessorSpec{
      // processing spec generated from the class configuartion
      mProcessName.c_str(),   // name of the process
      mInputs,                // list of inputs
      Outputs{},              // no outputs
      AlgorithmSpec(initFct), // return the init function
      Options{                // default options
               { "outfile", VariantType::String, mDefaultFileName.c_str(), { "Name of the output file" } },
               { "treename", VariantType::String, mDefaultTreeName.c_str(), { "Name of tree" } },
               { "nevents", VariantType::Int, mDefaultNofEvents, { "Number of events to execute" } } }
    };
  }

 private:
  /// helper function to recursively parse constructor arguments
  /// the default file and tree name can come before all the branch specs
  template <size_t N, typename... Args>
  void parseConstructorArgs(const char* name, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree options must come before branch specs");
    // this can be called twice, the first time we set the default file name
    // and if we are here for a second time, we set the default tree name
    if (mDefaultFileName.empty()) {
      mDefaultFileName = name;
    } else {
      mDefaultTreeName = name;
    }

    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// specialization for the in argument as default for nevents option
  template <size_t N, typename... Args>
  void parseConstructorArgs(int arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree options must come before branch specs");
    mDefaultNofEvents = arg;

    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// parse the branch definitions and store the input specs.
  /// Note: all other properties of the branch definition are handled in the
  /// constructor of the writer itself
  template <size_t N, typename T, typename... Args>
  void parseConstructorArgs(BranchDefinition<T>&& def, Args&&... args)
  {
    mInputs.emplace_back(def.key);
    LOG(INFO) << "adding input spec: " << mInputs.back();
    parseConstructorArgs<N + 1>(std::forward<Args>(args)...);
    if (N == 0) {
      mWriter = std::make_shared<WriterType>(nullptr, nullptr, std::forward<BranchDefinition<T>>(def), std::forward<Args>(args)...);
    }
  }

  // this terminates the argument parsing
  template <size_t N>
  void parseConstructorArgs()
  {
  }

  std::shared_ptr<WriterType> mWriter;
  std::string mProcessName;
  std::vector<InputSpec> mInputs;
  std::string mDefaultFileName;
  std::string mDefaultTreeName;
  int mDefaultNofEvents = -1;
};
}
}

#endif // FRAMEWORK_MAKEROOTTREEWRITERSPEC_H
