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
/// A branch definition is always bound to a data type, the advanced version supports
/// multiple branches for the same data type. See further down.
///
/// While the generic writer is primarily intended for ROOT serializable objects, a special
/// case is the writing of binary data when const char* is used as type. Data is written
/// as a std::vector<char>, this ensures separation on event basis as well as having binary
/// data in parallel to ROOT objects in the same file, e.g. a binary data format from the
/// reconstruction in parallel to MC labels.
///
/// The processor spec is generated with the following options:
///   --outfile
///   --treename
///   --nevents
///   --terminate
///
/// In addition to that, a custom option can be added for every branch to configure the
/// branch name, see below.
///
/// Default file name can be configured alone, tree name can only be specified after
/// file name. The default number of events can be specified at arbitrary place between
/// process name and branch configuration. The process will signal to the DPL that it
/// is ready for termination.
///
/// Termination policy:
/// The configurable termination policy specifies what to signal to the DPL when the event
/// count is reached. Currently option 'process' terminates only the process and 'workflow'
/// all the workflow processes. The default option of this command line argument can be
/// chosen by using either TerminationPolicy::Workflow or TerminationPolicy::Process. without
/// custom configuration the default argument is 'process'. Independently of the event count,
/// a custom termination condition can be added with MakeRootTreeWriterSpec::TerminationCondition.
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
///                        1,                                                   // default number of events
///                        MakeRootTreeWriterSpec::TerminationPolicy::Workflow, // terminate the workflow
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
/// Definition of branch name options:
/// The option key is specified as part of the branch definition, a key-value option is added
/// with branchname as default value
///   MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname", "optionkey" }
///
/// Advanced branch definition:
/// In order to write to multiple branches, the definition can be extended by
///   - number n of branches controlled by definition
///   - callback to calculate an index in the range [0,n-1] from the DataRef
///       argument(s): o2::framework::DataRef const&
///       result: size_t index, data set is skipped if ~(size_t)0
///   - callback to return the branch name for an index
///       argument(s): std::string basename, size_t index
///       result: std::string with branchname
///
/// Examples:
/// Multiple inputs for the same data type can be handled by the same branch definition
/// by specifying a vector of InputSpecs instead of a single InputSpec. The number of branches
/// has to be equal or larger than the size of the vector.
///
///     template <typename T>
///     using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
///
///     // the callback to retrieve index from DataHeader
///     auto getIndex = [](o2::framework::DataRef const& ref) {
///       using DataHeader = o2::header::DataHeader;
///       auto const* dataHeader = o2::framework::DataRefUtils::getHeader<DataHeader*>(ref);
///       return dataHeader->subSpecification;
///     };
///
///     // the callback for the branch name
///     auto getName = [](std::string base, size_t index) {
///       return base + " " + std::to_string(index);
///     };
///
///     // assuming 4 inputs distinguished by DataHeader sub specification
///     std::vector<InputSpec> inputs{ {"...", "...", 0}, ... , {"...", "...", 3} };
///
///     // advanced branch definition
///     auto def = BranchDefinition<DataType>{ inputs,
///                                            "branchname",
///                                            4,
///                                            getIndex,
///                                            getName };
///
/// A single input can also be distributed to multiple branches if the getIndex
/// callback calculates the index from another piece of information in the
/// header stack.
///
/// Custom termination condition:
///     auto checkReady = [](o2::framework::DataRef const& ref) {
///       // decide on the DataRef whether to terminate or not
///       return true;
///     };
/// Now add MakeRootTreeWriterSpec::TerminationCondition{ checkReady } to the constructor
/// arguments.
class MakeRootTreeWriterSpec
{
 public:
  using WriterType = RootTreeWriter;
  enum struct TerminationPolicy {
    Process,
    Workflow,
  };
  const std::map<std::string, TerminationPolicy> TerminationPolicyMap = {
    { "process", TerminationPolicy::Process },
    { "workflow", TerminationPolicy::Workflow },
  };

  struct TerminationCondition {
    using CheckReady = std::function<bool(o2::framework::DataRef const&)>;
    CheckReady check;
  };

  /// unary helper functor to extract the input key from the InputSpec
  struct KeyExtractor {
    static std::string asString(InputSpec const& arg) { return arg.binding; }
  };

  // branch definition structure uses InputSpec as key type
  template <typename T>
  struct BranchDefinition : public WriterType::BranchDef<T, InputSpec, KeyExtractor> {
    /// constructor allows to specify an optional key used to generate a command line
    /// option, all other parameters are simply forwarded to base class
    template <typename KeyType>
    BranchDefinition(KeyType&& key, std::string _branchName, std::string _optionKey = "")
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName), optionKey(_optionKey)
    {
    }
    /// constructor, all parameters are simply forwarded to base class
    template <typename KeyType, typename... Args>
    BranchDefinition(KeyType key, std::string _branchName, std::string _optionKey, Args&&... args)
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName, std::forward<Args>(args)...), optionKey(_optionKey)
    {
    }
    /// key for command line option
    std::string optionKey = "";
  };

  /// default constructor forbidden
  MakeRootTreeWriterSpec() = delete;

  template <typename... Args>
  MakeRootTreeWriterSpec(const char* processName, Args&&... args) : mProcessName(processName)
  {
    parseConstructorArgs<0>(std::forward<Args>(args)...);
  }

  DataProcessorSpec operator()()
  {
    // this creates the workflow spec
    auto initFct = [branchNameOptions = mBranchNameOptions, writer = mWriter, TerminationPolicyMap = TerminationPolicyMap, terminationCondition = mTerminationCondition](InitContext& ic) {
      auto filename = ic.options().get<std::string>("outfile");
      auto treename = ic.options().get<std::string>("treename");
      auto nEvents = ic.options().get<int>("nevents");
      TerminationPolicy terminationPolicy = TerminationPolicy::Process;
      try {
        terminationPolicy = TerminationPolicyMap.at(ic.options().get<std::string>("terminate"));
      } catch (std::out_of_range&) {
        throw std::invalid_argument(std::string("invalid termination policy: ") + ic.options().get<std::string>("terminate"));
      }
      auto counter = std::make_shared<int>();
      *counter = 0;
      if (filename.empty() || treename.empty()) {
        throw std::invalid_argument("output file name and tree name are mandatory options");
      }
      for (size_t branchIndex = 0; branchIndex < branchNameOptions.size(); branchIndex++) {
        // pair of key (first) - value (second)
        if (branchNameOptions[branchIndex].first.empty()) {
          continue;
        }
        auto branchName = ic.options().get<std::string>(branchNameOptions[branchIndex].first.c_str());
        writer->setBranchName(branchIndex, branchName.c_str());
      }
      writer->init(filename.c_str(), treename.c_str());

      // the callback to be set as hook at stop of processing for the framework
      auto finishWriting = [writer]() { writer->close(); };
      ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

      auto processingFct = [writer, nEvents, counter, terminationPolicy, terminationCondition](ProcessingContext& pc) {
        if (writer->isClosed()) {
          return;
        }
        (*writer)(pc.inputs());
        *counter = *counter + 1;

        auto checkTerminationCondition = [terminationCondition](auto const& inputs) {
          if (terminationCondition.check) {
            for (auto const& ref : inputs) {
              if (terminationCondition.check(ref)) {
                return true;
              }
            }
          }
          return false;
        };
        if ((nEvents >= 0 && *counter == nEvents) || checkTerminationCondition(pc.inputs())) {
          writer->close();
          pc.services().get<ControlService>().readyToQuit(terminationPolicy == TerminationPolicy::Workflow);
        }
      };

      return processingFct;
    };

    Options options{
      // default options
      { "outfile", VariantType::String, mDefaultFileName.c_str(), { "Name of the output file" } },
      { "treename", VariantType::String, mDefaultTreeName.c_str(), { "Name of tree" } },
      { "nevents", VariantType::Int, mDefaultNofEvents, { "Number of events to execute" } },
      { "terminate", VariantType::String, mDefaultTerminationPolicy.c_str(), { "Terminate the 'process' or 'workflow'" } },
    };
    for (size_t branchIndex = 0; branchIndex < mBranchNameOptions.size(); branchIndex++) {
      // adding option definitions for those ones defined in the branch definition
      if (mBranchNameOptions[branchIndex].first.empty()) {
        continue;
      }
      options.push_back(ConfigParamSpec(mBranchNameOptions[branchIndex].first.c_str(),  // option key
                                        VariantType::String,                            // option argument type
                                        mBranchNameOptions[branchIndex].second.c_str(), // default branch name
                                        { "configurable branch name" }                  // help message
                                        ));
    }

    return DataProcessorSpec{
      // processing spec generated from the class configuartion
      mProcessName.c_str(),   // name of the process
      mInputs,                // list of inputs
      Outputs{},              // no outputs
      AlgorithmSpec(initFct), // return the init function
      std::move(options),     // processor options
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

  /// specialization for parsing default for termination policy
  template <size_t N, typename... Args>
  void parseConstructorArgs(TerminationPolicy arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree, and all options must come before branch specs");
    for (const auto& policy : TerminationPolicyMap) {
      if (policy.second == arg) {
        mDefaultTerminationPolicy = policy.first;
        return parseConstructorArgs<N>(std::forward<Args>(args)...);
      }
    }
    // here we only get if the enum and map definitions are inconsistent
    throw std::logic_error("Internal mismatch of policy ids and keys");
  }

  /// specialization for parsing optional termination condition
  template <size_t N, typename... Args>
  void parseConstructorArgs(TerminationCondition arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree, and all options must come before branch specs");
    mTerminationCondition = arg;
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// parse the branch definitions and store the input specs.
  /// Note: all other properties of the branch definition are handled in the
  /// constructor of the writer itself
  template <size_t N, typename T, typename... Args>
  void parseConstructorArgs(BranchDefinition<T>&& def, Args&&... args)
  {
    mInputs.insert(mInputs.end(), def.keys.begin(), def.keys.end());
    mBranchNameOptions.emplace_back(def.optionKey, def.branchName);
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
  std::vector<std::pair<std::string, std::string>> mBranchNameOptions;
  std::string mDefaultFileName;
  std::string mDefaultTreeName;
  int mDefaultNofEvents = -1;
  std::string mDefaultTerminationPolicy = "process";
  TerminationCondition mTerminationCondition;
};
}
}

#endif // FRAMEWORK_MAKEROOTTREEWRITERSPEC_H
