// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

#include "DPLUtils/RootTreeWriter.h"
#include "Framework/InputSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include <unordered_set>
#include <tuple>

namespace o2
{
namespace framework
{

/// @class MakeRootTreeWriterSpec
/// @brief Generate a processor spec for the RootTreeWriter utility.
///
/// A DPL processor is defined using o2::framework::DataProcessorSpec which holds the basic
/// information for the processor, i.e. inputs, the algorithm, and outputs. In addition, the
/// processor may support command line options.
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
/// Processing spec is generated by invoking the class as functor;
///
/// \par Target objects:
/// While the generic writer is primarily intended for ROOT serializable objects, a special
/// case is the writing of binary data when <tt>const char*<\tt> is used as type. Data is written
/// as a \c std::vector<char>, this ensures separation on event basis as well as having binary
/// data in parallel to ROOT objects in the same file, e.g. a binary data format from the
/// reconstruction in parallel to MC labels.
///
/// \par The processor spec is generated with the following options:
///
///     --outfile
///     --output-dir
///     --treename
///     --nevents
///     --autosave
///     --terminate
///
/// \par
/// In addition to that, a custom option can be added for every branch to configure the
/// branch name, see below.
///
/// \par Constructor arguments:
/// Default file name can be configured alone, tree name can only be specified after
/// file name. The default number of events can be specified at arbitrary place between
/// process name and branch configuration. The number of events triggering autosaving
/// (by default - off) can be also specified in the constructor as an integer argument
/// coming after (not necessarilly immidiately) the number or events. The process will
/// signal to the DPL that it is ready for termination.
///
/// \par Termination policy:
/// The configurable termination policy specifies what to signal to the DPL when the event
/// count is reached. Currently option 'process' terminates only the process and 'workflow'
/// all the workflow processes. The default option of this command line argument can be
/// chosen by using either TerminationPolicy::Workflow or TerminationPolicy::Process. without
/// custom configuration, the default argument is 'process'.
///
/// \par
/// Independently of the event counter, custom termination evaluation can be added using
/// MakeRootTreeWriterSpec::TerminationCondition. The two available actions determine whether
/// to do the processing of an input object or skip it. The two types of evaluators are
/// checked before or after extraction of an object from the input.
///
/// \par Usage:
///
///     WorkflowSpec specs;
///     specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                        (
///                          "process_name",
///                          "default_file_name",
///                          "default_tree_name",
///                          1,                                                   // default number of events
///                          MakeRootTreeWriterSpec::TerminationPolicy::Workflow, // terminate the workflow
///                          MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                          ...                             // further input and branch config
///                        )()                               // invocation of operator()
///                       );
///
///     // skipping defaults
///     specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                        (
///                          "process_name",
///                          MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                          ...                             // further input and branch config
///                        )()                               // invocation of operator()
///                       );
///
///     // only file name default
///     specs.emplace_back(MakeRootTreeWriterSpec<Types...>
///                        (
///                          "process_name",
///                          "default_file_name",
///                          MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname" },
///                          ...                             // further input and branch config
///                        )()                               // invocation of operator()
///                       );
///
/// \par Definition of branch name options:
/// The option key is specified as part of the branch definition, a key-value option is added
/// with branchname as default value
///   MakeRootTreeWriterSpec::BranchDefinition<Type>{ InputSpec{ ... }, "branchname", "optionkey" }
///
/// \par Advanced branch definition:
/// In order to write to multiple branches, the definition can be extended by
///   - number n of branches controlled by definition
///   - callback to calculate an index in the range [0,n-1] from the DataRef
///       argument(s): o2::framework::DataRef const&
///       result: size_t index, data set is skipped if ~(size_t)0
///   - callback to return the branch name for an index
///       argument(s): std::string basename, size_t index
///       result: std::string with branchname
///
/// \par Examples:
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
/// \par Custom termination condition, either one of:
///
///     auto checkProcessing = [](o2::framework::DataRef const& ref, bool& isReady) {
///       // decide on the DataRef whether to terminate or not
///       isReady = true;
///       // return true if the normal processing in this cycle should be carried out
///       // return false to skip
///       return false;
///     };
///     auto checkReady = [](o2::framework::DataRef const& ref) {
///       // decide on the DataRef whether to terminate or not
///       return true;
///     };
/// Now add MakeRootTreeWriterSpec::TerminationCondition{ checkReady } to the constructor
/// arguments. Only one of the callback types can be used with TerminationCondition.
class MakeRootTreeWriterSpec
{
 public:
  using WriterType = RootTreeWriter;
  enum struct TerminationPolicy {
    Process,
    Workflow,
  };
  const std::map<std::string, TerminationPolicy> TerminationPolicyMap = {
    {"process", TerminationPolicy::Process},
    {"workflow", TerminationPolicy::Workflow},
  };

  struct TerminationCondition {
    /// @struct Action
    /// @brief define the action to be done by the writer
    enum struct Action {
      /// carry out processing of the input object
      DoProcessing,
      /// skip processing of the input object
      SkipProcessing,
    };
    /// Callback to be checked before processing of an input object, return value determines whether to process
    /// or skip inputs, and whether to consider an input as 'ready'
    /// @param dataref  the DPL DataRef object
    /// @return std::tuple of <TerminationCondition::Action, bool>
    using CheckProcessing = std::function<std::tuple<TerminationCondition::Action, bool>(framework::DataRef const&)>;
    /// Callback to be checked after processing of an input object to check if process is ready
    /// @param dataref  the DPL DataRef object
    /// @return true if ready
    using CheckReady = std::function<bool(o2::framework::DataRef const&)>;

    /// the actual evaluator
    std::variant<std::monostate, CheckReady, CheckProcessing> check;
  };

  struct Preprocessor {
    /// processing callback
    using Process = std::function<void(ProcessingContext&)>;

    /// the callback
    std::variant<std::monostate, Process> callback;

    /// check if the Preprocessor can be executed
    constexpr operator bool()
    {
      return std::holds_alternative<Process>(callback);
    }

    /// execute the preprocessor
    void operator()(ProcessingContext& context)
    {
      if (std::holds_alternative<Process>(callback)) {
        std::get<Process>(callback)(context);
      }
    }
  };

  /// unary helper functor to extract the input key from the InputSpec
  struct KeyExtractor {
    static std::string asString(InputSpec const& arg) { return arg.binding; }
  };
  /// helper for the constructor selection below
  template <typename T, typename _ = void>
  struct StringAssignable : public std::false_type {
  };
  template <typename T>
  struct StringAssignable<
    T,
    std::enable_if_t<std::is_same_v<std::decay_t<T>, char const*> ||
                     std::is_same_v<std::decay_t<T>, char*> ||
                     std::is_same_v<std::decay_t<T>, std::string>>> : public std::true_type {
  };

  /// branch definition structure uses InputSpec as key type
  /// Main pupose is to support specification of a branch name option key in addition to all other
  /// branch definition arguments. The spec generator will add this as an option to the processor
  template <typename T>
  struct BranchDefinition : public WriterType::BranchDef<T, InputSpec, KeyExtractor> {
    /// constructor allows to specify an optional key used to generate a command line
    /// option, base class uses default parameters
    template <typename KeyType, typename Arg = const char*, std::enable_if_t<StringAssignable<Arg>::value, int> = 0>
    BranchDefinition(KeyType&& key, std::string _branchName, Arg _optionKey = "")
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName), optionKey(_optionKey)
    {
    }
    /// constructor allows to specify number of branches and an optional key used to generate
    /// a command line option, base class uses default parameters
    template <typename KeyType, typename Arg = const char*, std::enable_if_t<StringAssignable<Arg>::value, int> = 0>
    BranchDefinition(KeyType&& key, std::string _branchName, int _nofBranches, Arg _optionKey = "")
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName, _nofBranches), optionKey(_optionKey)
    {
    }
    /// constructor, if the first argument from the pack can be assigned to string this is treated
    /// as option key, all other arguments are simply forwarded to base class
    template <typename KeyType, typename Arg, typename... Args, std::enable_if_t<StringAssignable<Arg>::value, int> = 0>
    BranchDefinition(KeyType key, std::string _branchName, Arg&& _optionKey, Args&&... args)
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName, std::forward<Args>(args)...), optionKey(_optionKey)
    {
    }
    /// constructor, the argument pack is simply forwarded to base class
    template <typename KeyType, typename Arg, typename... Args, std::enable_if_t<!StringAssignable<Arg>::value, int> = 0>
    BranchDefinition(KeyType key, std::string _branchName, Arg&& arg, Args&&... args)
      : WriterType::BranchDef<T, InputSpec, KeyExtractor>(std::forward<KeyType>(key), _branchName, std::forward<Arg>(arg), std::forward<Args>(args)...), optionKey()
    {
    }
    /// key for command line option
    std::string optionKey = "";
  };

  // helper to define auxiliary inputs, i.e. inputs not connected to a branch
  struct AuxInputRoute {
    InputSpec mSpec;
    operator InputSpec() const
    {
      return mSpec;
    }
  };

  // helper to define tree attributes
  struct TreeAttributes {
    std::string name;
    std::string title = "";
  };

  // callback with signature void(TFile*, TTree*)
  using CustomClose = WriterType::CustomClose;

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
    struct ProcessAttributes {
      // worker instance
      std::shared_ptr<WriterType> writer;
      // keys for branch name options
      std::vector<std::pair<std::string, std::string>> branchNameOptions;
      // number of events to be processed
      int nEvents = -1;
      // autosave every nEventsAutoSave events
      int nEventsAutoSave = -1;
      // starting with all inputs, every input which has been indicated 'ready' is removed
      std::unordered_set<std::string> activeInputs;
      // event counter
      int counter = 0;
      // indicate what to terminate upon ready: process or workflow
      TerminationPolicy terminationPolicy = TerminationPolicy::Process;
      // custom termination condition
      TerminationCondition terminationCondition;
      // custom preprocessor
      Preprocessor preprocessor;
      // the total number of served branches on the n inputs
      size_t nofBranches;
    };
    auto processAttributes = std::make_shared<ProcessAttributes>();
    processAttributes->writer = mWriter;
    processAttributes->branchNameOptions = mBranchNameOptions;
    processAttributes->terminationCondition = std::move(mTerminationCondition);
    processAttributes->preprocessor = std::move(mPreprocessor);

    // set the list of active inputs, every input which is indecated as 'ready' will be removed from the list
    // the process is ready if the list is empty
    for (auto const& input : mInputs) {
      processAttributes->activeInputs.emplace(input.binding);
    }
    processAttributes->nofBranches = mNofBranches;

    // the init function is returned to the DPL in order to init the process
    auto initFct = [processAttributes, TerminationPolicyMap = TerminationPolicyMap](InitContext& ic) {
      auto& branchNameOptions = processAttributes->branchNameOptions;
      auto filename = ic.options().get<std::string>("outfile");
      auto treename = ic.options().get<std::string>("treename");
      auto treetitle = ic.options().get<std::string>("treetitle");
      auto outdir = ic.options().get<std::string>("output-dir");
      processAttributes->nEvents = ic.options().get<int>("nevents");
      if (processAttributes->nEvents > 0 && processAttributes->activeInputs.size() != processAttributes->nofBranches) {
        LOG(WARNING) << "the n inputs serve in total m branches with n != m, this means that there will be data for\n"
                     << "different branches on the same input. Be aware that the --nevents option might lead to incomplete\n"
                     << "data in the output file as the number of processed input sets is counted";
      }
      processAttributes->nEventsAutoSave = ic.options().get<int>("autosave");
      try {
        processAttributes->terminationPolicy = TerminationPolicyMap.at(ic.options().get<std::string>("terminate"));
      } catch (std::out_of_range&) {
        throw std::invalid_argument(std::string("invalid termination policy: ") + ic.options().get<std::string>("terminate"));
      }
      if (filename.empty() || treename.empty()) {
        throw std::invalid_argument("output file name and tree name are mandatory options");
      }
      for (size_t branchIndex = 0; branchIndex < branchNameOptions.size(); branchIndex++) {
        // pair of key (first) - value (second)
        if (branchNameOptions[branchIndex].first.empty()) {
          continue;
        }
        auto branchName = ic.options().get<std::string>(branchNameOptions[branchIndex].first.c_str());
        processAttributes->writer->setBranchName(branchIndex, branchName.c_str());
      }
      if (!outdir.empty() && outdir != "none") {
        if (outdir.back() != '/') {
          outdir += '/';
        }
        filename = outdir + filename;
      }
      processAttributes->writer->init(filename.c_str(), treename.c_str(), treetitle.c_str());
      // the callback to be set as hook at stop of processing for the framework
      auto finishWriting = [processAttributes]() {
        processAttributes->writer->close();
      };

      ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishWriting);

      auto processingFct = [processAttributes](ProcessingContext& pc) {
        auto& writer = processAttributes->writer;
        auto& terminationPolicy = processAttributes->terminationPolicy;
        auto& terminationCondition = processAttributes->terminationCondition;
        auto& preprocessor = processAttributes->preprocessor;
        auto& activeInputs = processAttributes->activeInputs;
        auto& counter = processAttributes->counter;
        auto& nEvents = processAttributes->nEvents;
        auto& nEventsAutoSave = processAttributes->nEventsAutoSave;
        if (writer->isClosed()) {
          return;
        }

        // if the termination condition contains function of type CheckProcessing, this is checked
        // before the processing, currently implemented logic:
        // - processing is skipped if this is indicated by at least one input
        // - if input is 'ready' it is removed from the list of active inputs
        auto checkProcessing = [&terminationCondition, &activeInputs](auto const& inputs) {
          bool doProcessing = true;
          if (std::holds_alternative<TerminationCondition::CheckProcessing>(terminationCondition.check)) {
            auto& check = std::get<TerminationCondition::CheckProcessing>(terminationCondition.check);
            for (auto const& ref : inputs) {
              auto iter = activeInputs.find(ref.spec->binding);
              auto [action, ready] = check(ref);
              if (action == TerminationCondition::Action::SkipProcessing) {
                // this condition tells us not to process the data any further
                doProcessing = false;
              }
              if (iter != activeInputs.end() && ready) {
                // this input is ready, remove from active inputs
                activeInputs.erase(iter);
              }
            }
          }
          return doProcessing;
        };
        // if the termination condition contains function of type CheckReady, this is checked
        // after the processing. All inputs marked 'ready' are removed from the list of active inputs,
        // process treated as 'ready' if no active inputs
        auto checkReady = [&terminationCondition, &activeInputs](auto const& inputs) {
          if (std::holds_alternative<TerminationCondition::CheckReady>(terminationCondition.check)) {
            auto& check = std::get<TerminationCondition::CheckReady>(terminationCondition.check);
            for (auto const& ref : inputs) {
              auto iter = activeInputs.find(ref.spec->binding);
              if (iter != activeInputs.end() && check(ref)) {
                // this input is ready, remove from active inputs
                activeInputs.erase(iter);
              }
            }
          }
          return activeInputs.size() == 0;
        };

        if (preprocessor) {
          preprocessor(pc);
        }
        if (checkProcessing(pc.inputs())) {
          (*writer)(pc.inputs());
          counter = counter + 1;
        }

        if ((nEvents >= 0 && counter == nEvents) || checkReady(pc.inputs())) {
          writer->close();
          pc.services().get<ControlService>().readyToQuit(terminationPolicy == TerminationPolicy::Workflow ? QuitRequest::All : QuitRequest::Me);
        } else if (nEventsAutoSave > 0 && counter && (counter % nEventsAutoSave) == 0) {
          writer->autoSave();
        }
      };

      return processingFct;
    };

    Options options{
      // default options
      {"outfile", VariantType::String, mDefaultFileName.c_str(), {"Name of the output file"}},
      {"output-dir", VariantType::String, mDefaultDir.c_str(), {"Output directory"}},
      {"treename", VariantType::String, mDefaultTreeName.c_str(), {"Name of tree"}},
      {"treetitle", VariantType::String, mDefaultTreeTitle.c_str(), {"Title of tree"}},
      {"nevents", VariantType::Int, mDefaultNofEvents, {"Number of events to execute"}},
      {"autosave", VariantType::Int, mDefaultAutoSave, {"Autosave after number of events"}},
      {"terminate", VariantType::String, mDefaultTerminationPolicy.c_str(), {"Terminate the 'process' or 'workflow'"}},
    };
    for (size_t branchIndex = 0; branchIndex < mBranchNameOptions.size(); branchIndex++) {
      // adding option definitions for those ones defined in the branch definition
      if (mBranchNameOptions[branchIndex].first.empty()) {
        continue;
      }
      options.push_back(ConfigParamSpec(mBranchNameOptions[branchIndex].first.c_str(),  // option key
                                        VariantType::String,                            // option argument type
                                        mBranchNameOptions[branchIndex].second.c_str(), // default branch name
                                        {"configurable branch name"}                    // help message
                                        ));
    }

    mInputRoutes.insert(mInputRoutes.end(), mInputs.begin(), mInputs.end());
    return DataProcessorSpec{
      // processing spec generated from the class configuartion
      mProcessName.c_str(),   // name of the process
      mInputRoutes,           // list of inputs
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
      mDefaultTreeTitle = name;
    }

    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// specialization for the in argument as default for nevents option
  template <size_t N, typename... Args>
  void parseConstructorArgs(int arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree options must come before branch specs");
    if (mNIntArgCounter == 0) {
      mDefaultNofEvents = arg;
    } else if (mNIntArgCounter == 1) {
      mDefaultAutoSave = arg;
    } else {
      throw std::logic_error("Too many integer arguments in the constructor");
    }
    mNIntArgCounter++;
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
  void parseConstructorArgs(TerminationCondition&& arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree, and all options must come before branch specs");
    mTerminationCondition = std::move(arg);
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// specialization for parsing optional preprocessor definition
  template <size_t N, typename... Args>
  void parseConstructorArgs(Preprocessor&& arg, Args&&... args)
  {
    static_assert(N == 0, "wrong argument order, default file and tree, and all options must come before branch specs");
    mPreprocessor = std::move(arg);
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  template <size_t N, typename... Args>
  void parseConstructorArgs(AuxInputRoute&& aux, Args&&... args)
  {
    mInputRoutes.emplace_back(aux);
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  template <size_t N, typename... Args>
  void parseConstructorArgs(CustomClose&& callback, Args&&... args)
  {
    mCustomClose = callback;
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  template <size_t N, typename... Args>
  void parseConstructorArgs(TreeAttributes&& att, Args&&... args)
  {
    mDefaultTreeName = att.name;
    mDefaultTreeTitle = att.title.empty() ? att.name : att.title;
    parseConstructorArgs<N>(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// parse the branch definitions and store the input specs.
  /// Note: all other properties of the branch definition are handled in the
  /// constructor of the writer itself
  template <size_t N, typename T, typename... Args>
  void parseConstructorArgs(BranchDefinition<T>&& def, Args&&... args)
  {
    if (def.nofBranches > 0) {
      // number of branches set to 0 will skip the definition, this allows to
      // dynamically disable branches, while all possible definitions can
      // be specified at compile time
      mInputs.insert(mInputs.end(), def.keys.begin(), def.keys.end());
      mBranchNameOptions.emplace_back(def.optionKey, def.branchName);
      mNofBranches += def.nofBranches;
    } else {
      // insert an empty placeholder
      mBranchNameOptions.emplace_back("", "");
    }
    parseConstructorArgs<N + 1>(std::forward<Args>(args)...);
    if constexpr (N == 0) {
      mWriter = std::make_shared<WriterType>(nullptr, nullptr, mCustomClose, std::forward<BranchDefinition<T>>(def), std::forward<Args>(args)...);
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
  std::vector<InputSpec> mInputRoutes;
  std::vector<std::pair<std::string, std::string>> mBranchNameOptions;
  std::string mDefaultFileName;
  std::string mDefaultTreeName;
  std::string mDefaultTreeTitle;
  std::string mDefaultDir = "none";
  int mDefaultNofEvents = -1;
  int mDefaultAutoSave = -1;
  std::string mDefaultTerminationPolicy = "process";
  TerminationCondition mTerminationCondition;
  Preprocessor mPreprocessor;
  size_t mNofBranches = 0;
  int mNIntArgCounter = 0;
  CustomClose mCustomClose;
};
} // namespace framework
} // namespace o2

#endif // FRAMEWORK_MAKEROOTTREEWRITERSPEC_H
