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
#ifndef FRAMEWORK_ROOTTREEREADER_H
#define FRAMEWORK_ROOTTREEREADER_H

/// @file   RootTreeReader.h
/// @author Matthias Richter
/// @since  2018-03-15
/// @brief  A generic reader for ROOT TTrees

#include "Framework/RootSerializationSupport.h"
#include "Framework/Output.h"
#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/RootMessageContext.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include <TChain.h>
#include <TTree.h>
#include <TBranch.h>
#include <TClass.h>
#include <vector>
#include <string>
#include <stdexcept> // std::runtime_error
#include <type_traits>
#include <memory>     // std::make_unique
#include <functional> // std::function
#include <utility>    // std::forward

namespace o2::framework
{

namespace rtr
{
// The class 'Output' has a unique header stack member which makes the class not
// usable as key, in particular there is no copy constructor. Thats why a dedicated
// structure is used which can be transformed into Output. Moreover it also handles the
// conversion from OutputStack to Output
struct DefaultKey {
  header::DataOrigin origin;
  header::DataDescription description;
  header::DataHeader::SubSpecificationType subSpec = 0;
  enum Lifetime lifetime = Lifetime::Timeframe;

  DefaultKey(const Output& desc)
    : origin(desc.origin), description(desc.description), subSpec(desc.subSpec), lifetime(desc.lifetime)
  {
  }

  operator Output() const { return Output{origin, description, subSpec, lifetime}; }
};
} // namespace rtr

/// @class RootTreeReader
/// @brief A generic reader interface for ROOT TTrees
///
/// The reader interfaces one TTree specified by its name, and supports this
/// tree to be distributed over multiple files.
///
/// The class uses a KeyType to define sinks for specific branches, the default
/// key type is DPL \ref framework::Output. Branches are defined for processing
/// by either pairs of key type and branch name, or by struct BranchDefinition.
/// The latter allows strong typing, the type is specified as template parameter
/// and the DPL IO decides upon serialization method.
///
/// Output data is always ROOT serialized if no type is specified.
///
/// \par Usage, with some KeyType:
/// The first argument is the tree name, the list of files is variable, all files
/// must be passed before the first key-branchname pair.
///
///     RootTreeReader(treename,
///                    filename1, filename2, ...,
///                    KeyType{...}, branchname1,
///                    BranchDefinition<type>{KeyType{...}, branchname2},
///                   ) reader;
///     auto processSomething = [] (auto& key, auto& object) {
///       // do something
///     };
///     while (reader.next()) {
///       reader.process(processSomething);
///     }
///
/// \par Further optional constructor arguments
/// Optional arguments can follow in arbitrary sequence, but all arguments must be passed
/// before the first branch definition.
///    - number of entries \a n (\c int)
///    - publishing mode, \ref PublishingMode
///
/// \par Integration into DPL:
/// The class defines the functor operator which takes the DPL ProcessingContext
/// as argument and optionally an argument pack of headers which inherit from BaseHeader,
/// the latter will form the header stack of the message.
///
///     auto reader = std::make_shared<RootTreeReader>(treename,
///                                                    filename1, filename2, ...,
///                                                    Output{...}, branchname1,
///                                                    BranchDefinition<type>{Output{...}, branchname2},
///                                                   );
///     // In the DPL AlgorithmSpec, the processing lambda can simply look like:
///     // (note that the shared pointer is propagated by copy)
///     auto processingFct = [reader](ProcessingContext& pc) {
///       // increment the reader and invoke it for the processing context
///       if (reader->next()) {
///         SomeHeader specificHeader;
///         reader(pc, specificHeader);
///       } else {
///         // no more data
///       }
///     }
///
///     // another example, simply incrementing by operator++ and providing additional
///     // custom header
///     auto someOtherProcessingFct = [reader](ProcessingContext& pc) {
///       // increment the reader and invoke it for the processing context
///       SomeHeader specificHeader;
///       if ((++reader)(pc, specificHeader) == false) {
///         // no more data
///       }
///     }
///
/// \par Binary format:
/// The reader supports the binary format of the RootTreeWriter as counterpart.
/// Binary data is stored as vector of char alongside with a branch storing the
/// size, as both indicator and consistency check.
///
/// \note
/// In the examples, `reader` has to be set up in the init callback and it must
/// be static there to persist. It can also be a shared_pointer, which then
/// requires additional dereferencing in the syntax. The processing lambda has
/// to capture the shared pointer instance by copy.
template <typename KeyType>
class GenericRootTreeReader
{
 public:
  using self_type = GenericRootTreeReader<KeyType>;
  using key_type = KeyType;
  // The RootTreeWriter/Reader support storage of binary data as a vector of char.
  // This allows to store e.g. raw data or some intermediate binary data alongside
  // with ROOT objects like the MC labels.
  using BinaryDataStoreType = std::vector<char>;

  /// Publishing mode determines what to do when the number of entries in the tree is reached
  enum struct PublishingMode {
    /// no more data after end of tree
    Single,
    /// start over at entry 0
    Loop,
  };

  /// A struct holding a callback to specialize publishing based on branch name. This might be useful
  /// when the read data needs to be transformed before publishing.
  /// The hook gets the following input:
  /// name: name of branch (may be used for selecting filtering)
  /// context: The processing context (so that we can snapshot or publish)
  /// Output: The DPL output channel on which to publish
  /// data: pointer to the data object read by the TreeReader
  /// The hook should return true when publishing succeeded and false when the ordinary publishing procedure
  /// should proceed.
  struct SpecialPublishHook {
    std::function<bool(std::string_view name, ProcessingContext& context, Output const&, char* data)> hook;
  };

  // the key must not be of type const char* to make sure that the variable argument
  // list of the constructor can be parsed
  static_assert(std::is_same<KeyType, const char*>::value == false, "the key type must not be const char*");

  /// helper structure to hold the constructor arguments for BranchConfigurationElement stages
  struct ConstructorArg {
    ConstructorArg(key_type _key, const char* _name)
      : key(_key), name(_name) {}
    ConstructorArg(key_type _key, std::string const& _name)
      : key(_key), name(_name) {}
    key_type key;
    std::string name;
  };

  // the vector of arguments is filled during the build up of the branch configuration and
  // passed to the construction of the constructed mixin class
  using ConstructorArgs = std::vector<ConstructorArg>;

  /// @class BranchConfigurationInterface
  /// The interface for the branch configuration. The branch configuration is constructed at
  /// compile time from the constructor argments of the tree reader. A mixin class is constructed
  /// from nested instances of @class BranchConfigurationElement, each holding a type information
  /// and the runtime configuration for the specific branch.
  ///
  /// The purpose of this interface is to provide the foundation of the mixin class and the virtual
  /// interface for setup and exec to enter in the upper most stage of the mixin.
  class BranchConfigurationInterface
  {
   public:
    static const size_t STAGE = 0;
    BranchConfigurationInterface() = default;
    BranchConfigurationInterface(ConstructorArgs const&){};
    virtual ~BranchConfigurationInterface() = default;

    /// Setup the branch configuration, namely get the branch and class information from the tree
    virtual void setup(TTree&, SpecialPublishHook* h = nullptr) {}
    /// Run the reader process
    /// This will recursively run every level of the the branch configuration and fetch the object
    /// at position \a entry. The object is published via the DPL ProcessingContext. A creator callback
    /// for the header stack is provided to build the stack from the variadic list of header template
    /// arguments.
    virtual void exec(ProcessingContext& ctx, int entry, std::function<o2::header::Stack()> stackcreator) {}

   private:
  };

  /// one element in the branch configuration structure
  /// it contains the previous element as base class and is bound to a data type.
  template <typename DataT, typename BASE>
  class BranchConfigurationElement : public BASE
  {
   public:
    using PrevT = BASE;
    using value_type = DataT;
    using publish_type = void;
    static const size_t STAGE = BASE::STAGE + 1;
    BranchConfigurationElement() = default;
    BranchConfigurationElement(ConstructorArgs const& args)
      : PrevT(args), mKey(args[STAGE - 1].key), mName(args[STAGE - 1].name)
    {
    }
    ~BranchConfigurationElement() override = default;

    /// Run the reader process
    /// This is the virtal overload entry point to the upper most stage of the branch configuration
    void exec(ProcessingContext& ctx, int entry, std::function<o2::header::Stack()> stackcreator) override
    {
      process(ctx, entry, stackcreator);
    }

    /// Setup branch configuration
    /// This is the virtal overload entry point to the upper most stage of the branch configuration
    void setup(TTree& tree, SpecialPublishHook* publishhook = nullptr) override
    {
      setupInstance(tree, publishhook);
    }

    /// Run the setup, first recursively for all lower stages, and then the current stage
    /// This fetches the branch corresponding to the configured name
    void setupInstance(TTree& tree, SpecialPublishHook* publishhook = nullptr)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      if constexpr (STAGE > 1) {
        PrevT::setupInstance(tree, publishhook);
      }
      mPublishHook = publishhook;

      // right now we allow the same key to appear for multiple branches
      mBranch = tree.GetBranch(mName.c_str());
      if (mBranch) {
        std::string sizebranchName = std::string(mName) + "Size";
        auto sizebranch = tree.GetBranch(sizebranchName.c_str());
        auto* classinfo = TClass::GetClass(mBranch->GetClassName());
        if constexpr (not std::is_void<value_type>::value) {
          // check if the configured type matches the stored type
          auto* storedclass = TClass::GetClass(typeid(value_type));
          if (classinfo != storedclass) {
            throw std::runtime_error(std::string("Configured type ") +
                                     (storedclass != nullptr ? storedclass->GetName() : typeid(value_type).name()) +
                                     " does not match the stored data type " +
                                     (classinfo != nullptr ? classinfo->GetName() : "") +
                                     " in branch " + mName);
          }
        }
        if (!sizebranch) {
          if (classinfo == nullptr) {
            throw std::runtime_error(std::string("can not find class description for branch ") + mName);
          }
          LOG(info) << "branch set up: " << mName;
        } else {
          if (classinfo == nullptr || classinfo != TClass::GetClass(typeid(BinaryDataStoreType))) {
            throw std::runtime_error("mismatching class type, expecting std::vector<char> for binary branch");
          }
          mSizeBranch = sizebranch;
          LOG(info) << "binary branch set up: " << mName;
        }
        mClassInfo = classinfo;
      } else {
        throw std::runtime_error(std::string("can not find branch ") + mName);
      }
    }

    /// Run the reader, first recursively for all lower stages, and then the current stage
    void process(ProcessingContext& context, int entry, std::function<o2::header::Stack()>& stackcreator)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      if constexpr (STAGE > 1) {
        PrevT::process(context, entry, stackcreator);
      }

      auto snapshot = [&context, &stackcreator](const KeyType& key, const auto& object) {
        context.outputs().snapshot(Output{key.origin, key.description, key.subSpec, key.lifetime, std::move(stackcreator())}, object);
      };

      char* data = nullptr;
      mBranch->SetAddress(&data);
      mBranch->GetEntry(entry);

      // execute hook if it was registered; if this return true do not proceed further
      if (mPublishHook != nullptr && (*mPublishHook).hook(mName, context, Output{mKey.origin, mKey.description, mKey.subSpec, mKey.lifetime, std::move(stackcreator())}, data)) {

      }
      // try to figureout when we need to do something special
      else {
        if (mSizeBranch != nullptr) {
          size_t datasize = 0;
          mSizeBranch->SetAddress(&datasize);
          mSizeBranch->GetEntry(entry);
          auto* buffer = reinterpret_cast<BinaryDataStoreType*>(data);
          if (buffer->size() == datasize) {
            LOG(info) << "branch " << mName << ": publishing binary chunk of " << datasize << " bytes(s)";
            snapshot(mKey, std::move(*buffer));
          } else {
            LOG(error) << "branch " << mName << ": inconsitent size of binary chunk "
                       << buffer->size() << " vs " << datasize;
            BinaryDataStoreType empty;
            snapshot(mKey, empty);
          }
        } else {
          if constexpr (std::is_void<value_type>::value == true) {
            // the default branch configuration publishes the object ROOT serialized
            snapshot(mKey, std::move(ROOTSerializedByClass(*data, mClassInfo)));
          } else {
            // if type is specified in the branch configuration, the allocator API decides
            // upon serialization
            snapshot(mKey, *reinterpret_cast<value_type*>(data));
          }
        }
      }
      // cleanup the memory
      auto* delfunc = mClassInfo->GetDelete();
      if (delfunc) {
        (*delfunc)(data);
      }
      mBranch->DropBaskets("all");
    }

   private:
    key_type mKey;
    std::string mName;
    TBranch* mBranch = nullptr;
    TBranch* mSizeBranch = nullptr;
    TClass* mClassInfo = nullptr;
    SpecialPublishHook* mPublishHook = nullptr;
  };

  /// branch definition structure
  /// This is a helper class to pass a branch definition to the reader constructor. The branch definition
  /// is bound to a concrete type which will be used to determin the serialization method at DPL output.
  /// The key parameter describes the DPL output, the name parameter to branch name to publish.
  template <typename T>
  struct BranchDefinition {
    using type = T;
    template <typename U>
    BranchDefinition(U _key, const char* _name)
      : key(_key), name(_name)
    {
    }
    template <typename U>
    BranchDefinition(U _key, std::string const& _name)
      : key(_key), name(_name)
    {
    }
    template <typename U>
    BranchDefinition(U _key, std::string&& _name)
      : key(_key), name(std::move(_name))
    {
    }

    key_type key;
    std::string name;
  };

  /// default constructor
  GenericRootTreeReader();

  /// constructor
  /// @param treename  name of tree to process
  /// variable argument list of file names (const char*), the number of entries to publish (int), or
  /// the publishing mode, followed by pairs of KeyType and branch name
  template <typename... Args>
  GenericRootTreeReader(const char* treename, // name of the tree to read from
                        Args&&... args)       // file names, followed by branch info
    : mInput(treename)
  {
    mInput.SetCacheSize(0);
    parseConstructorArgs<0>(std::forward<Args>(args)...);
    mBranchConfiguration->setup(mInput, mPublishHook);
  }

  /// add a file as source for the tree
  void addFile(const char* fileName)
  {
    mInput.AddFile(fileName);
    mNEntries = mInput.GetEntries();
  }

  /// move to the next entry
  /// @return true if data is available
  bool next()
  {
    if ((mReadEntry + 1) >= mNEntries || mNEntries == 0) {
      if (mPublishingMode == PublishingMode::Single) {
        // stop here
        if (mReadEntry < mNEntries) {
          mReadEntry = mNEntries;
        }
        return false;
      }
      // start over in loop mode
      mReadEntry = -1;
    }
    if (mMaxEntries > 0 && (mNofPublished + 1) >= mMaxEntries) {
      if (mReadEntry < mNEntries) {
        mReadEntry = mNEntries;
      }
      return false;
    }
    ++mReadEntry;
    ++mNofPublished;
    return true;
  }

  /// prefix increment, move to the next entry
  self_type& operator++()
  {
    next();
    return *this;
  }
  /// postfix increment forbidden
  self_type& operator++(int) = delete;

  /// the type wrapper to mark the data to be ROOT serialized, object is passed
  /// by not-type-aware char*, and the actual class info is provided.
  using ROOTSerializedByClass = o2::framework::ROOTSerialized<char, TClass>;

  /// process functor
  /// It expects a context which is used by lambda capture in the snapshot function.
  /// Loop over all branch definitions and publish by using the snapshot function.
  /// The default forwards to DPL DataAllocator snapshot.
  ///
  /// Note: For future extension we probably get rid of the context and want to use
  /// o2::snapshot, can be easily adjusted by exchanging the lambda.
  template <typename ContextType, typename... HeaderTypes>
  bool operator()(ContextType& context,
                  HeaderTypes&&... headers) const
  {
    if (mReadEntry >= mNEntries || mNEntries == 0 || (mMaxEntries > 0 && mNofPublished >= mMaxEntries)) {
      return false;
    }

    auto stackcreator = [&headers...]() {
      return o2::header::Stack{std::forward<HeaderTypes>(headers)...};
    };

    mBranchConfiguration->exec(context, mReadEntry, stackcreator);
    return true;
  }

  /// return the number of published entries
  int getCount() const
  {
    return mNofPublished + 1;
  }

 private:
  // special helper to get the char argument from the argument pack
  template <typename T, typename... Args>
  const char* getCharArg(T arg, Args&&...)
  {
    static_assert(std::is_same<T, const char*>::value, "missing branch name after publishing key, use const char* argument right after the key");
    return arg;
  }

  /// helper function to recursively parse constructor arguments
  template <size_t skip, typename U, typename... Args>
  void parseConstructorArgs(U key, Args&&... args)
  {
    // all argument parsing is done in the creation of the branch configuration
    mBranchConfiguration = createBranchConfiguration<0, BranchConfigurationInterface>({}, std::forward<U>(key), std::forward<Args>(args)...);
  }

  /// recursively step through all branch definitions from the command line arguments
  /// and build all types nested into a mixin type. Each level of the mixin derives
  /// from the previous level and by that one can later step through the list.
  template <size_t skip, typename BASE, typename U, typename... Args>
  std::unique_ptr<BranchConfigurationInterface> createBranchConfiguration(ConstructorArgs&& cargs, U def, Args&&... args)
  {
    if constexpr (skip > 0) {
      return createBranchConfiguration<skip - 1, BASE>(std::move(cargs), std::forward<Args>(args)...);
    } else if constexpr (std::is_same<U, const char*>::value) {
      addFile(def);
    } else if constexpr (std::is_same<U, int>::value) {
      mMaxEntries = def;
    } else if constexpr (std::is_same<U, PublishingMode>::value) {
      mPublishingMode = def;
    } else if constexpr (std::is_same<U, SpecialPublishHook*>::value) {
      mPublishHook = def;
    } else if constexpr (is_specialization_v<U, BranchDefinition>) {
      cargs.emplace_back(key_type(def.key), def.name);
      using type = BranchConfigurationElement<typename U::type, BASE>;
      return std::move(createBranchConfiguration<0, type>(std::move(cargs), std::forward<Args>(args)...));
    } else if constexpr (sizeof...(Args) > 0) {
      const char* arg = getCharArg(std::forward<Args>(args)...);
      if (arg != nullptr && *arg != 0) {
        cargs.emplace_back(key_type(def), arg);
        using type = BranchConfigurationElement<void, BASE>;
        return std::move(createBranchConfiguration<1, type>(std::move(cargs), std::forward<Args>(args)...));
      }
      throw std::runtime_error("expecting valid branch name string after key");
    } else {
      static_assert(always_static_assert<U>::value, "argument mismatch, define branches either as argument pairs of key and branchname or using the BranchDefinition helper struct");
    }
    return createBranchConfiguration<0, BASE>(std::move(cargs), std::forward<Args>(args)...);
  }

  /// the final method of the recursive argument parsing
  /// the mixin type is now fully constructed and the configuration object is created
  template <size_t skip, typename T>
  std::unique_ptr<BranchConfigurationInterface> createBranchConfiguration(ConstructorArgs&& cargs)
  {
    static_assert(skip == 0);
    return std::move(std::make_unique<T>(cargs));
  }

  /// the input tree, using TChain to support multiple input files
  TChain mInput;
  /// configuration of branches
  std::unique_ptr<BranchConfigurationInterface> mBranchConfiguration;
  /// number of entries in the tree
  int mNEntries = 0;
  /// current read position
  int mReadEntry = -1;
  /// nof of published entries
  int mNofPublished = -1;
  /// maximum number of entries to be processed
  int mMaxEntries = -1;
  /// publishing mode
  PublishingMode mPublishingMode = PublishingMode::Single;
  /// special user hook
  SpecialPublishHook* mPublishHook = nullptr;
};

using RootTreeReader = GenericRootTreeReader<rtr::DefaultKey>;

} // namespace o2::framework
#endif
