// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Framework/Output.h"
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

namespace o2
{
namespace framework
{

namespace rtr
{
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
/// by pairs of key type and branch name.
///
/// \par Usage, with some KeyType:
/// The first argument is the tree name, the list of files is variable, all files
/// must be passed before the first key-branchname pair.
///
///     RootTreeReader(treename,
///                    filename1, filename2, ...,
///                    KeyType{...}, branchname1,
///                    KeyType{...}, branchname2,
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
///                                                    Output{...}, branchname2,
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
  // the key must not be of type const char* to make sure that the variable argument
  // list of the constructor can be parsed
  static_assert(std::is_same<KeyType, const char*>::value == false, "the key type must not be const char*");

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
    auto snapshot = [&context, &headers...](const KeyType& key, const auto& object) {
      o2::header::Stack stack{std::forward<HeaderTypes>(headers)...};
      context.outputs().snapshot(Output{key.origin, key.description, key.subSpec, key.lifetime, std::move(stack)}, object);
    };

    return process(snapshot);
  }

  template <typename F>
  bool process(F&& snapshot) const
  {
    if (mReadEntry >= mNEntries || mNEntries == 0 || (mMaxEntries > 0 && mNofPublished >= mMaxEntries)) {
      return false;
    }

    for (auto& spec : mBranchSpecs) {
      char* data = nullptr;
      spec.second->branch->SetAddress(&data);
      spec.second->branch->GetEntry(mReadEntry);
      if (spec.second->sizebranch == nullptr) {
        snapshot(spec.first, std::move(ROOTSerializedByClass(*data, spec.second->classinfo)));
      } else {
        size_t datasize = 0;
        spec.second->sizebranch->SetAddress(&datasize);
        spec.second->sizebranch->GetEntry(mReadEntry);
        auto* buffer = reinterpret_cast<BinaryDataStoreType*>(data);
        if (buffer->size() == datasize) {
          LOG(INFO) << "branch " << spec.second->name << ": publishing binary chunk of " << datasize << " bytes(s)";
          snapshot(spec.first, std::move(*buffer));
        } else {
          LOG(ERROR) << "branch " << spec.second->name << ": inconsitent size of binary chunk "
                     << buffer->size() << " vs " << datasize;
          BinaryDataStoreType empty;
          snapshot(spec.first, empty);
        }
      }
      auto* delfunc = spec.second->classinfo->GetDelete();
      if (delfunc) {
        (*delfunc)(data);
      }
      spec.second->branch->DropBaskets("all");
    }
    return true;
  }

  /// return the number of published entries
  int getCount() const
  {
    return mNofPublished + 1;
  }

 private:
  struct BranchSpec {
    std::string name;
    TBranch* branch = nullptr;
    TBranch* sizebranch = nullptr;
    TClass* classinfo = nullptr;
  };

  // helper for the invalid code path of if constexpr statement
  template <typename T>
  struct type_dependent : std::false_type {
  };

  /// add a new branch definition
  /// we allow for multiple branch definition for the same key
  void addBranchSpec(KeyType key, const char* branchName)
  {
    // right now we allow the same key to appear for multiple branches
    auto branch = mInput.GetBranch(branchName);
    if (branch) {
      mBranchSpecs.emplace_back(key, std::make_unique<BranchSpec>(BranchSpec{branchName}));
      mBranchSpecs.back().second->branch = branch;
      std::string sizebranchName = std::string(branchName) + "Size";
      auto sizebranch = mInput.GetBranch(sizebranchName.c_str());
      auto* classinfo = TClass::GetClass(branch->GetClassName());
      if (!sizebranch) {
        if (classinfo == nullptr) {
          throw std::runtime_error(std::string("can not find class description for branch ") + branchName);
        }
        LOG(INFO) << "branch set up: " << branchName;
      } else {
        if (classinfo == nullptr || classinfo != TClass::GetClass(typeid(BinaryDataStoreType))) {
          throw std::runtime_error("mismatching class type, expecting std::vector<char> for binary branch");
        }
        mBranchSpecs.back().second->sizebranch = sizebranch;
        LOG(INFO) << "binary branch set up: " << branchName;
      }
      mBranchSpecs.back().second->classinfo = classinfo;
    } else {
      throw std::runtime_error(std::string("can not find branch ") + branchName);
    }
  }

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
    if constexpr (skip > 0) {
      return parseConstructorArgs<skip - 1>(std::forward<Args>(args)...);
    }
    if constexpr (std::is_same<U, const char*>::value) {
      addFile(key);
    } else if constexpr (std::is_same<U, int>::value) {
      mMaxEntries = key;
    } else if constexpr (std::is_same<U, PublishingMode>::value) {
      mPublishingMode = key;
    } else if constexpr (sizeof...(Args) > 0) {
      const char* arg = getCharArg(std::forward<Args>(args)...);
      if (arg != nullptr && *arg != 0) {
        // add branch spec if the name is not empty
        addBranchSpec(KeyType{key}, arg);
      }
      return parseConstructorArgs<1>(std::forward<Args>(args)...);
    } else {
      static_assert(type_dependent<U>::value, "argument mismatch, allowed are: file names, int to specify number of events, publishing mode, and key-branchname pairs");
    }
    parseConstructorArgs<0>(std::forward<Args>(args)...);
  }

  // this terminates the argument parsing
  template <size_t skip>
  void parseConstructorArgs()
  {
  }

  /// the input tree, using TChain to support multiple input files
  TChain mInput;
  /// definitions of branch specs
  std::vector<std::pair<KeyType, std::unique_ptr<BranchSpec>>> mBranchSpecs;
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
};

using RootTreeReader = GenericRootTreeReader<rtr::DefaultKey>;

} // namespace framework
} // namespace o2
#endif
