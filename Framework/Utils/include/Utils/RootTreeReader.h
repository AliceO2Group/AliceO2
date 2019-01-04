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

  operator Output() const { return Output{ origin, description, subSpec, lifetime }; }
};
}

/// @class RootTreeReader
/// A generic reader interface for ROOT TTrees
///
/// The reader interfaces one TTree specified by its name, and supports this
/// tree to be distributed over multiple files.
///
/// The class uses a KeyType to define sinks for specific branches, the default
/// key type is DPL `Output`. Branches are defined for processing by pairs of
/// key type and branch name.
/// Note that `Output` is probably going to be changed to `OutputDesc`
///
/// Usage (for the default KeyType):
///   RootTreeReader(treename,
///                  filename1, filename2, ...,
///                  Output{...}, branchname1,
///                  Output{...}, branchname2,
///                 ) reader;
///   auto processSomething = [] (auto& key, auto& object) {/*do something*/};
///   while (reader.next()) {
///     reader.process(processSomething);
///   }
///   // -- or --
///   while ((++reader)(processSomething));
///
/// In the DPL AlgorithmSpec, the processing lambda can simply look like
///   auto processingFct = [reader](ProcessingContext& pc) {
///     // increment the reader and invoke it for the processing context
///     (++reader)(pc);
///   };
/// Note that `reader` has to be set up in the init callback and it must
/// be static there to persist. It can also be a shared_pointer, which then
/// requires additional dereferencing in the syntax. The processing lambda has
/// to capture the shared pointer instance by copy.
template <typename KeyType>
class GenericRootTreeReader
{
 public:
  using self_type = GenericRootTreeReader<KeyType>;
  using key_type = KeyType;

  // the key must not be of type const char* to make sure that the variable argument
  // list of the constructor can be parsed
  static_assert(std::is_same<KeyType, const char*>::value == false, "the key type must not be const char*");

  /// default constructor
  GenericRootTreeReader();

  /// constructor
  /// @param treename  name of tree to process
  /// variable argument list of file names followed by pairs of KeyType and branch name
  template <typename... Args>
  GenericRootTreeReader(const char* treename, // name of the tree to read from
                        Args&&... args)       // file names, followed by branch info
    : mInput(treename)
  {
    mInput.SetCacheSize(0);
    parseConstructorArgs(std::forward<Args>(args)...);
  }

  /// constructor
  /// @param treename  name of tree to process
  /// @nMaxEntries maximum number of entries to be processed
  /// variable argument list of file names followed by pairs of KeyType and branch name
  template <typename... Args>
  GenericRootTreeReader(const char* treename, // name of the tree to read from
                        int nMaxEntries,      // max number of entries to be read
                        Args&&... args)       // file names, followed by branch info
    : mInput(treename),
      mMaxEntries(nMaxEntries)
  {
    mInput.SetCacheSize(0);
    parseConstructorArgs(std::forward<Args>(args)...);
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
    if ((mEntry + 1) >= mNEntries || mNEntries == 0) {
      // TODO: decide what to do, maybe different modes can be supported
      // e.g. loop or single shot mode
      if (mEntry < mNEntries) {
        ++mEntry;
      }
      return false;
    }
    if (mMaxEntries > 0 && (mEntry + 1) >= mMaxEntries) {
      if (mEntry < mNEntries) {
        ++mEntry;
      }
      return false;
    }
    ++mEntry;
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
      o2::header::Stack stack{ std::forward<HeaderTypes>(headers)... };
      context.outputs().snapshot(Output{ key.origin, key.description, key.subSpec, key.lifetime, std::move(stack) }, object);
    };

    return process(snapshot);
  }

  template <typename F>
  bool process(F&& snapshot) const
  {
    if (mEntry >= mNEntries || mNEntries == 0 || (mMaxEntries > 0 && mEntry >= mMaxEntries)) {
      return false;
    }

    for (auto& spec : mBranchSpecs) {
      char* data = nullptr;
      spec.second->branch->SetAddress(&data);
      spec.second->branch->GetEntry(mEntry);
      if (spec.second->sizebranch == nullptr) {
        snapshot(spec.first, ROOTSerializedByClass(*data, spec.second->classinfo));
      } else {
        size_t datasize = 0;
        spec.second->sizebranch->SetAddress(&datasize);
        spec.second->sizebranch->GetEntry(mEntry);
        auto* buffer = reinterpret_cast<std::vector<char>*>(data);
        if (buffer->size() == datasize) {
          LOG(INFO) << "branch " << spec.second->name << ": publishing binary chunk of " << datasize << " bytes(s)";
          snapshot(spec.first, *buffer);
        } else {
          LOG(ERROR) << "branch " << spec.second->name << ": inconsitent size of binary chunk "
                     << buffer->size() << " vs " << datasize;
          std::vector<char> empty;
          snapshot(spec.first, empty);
        }
      }
      spec.second->branch->DropBaskets("all");
    }
    return true;
  }

  /// return the number of published entries
  int getCount() const
  {
    return mEntry + 1;
  }

 private:
  struct BranchSpec {
    std::string name;
    TBranch* branch = nullptr;
    TBranch* sizebranch = nullptr;
    TClass* classinfo = nullptr;
  };

  /// add a new branch definition
  /// we allow for multiple branch definition for the same key
  void addBranchSpec(KeyType key, const char* branchName)
  {
    // right now we allow the same key to appear for multiple branches
    auto branch = mInput.GetBranch(branchName);
    if (branch) {
      mBranchSpecs.emplace_back(key, std::make_unique<BranchSpec>(BranchSpec{ branchName }));
      mBranchSpecs.back().second->branch = branch;
      std::string sizebranchName = std::string(branchName) + "Size";
      auto sizebranch = mInput.GetBranch(sizebranchName.c_str());
      if (!sizebranch) {
        mBranchSpecs.back().second->classinfo = TClass::GetClass(branch->GetClassName());
        LOG(INFO) << "branch set up: " << branchName;
      } else {
        auto* classinfo = TClass::GetClass(branch->GetClassName());
        if (classinfo == nullptr || classinfo != TClass::GetClass(typeid(std::vector<char>))) {
          throw std::runtime_error("mismatching class type, expecting std::vector<char> for binary branch");
        }
        mBranchSpecs.back().second->sizebranch = sizebranch;
        LOG(INFO) << "binary branch set up: " << branchName;
      }
    } else {
      std::string msg("can not find branch ");
      msg += branchName;
      throw std::runtime_error(msg);
    }
  }

  /// helper function to recursively parse constructor arguments
  /// this is the first part pasing all the file name, stops when the first key
  /// is found
  template <typename... Args>
  void parseConstructorArgs(const char* fileName, Args&&... args)
  {
    addFile(fileName);
    parseConstructorArgs(std::forward<Args>(args)...);
  }

  /// helper function to recursively parse constructor arguments
  /// parse the branch definitions with key and branch name.
  template <typename T, typename... Args>
  void parseConstructorArgs(T key, const char* name, Args&&... args)
  {
    if (name != nullptr && *name != 0) {
      // add branch spec if the name is not empty
      addBranchSpec(KeyType{ key }, name);
    }
    parseConstructorArgs(std::forward<Args>(args)...);
  }

  // this terminates the argument parsing
  void parseConstructorArgs() {}

  /// the input tree, using TChain to support multiple input files
  TChain mInput;
  /// definitions of branch specs
  std::vector<std::pair<KeyType, std::unique_ptr<BranchSpec>>> mBranchSpecs;
  /// number of entries in the tree
  int mNEntries = 0;
  /// current entry
  int mEntry = -1;
  /// maximum number of entries to be processed
  int mMaxEntries = -1;
};

using RootTreeReader = GenericRootTreeReader<rtr::DefaultKey>;

} // namespace framework
} // namespace o2
#endif
