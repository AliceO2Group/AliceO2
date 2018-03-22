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

#include "Framework/OutputSpec.h"
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

/// @class RootTreeReader
/// A generic reader interface for ROOT TTrees
///
/// The reader interfaces one TTree specified by its name, and supports this
/// tree to be distributed over multiple files multiple files.
///
/// The class uses a KeyType to define sinks for specific branches, the default
/// key type is OutputSpec. Branches are defined for processing by pairs of
/// key type and branch name.
///
/// Usage (for the default KeyType):
///   RootTreeReader(treename,
///                  filename1, filename2, ...,
///                  OutputSpec{...}, branchname1,
///                  OutputSpec{...}, branchname2,
///                 ) reader;
///   auto processSomething = [] (auto& key, auto& object) {/*do something*/};
///   while (reader.next()) {
///     reader.process(processSomething);
///   }
///   // -- or --
///   while ((++reader)(processSomething));
///
/// In the DPL AlgorithmSpec the processing lambda can simple look like
///   auto processingFct = [reader](ProcessingContext& pc) {
///     // increment the reader and invoke it for the processing context
///     (++reader)(pc);
///   };
/// Note that reader has to be set up in the init callback and it must
/// be static there to persist. It can also be a shared_pointer, which then
/// requires additional dereferencing in the syntax.
///
template <typename KeyType = OutputSpec>
class RootTreeReader
{
 public:
  using self_type = RootTreeReader<KeyType>;
  using key_type = KeyType;

  // the key must not be of type const char* to make sure that the variable argument
  // list of the constructor can be parsed
  static_assert(std::is_same<KeyType, const char*>::value == false, "the key type must not be const char*");

  /// default constructor
  RootTreeReader();

  /// constructor
  /// @param treename  name of tree to process
  /// variable argument list of file names followed by pairs of KeyType and branch name
  template <typename... Args>
  RootTreeReader(const char* treename, // name of the tree to read from
                 Args&&... args)       // file names, followed by branch info
    : mInput(treename)
  {
    parseConstructorArgs(std::forward<Args>(args)...);
  }

  /// constructor
  /// @param treename  name of tree to process
  /// @nMaxEntries maximum number of entries to be processed
  /// variable argument list of file names followed by pairs of KeyType and branch name
  template <typename... Args>
  RootTreeReader(const char* treename, // name of the tree to read from
                 int nMaxEntries,      // max number of entries to be read
                 Args&&... args)       // file names, followed by branch info
    : mInput(treename),
      mMaxEntries(nMaxEntries)
  {
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
      ++mEntry;
      return false;
    }
    if (mMaxEntries > 0 && (mEntry + 1) >= mMaxEntries) {
      ++mEntry;
      return false;
    }
    mInput.GetEntry(++mEntry);
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
  template <typename ContextType>
  bool operator()(ContextType& context,
                  std::function<void(const KeyType&, const ROOTSerializedByClass&)> snapshot = nullptr)
  {
    if (!snapshot) {
      snapshot = [&context](const KeyType& key, const ROOTSerializedByClass& object) {
        context.allocator().snapshot(key, object);
      };
    }

    return process(snapshot);
  }

  bool process(std::function<void(const KeyType&, const ROOTSerializedByClass&)> snapshot = nullptr)
  {
    if (!snapshot) {
      return false;
    }

    if (mEntry >= mNEntries || mNEntries == 0 || (mMaxEntries > 0 && mEntry >= mMaxEntries)) {
      return false;
    }

    for (auto& spec : mBranchSpecs) {
      if (spec.second->data == nullptr) {
        // FIXME: is this an error?
        continue;
      }
      snapshot(spec.first, ROOTSerializedByClass(*spec.second->data, spec.second->classinfo));
    }
    return true;
  }

 private:
  struct BranchSpec {
    std::string name;
    char* data = nullptr;
    TBranch* branch = nullptr;
    TClass* classinfo = nullptr;
  };

  /// add a new branch definition
  /// we allow for multiple branch definition for the same key
  void addBranchSpec(KeyType key, const char* branchName)
  {
    // right now we allow the same key to appear for multiple branches
    mBranchSpecs.emplace_back(key, std::make_unique<BranchSpec>(BranchSpec{ branchName }));
    auto branch = mInput.GetBranch(mBranchSpecs.back().second->name.c_str());
    if (branch) {
      branch->SetAddress(&(mBranchSpecs.back().second->data));
      mBranchSpecs.back().second->branch = branch;
      mBranchSpecs.back().second->classinfo = TClass::GetClass(branch->GetClassName());
      LOG(INFO) << "branch set up: " << branchName;
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
  template <typename... Args>
  void parseConstructorArgs(KeyType key, const char* name, Args&&... args)
  {
    if (name != nullptr && *name != 0) {
      // add branch spec if the name is not empty
      addBranchSpec(key, name);
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

} // namespace framework
} // namespace o2
#endif
