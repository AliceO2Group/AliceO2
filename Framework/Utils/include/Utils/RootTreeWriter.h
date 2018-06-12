// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ROOTTREEWRITER_H
#define FRAMEWORK_ROOTTREEWRITER_H

/// @file   RootTreeWriter.h
/// @author Matthias Richter
/// @since  2018-05-15
/// @brief  A generic writer for ROOT TTrees

#include "Framework/ProcessingContext.h"
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TClass.h>
#include <vector>
#include <functional>
#include <string>
#include <stdexcept> // std::runtime_error
#include <type_traits>
#include <typeinfo>
#include <memory>     // std::make_unique
#include <functional> // std::function
#include <utility>    // std::forward

namespace o2
{
namespace framework
{

/// @class RootTreeWriter
/// A generic writer interface for ROOT TTrees
///
/// The writer class is configured with a list of types to be processed by defining
/// the class with the appropriate template argumets. Each type needs to be associated
/// with an input key and a branch name for the output.
///
/// The class is currently fixed to the DPL ProcessingContext and InputRecord for
/// reading the input.
///
/// Usage:
///   RootTreeWriter<Types...>("file_name",
///                            "tree_name",
///                            "key", "branchname",
///                            ...// further input and branch config
///                           ) writer;
///   writer(processingContext);
///
/// See also the MakeRootTreeWriterSpec helper class for easy generation of a
/// processor spec using RootTreeWriter.
template <typename... Types>
class RootTreeWriter
{
 public:
  using self_type = RootTreeWriter<Types...>;
  using key_type = std::string;
  using store_type = std::tuple<Types...>;
  static constexpr std::size_t store_size = sizeof...(Types);
  template <size_t Index>
  struct element {
    static_assert(Index < store_size, "Index out of range");
    using type = typename std::tuple_element<Index, store_type>::type;
  };

  /// default constructor forbidden
  RootTreeWriter() = delete;

  /// constructor
  /// @param treename  name of file
  /// @param treename  name of tree to write
  /// variable argument list of file names followed by pairs of key_type and branch name
  template <typename... Args>
  RootTreeWriter(const char* filename, // file name
                 const char* treename, // name of the tree to write to
                 Args&&... args)       // followed by branch info
  {
    mFile = std::make_unique<TFile>(filename, "RECREATE");
    mTree = std::make_unique<TTree>(treename, treename);
    mBranchSpecs.resize(store_size);
    parseConstructorArgs<0>(std::forward<Args>(args)...);
    setupBranches<store_size>();
  }

  /// process functor
  /// It expects a context which is used by lambda capture in the snapshot function.
  /// Loop over all branch definitions and publish by using the snapshot function.
  /// The default forwards to DPL DataAllocator snapshot.
  template <typename ContextType>
  void operator()(ContextType& context)
  {
    if (!mTree || !mFile || mFile->IsZombie()) {
      throw std::runtime_error("Writer is invalid state, pabably closed previously");
    }
    process<store_size>(context);
    mTree->Fill();
    for (auto& spec : mBranchSpecs) {
      spec.releaseFct(spec.data);
    }
  }

  /// write the tree and close the file
  /// the writer is invalid after calling close
  void close()
  {
    mTree->Write();
    mFile->Close();
    // this is a feature of ROOT, the tree belongs to the file and will be deleted
    // automatically
    mTree.release();
    mFile.reset(nullptr);
  }

 private:
  struct BranchSpec {
    key_type key;
    std::string name;
    void* data = nullptr;
    std::function<void(void*)> releaseFct = nullptr;
    TBranch* branch = nullptr;
    TClass* classinfo = nullptr;
  };

  /// recursively step through all members of the store and set up corresponding branch
  template <size_t N>
  typename std::enable_if<(N != 0)>::type setupBranches()
  {
    setupBranches<N - 1>();
    setupBranch<N - 1>();
  }

  // specializtion to end the recursive loop
  template <size_t N>
  typename std::enable_if<(N == 0)>::type setupBranches()
  {
  }

  /// set up a branch for the store type at Index position
  template <size_t Index>
  void setupBranch()
  {
    using ElementT = typename element<Index>::type;
    // the variable from the store is actually only used when setting up the branch, during
    // processing of input, the branch address is directly set to the extracted object
    ElementT& storeRef = std::get<Index>(mStore);
    mBranchSpecs[Index].branch = mTree->Branch(mBranchSpecs[Index].name.c_str(), &storeRef);
    mBranchSpecs[Index].classinfo = TClass::GetClass(typeid(ElementT));
    LOG(INFO) << "branch set up: " << mBranchSpecs[Index].name;
  }

  /// helper function to read branch config from vector, this excludes other variadic arguments
  template <size_t N>
  void parseConstructorArgs(const std::vector<std::pair<key_type, std::string>>& branchconfig)
  {
    static_assert(N == 0, "can not mix variadic arguments and vector config");
    if (branchconfig.size() != store_size) {
      throw std::runtime_error("number of branch specifications has to match number of types");
    }
    size_t index = 0;
    for (auto& config : branchconfig) {
      mBranchSpecs[index].key = config.first;
      mBranchSpecs[index].name = config.second;
      index++;
    }
  }

  /// helper function to recursively parse constructor arguments
  /// parse the branch definitions with key and branch name.
  template <size_t N, typename... Args>
  void parseConstructorArgs(key_type key, const char* name, Args&&... args)
  {
    static_assert(N < store_size, "too many branch arguments");
    // init branch spec
    using ElementT = typename element<N>::type;
    mBranchSpecs[N].key = key;
    mBranchSpecs[N].name = name;

    parseConstructorArgs<N + 1>(std::forward<Args>(args)...);
  }

  // this terminates the argument parsing, at this point we should have
  // parsed as many arguments as we have types
  template <size_t N>
  void parseConstructorArgs()
  {
    // at this point we should have processed argument pairs for all types in the stare
    static_assert(N == store_size, "too few branch arguments");
  }

  // want to have the ContextType as a template parameter, but the compiler
  // complains about calling the templated get function, to be investigated
  using ContextType = ProcessingContext;
  template <size_t N>
  typename std::enable_if<(N != 0)>::type process(ContextType& context)
  {
    constexpr static size_t Index = N - 1;
    process<Index>(context);
    using ElementT = typename element<Index>::type;
    auto data = context.inputs().get<ElementT*>(mBranchSpecs[Index].key.c_str());
    // could either copy to the corresponding store variable or use the object
    // directly. TBranch::SetAddress supports only non-const pointers, so this is
    // a hack
    // FIXME: get rid of the const_cast
    mBranchSpecs[Index].branch->SetAddress(const_cast<ElementT*>(data.get()));
    // the data object is a smart pointer and we have to keep the data alieve
    // until the tree is actually filled after the recursive processing of inputs
    // release the base pointer from the smart pointer instance and keep the
    // daleter to be called after tree fill.
    mBranchSpecs[Index].data = const_cast<ElementT*>(data.release());
    auto deleter = data.get_deleter();
    mBranchSpecs[Index].releaseFct = [deleter](void* buffer) { deleter(reinterpret_cast<ElementT*>(buffer)); };
  }

  template <size_t N, typename ContextType>
  typename std::enable_if<(N == 0)>::type process(ContextType& context)
  {
  }

  template <typename T>
  typename std::enable_if<std::is_move_assignable<T>::value == true>::type copyOrMove(T& from, T& to)
  {
    to = std::move(from);
  }

  template <typename T>
  typename std::enable_if<std::is_move_assignable<T>::value == false>::type copyOrMove(T& from, T& to)
  {
    to = from;
  }

  /// the output file
  std::unique_ptr<TFile> mFile;
  /// the output tree
  std::unique_ptr<TTree> mTree;
  /// definitions of branch specs
  std::vector<BranchSpec> mBranchSpecs;
  // store of underlying branch variables
  store_type mStore;
};

} // namespace framework
} // namespace o2
#endif
