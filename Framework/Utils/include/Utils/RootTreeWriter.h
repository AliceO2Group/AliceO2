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

#include "Framework/InputRecord.h"
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
/// The writer class is configured with a list of branch definitions passed to the
/// constructor. Each branch definition holds the data type as template parameter,
/// as well as an input key and a branch name for the output.
///
/// The class is currently fixed to the DPL ProcessingContext and InputRecord for
/// reading the input.
///
/// Usage:
///   RootTreeWriter("file_name",
///                  "tree_name",
///                  BranchDef<type>{ "key", "branchname" },
///                  ...// further input and branch config
///                 ) writer;
///   writer(processingContext);
///
/// See also the MakeRootTreeWriterSpec helper class for easy generation of a
/// processor spec using RootTreeWriter.
class RootTreeWriter
{
 public:
  // use string as input binding to be used with DPL input API
  using key_type = std::string;

  struct DefaultKeyExtractor {
    template <typename T>
    static key_type asString(T const& arg)
    {
      return arg;
    }
  };
  /// branch definition
  /// FIXME: could become BranchSpec, but that name is currently used for the elements
  /// of the internal store
  template <typename T, typename KeyType = key_type, typename KeyExtractor = DefaultKeyExtractor>
  struct BranchDef {
    using type = T;
    using key_type = KeyType;
    using key_extractor = KeyExtractor;
    KeyType key;
    std::string branchName;
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
                 Args&&... args)       // followed by branch definition
  {
    mTreeStructure = createTreeStructure<TreeStructureInterface>(std::forward<Args>(args)...);
    if (filename && treename) {
      init(filename, treename);
    }
  }

  void init(const char* filename, const char* treename)
  {
    mFile = std::make_unique<TFile>(filename, "RECREATE");
    mTree = std::make_unique<TTree>(treename, treename);
    mTreeStructure->setup(mBranchSpecs, mTree.get());
  }

  /// process functor
  /// It expects a context which is used by lambda capture in the snapshot function.
  /// Recursively process all inputs and set the branch address to extracted objects.
  /// Release function is called on the object after filling the tree.
  template <typename ContextType>
  void operator()(ContextType&& context)
  {
    if (!mTree || !mFile || mFile->IsZombie()) {
      throw std::runtime_error("Writer is invalid state, probably closed previously");
    }
    mTreeStructure->exec(std::forward<ContextType>(context), mBranchSpecs);
    mTree->Fill();
    for (auto& spec : mBranchSpecs) {
      if (!spec.releaseFct) {
        continue;
      }
      spec.releaseFct(spec.data);
      spec.data = nullptr;
      spec.releaseFct = nullptr;
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

  size_t getStoreSize() const
  {
    return (mTreeStructure != nullptr ? mTreeStructure->size() : 0);
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

  using InputContext = InputRecord;

  /// polymorphic interface for the mixin stack of branch type descriptions
  /// it implements the entry point for processing through exec method
  class TreeStructureInterface
  {
   public:
    static const size_t STAGE = 0;
    TreeStructureInterface() = default;
    virtual ~TreeStructureInterface() = default;

    virtual void setup(std::vector<BranchSpec>&, TTree*) {}
    virtual void exec(InputContext&, std::vector<BranchSpec>&) {}
    virtual size_t size() const { return STAGE; }

    // a dummy method called in the recursive processing
    void setupInstance(std::vector<BranchSpec>&, TTree*) {}
    // a dummy method called in the recursive processing
    void process(InputContext&, std::vector<BranchSpec>&) {}
  };

  template <typename DataT, typename BASE>
  class TreeStructureElement : public BASE
  {
   public:
    using PrevT = BASE;
    using type = DataT;
    static const size_t STAGE = BASE::STAGE + 1;
    TreeStructureElement() = default;
    ~TreeStructureElement() override = default;

    void setup(std::vector<BranchSpec>& specs, TTree* tree) override
    {
      setupInstance(specs, tree);
    }

    // this is the polymorphic entry point for processing of branch specs
    // recursive processing starting from the highest instance
    void exec(InputContext& context, std::vector<BranchSpec>& specs) override
    {
      process(context, specs);
    }
    size_t size() const override { return STAGE; }

    /// setup this instance and recurse to the parent one
    void setupInstance(std::vector<BranchSpec>& specs, TTree* tree)
    {
      static_cast<PrevT>(*this).setupInstance(specs, tree);
      constexpr size_t Index = STAGE - 1;
      specs[Index].branch = tree->Branch(specs[Index].name.c_str(), &mStore);
      specs[Index].classinfo = TClass::GetClass(typeid(type));
      LOG(INFO) << Index << ": branch  " << specs[Index].name << " set up";
    }

    // process previous stage and this stage
    void process(InputContext& context, std::vector<BranchSpec>& specs)
    {
      static_cast<PrevT>(*this).process(context, specs);
      constexpr size_t Index = STAGE - 1;
      auto data = context.get<type*>(specs[Index].key.c_str());
      // could either copy to the corresponding store variable or use the object
      // directly. TBranch::SetAddress supports only non-const pointers, so this is
      // a hack
      // FIXME: get rid of the const_cast
      // FIXME: using object directly results in a segfault in the Streamer when
      // using std::vector<o2::test::Polymorphic> in test_RootTreeWriter.cxx
      //specs[Index].branch->SetAddress(const_cast<type*>(data.get()));
      // handling copy-or-move is also more complicated because the returned smart
      // pointer wraps a const buffer which might reside in the input queue and
      // thus can not be moved.
      //copyOrMove(mStore, (type&)*data);
      mStore = *data;
      // the data object is a smart pointer and we have to keep the data alieve
      // until the tree is actually filled after the recursive processing of inputs
      // release the base pointer from the smart pointer instance and keep the
      // daleter to be called after tree fill.
      // Note: the deleter might not be existent if the content was moved before
      specs[Index].data = const_cast<type*>(data.release());
      auto deleter = data.get_deleter();
      specs[Index].releaseFct = [deleter](void* buffer) { deleter(reinterpret_cast<type*>(buffer)); };
    }

   private:
    /// internal store variable of the type wraped by this instance
    type mStore;
  };

  /// recursively step through all members of the store and set up corresponding branch
  template <typename BASE, typename T, typename... Args>
  std::enable_if_t<is_specialization<T, BranchDef>::value, std::unique_ptr<TreeStructureInterface>>
    createTreeStructure(T def, Args&&... args)
  {
    mBranchSpecs.push_back({ T::key_extractor::asString(def.key), def.branchName });
    using type = TreeStructureElement<typename T::type, BASE>;
    return std::move(createTreeStructure<type>(std::forward<Args>(args)...));
  }

  template <typename T>
  std::unique_ptr<TreeStructureInterface> createTreeStructure()
  {
    std::unique_ptr<TreeStructureInterface> ret(new T);
    return std::move(ret);
  }

  template <typename T>
  static typename std::enable_if<std::is_move_assignable<T>::value == true>::type copyOrMove(T& from, T& to)
  {
    to = std::move(from);
  }

  template <typename T>
  static typename std::enable_if<std::is_move_assignable<T>::value == false>::type copyOrMove(T& from, T& to)
  {
    to = from;
  }

  /// the output file
  std::unique_ptr<TFile> mFile;
  /// the output tree
  std::unique_ptr<TTree> mTree;
  /// definitions of branch specs
  std::vector<BranchSpec> mBranchSpecs;
  /// the underlying tree structure
  std::unique_ptr<TreeStructureInterface> mTreeStructure;
};

} // namespace framework
} // namespace o2
#endif
