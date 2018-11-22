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
#include "Framework/DataRef.h"
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
#include <algorithm>  // std::generate

namespace o2
{
namespace framework
{

/// @class RootTreeWriter
/// A generic writer interface for ROOT TTrees
///
/// The writer class is configured with a list of branch definitions passed to the
/// constructor. Each branch definition holds the data type as template parameter,
/// as well as input definition and a branch name for the output.
///
/// The class is currently fixed to the DPL ProcessingContext and InputRecord for
/// reading the input, but the implementation has been kept open for other interfaces.
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
///
/// One branch definition can handle multiple branches as output for the same data type
/// and a single input or list of inputs. BranchDef needs to be configured with the
/// number n of output branches, a callback to retrieve an index in the range [0, n-1],
/// and a callback creating the branch name for base name and index. A single input can
/// also be distributed to multiple branches if the getIndex callback calculates the index
/// from another piece of information in the header stack.
class RootTreeWriter
{
 public:
  // use string as input binding to be used with DPL input API
  using key_type = std::string;
  // extract the branch index from DataRef, ~(size_t)) indicates "no data", then nothing
  // is extracted from input and writing is skipped
  using IndexExtractor = std::function<size_t(o2::framework::DataRef const&)>;
  // mapper between branch name and base/index
  using BranchNameMapper = std::function<std::string(std::string, size_t)>;

  /// DefaultKeyExtractor maps a data type used as key in the branch definition
  /// to the default internal key type std::string
  struct DefaultKeyExtractor {
    template <typename T>
    static key_type asString(T const& arg)
    {
      return arg;
    }
  };

  /// BranchDef is used to define the mapping between inputs and branches. It is always bound
  /// to data type of object to be written to tree branch.
  /// can be used with the default key type and extractor of the RootTreeWriter class, or
  /// by using a custom key type and extractor
  ///
  /// The same definition can handle more than one branch as target for writing the objects
  /// Two callbacks, getIndex and getName have to be provided together with the number of
  /// branches
  template <typename T, typename KeyType = key_type, typename KeyExtractor = DefaultKeyExtractor>
  struct BranchDef {
    using type = T;
    using key_type = KeyType;
    using key_extractor = KeyExtractor;
    std::vector<key_type> keys;
    std::string branchName;
    /// number of branches controlled by this definition for the same type
    size_t nofBranches = 1;
    /// extractor function for the index for parallel branches
    IndexExtractor getIndex; // = [](o2::framework::DataRef const&) {return 0;}
    /// get name of branch from base name and index
    BranchNameMapper getName = [](std::string base, size_t i) { return base + "_" + std::to_string(i); };

    /// simple constructor for single input and one branch
    BranchDef(key_type key, std::string _branchName) : keys({ key }), branchName(_branchName) {}

    /// constructor for single input and multiple output branches
    BranchDef(key_type key, std::string _branchName, size_t _nofBranches, IndexExtractor _getIndex, BranchNameMapper _getName) : keys({ key }), branchName(_branchName), nofBranches(_nofBranches), getIndex(_getIndex), getName(_getName) {}

    /// constructor for multiple inputs and multiple output branches
    BranchDef(std::vector<key_type> vec, std::string _branchName, size_t _nofBranches, IndexExtractor _getIndex, BranchNameMapper _getName) : keys(vec), branchName(_branchName), nofBranches(_nofBranches), getIndex(_getIndex), getName(_getName) {}
  };

  /// default constructor forbidden
  RootTreeWriter() = delete;

  /// constructor
  /// @param treename  name of file
  /// @param treename  name of tree to write
  /// variable argument list of branch definitions
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

  /// init the output file and tree
  /// After setting up the tree, the branches will be created according to the
  /// branch definition provided to the constructor.
  void init(const char* filename, const char* treename)
  {
    mFile = std::make_unique<TFile>(filename, "RECREATE");
    mTree = std::make_unique<TTree>(treename, treename);
    mTreeStructure->setup(mBranchSpecs, mTree.get());
  }

  /// set the branch name for a branch definition from the constructor argument list
  /// @param index       position in the argument list
  /// @param branchName  (base)branch name
  ///
  /// If the branch definition handles multiple output branches, the getName callback
  /// of the definition is used to build the names of the output branches
  void setBranchName(size_t index, const char* branchName)
  {
    auto& spec = mBranchSpecs.at(index);
    if (spec.getName) {
      // set the branch names for this group
      size_t idx = 0;
      std::generate(spec.names.begin(), spec.names.end(), [&]() { return spec.getName(branchName, idx++); });
    } else {
      // single branch for this definition
      spec.names.at(0) = branchName;
    }
  }

  /// process functor
  /// It expects a context which is used by lambda capture in the snapshot function.
  /// Recursively process all inputs and fill branches individually from extracted
  /// objects.
  template <typename ContextType>
  void operator()(ContextType&& context)
  {
    if (!mTree || !mFile || mFile->IsZombie()) {
      throw std::runtime_error("Writer is invalid state, probably closed previously");
    }
    // execute tree structure handlers and fill the individual branches
    mTreeStructure->exec(std::forward<ContextType>(context), mBranchSpecs);
    // Note: number of entries will be set when closing the writer
  }

  /// write the tree and close the file
  /// the writer is invalid after calling close
  void close()
  {
    // set the number of elements according to branch content and write tree
    mTree->SetEntries();
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
  /// internal input and branch properties
  struct BranchSpec {
    std::vector<key_type> keys;
    std::vector<std::string> names;
    std::vector<TBranch*> branches;
    TClass* classinfo = nullptr;
    IndexExtractor getIndex;
    BranchNameMapper getName;
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

    /// create branches according to the branch definition
    /// enters at the outermost element and recurses to the base elements
    virtual void setup(std::vector<BranchSpec>&, TTree*) {}
    /// exec the branch structure
    /// enters at the outermost element and recurses to the base elements
    /// Read the configured inputs from the input context, select the output branch
    /// and write the object
    virtual void exec(InputContext&, std::vector<BranchSpec>&) {}
    /// get the size of the branch structure, i.e. the number of registered branch
    /// definitions
    virtual size_t size() const { return STAGE; }

    // a dummy method called in the recursive processing
    void setupInstance(std::vector<BranchSpec>&, TTree*) {}
    // a dummy method called in the recursive processing
    void process(InputContext&, std::vector<BranchSpec>&) {}
  };

  /// one element in the tree structure object
  /// it contains the previous element as base class and is bound to a data type.
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
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      PrevT::setupInstance(specs, tree);
      constexpr size_t SpecIndex = STAGE - 1;
      size_t branchIdx = 0;
      mStore.resize(specs[SpecIndex].names.size());
      for (auto const& name : specs[SpecIndex].names) {
        specs[SpecIndex].branches.at(branchIdx) = tree->Branch(name.c_str(), &(mStore.at(branchIdx)));
        LOG(INFO) << SpecIndex << ": branch  " << name << " set up";
        branchIdx++;
      }
      specs[SpecIndex].classinfo = TClass::GetClass(typeid(type));
    }

    // process previous stage and this stage
    void process(InputContext& context, std::vector<BranchSpec>& specs)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      PrevT::process(context, specs);
      constexpr size_t SpecIndex = STAGE - 1;
      BranchSpec const& spec = specs[SpecIndex];
      // loop over all defined inputs
      for (auto const& key : spec.keys) {
        auto dataref = context.get(key.c_str());
        size_t branchIdx = 0;
        if (spec.getIndex) {
          branchIdx = spec.getIndex(dataref);
          if (branchIdx == ~(size_t)0) {
            // this indicates skipping
            continue;
          }
        }
        auto& store = mStore[branchIdx];
        auto data = context.get<type*>(key.c_str());
        // could either copy to the corresponding store variable or use the object
        // directly. TBranch::SetAddress supports only non-const pointers, so this is
        // a hack
        // FIXME: get rid of the const_cast
        // FIXME: using object directly results in a segfault in the Streamer when
        // using std::vector<o2::test::Polymorphic> in test_RootTreeWriter.cxx
        // for std::vector, the branch has sub-branches so maybe the address can not
        // simply be set
        //spec.branches.at(branchIdx)->SetAddress(const_cast<type*>(data.get()));
        // handling copy-or-move is also more complicated because the returned smart
        // pointer wraps a const buffer which might reside in the input queue and
        // thus can not be moved.
        //copyOrMove(store, (type&)*data);
        store = *data;
        spec.branches.at(branchIdx)->Fill();
      }
    }

   private:
    /// internal store variable of the type wraped by this instance
    std::vector<type> mStore;
  };

  /// recursively step through all members of the store and set up corresponding branch
  template <typename BASE, typename T, typename... Args>
  std::enable_if_t<std::is_base_of<BranchDef<typename T::type, typename T::key_type, typename T::key_extractor>, T>::value, std::unique_ptr<TreeStructureInterface>>
    createTreeStructure(T def, Args&&... args)
  {
    mBranchSpecs.push_back({ {}, { def.branchName } });
    auto& spec = mBranchSpecs.back();

    // extract the internal keys for the list of provided input definitions
    size_t idx = 0;
    spec.keys.resize(def.keys.size());
    std::generate(spec.keys.begin(), spec.keys.end(), [&def, &idx] { return T::key_extractor::asString(def.keys[idx++]); });
    mBranchSpecs.back().branches.resize(def.nofBranches, nullptr);
    // the number of branches has to match the number of inputs but can be larger depending
    // on the exact functionality provided with the getIndex callback. In any case, the
    // callbacks only need to be propagated if multiple branches are defined
    assert(def.nofBranches >= spec.keys.size());
    // a getIndex function makes only sense if there are multiple branches
    assert(def.nofBranches == 1 || def.getIndex);
    if (def.nofBranches > 1) {
      assert(def.getIndex && def.getName);
      mBranchSpecs.back().getIndex = def.getIndex;
      mBranchSpecs.back().getName = def.getName;
      mBranchSpecs.back().names.resize(def.nofBranches);

      // fill the branch names by calling the getName callback
      idx = 0;
      std::generate(mBranchSpecs.back().names.begin(), mBranchSpecs.back().names.end(),
                    [&def, &idx]() { return def.getName(def.branchName, idx++); });
    }
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
