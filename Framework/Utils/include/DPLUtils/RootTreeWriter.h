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
///
/// While the generic writer is primarily intended for ROOT serializable objects, a special
/// case is the writing of binary data when const char* is used as type. Data is written
/// as a std::vector<char>, this ensures separation on event basis as well as having binary
/// data in parallel to ROOT objects in the same file, e.g. a binary data format from the
/// reconstruction in parallel to MC labels.
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
    mIsClosed = true;
    if (!mFile) {
      return;
    }
    // set the number of elements according to branch content and write tree
    mTree->SetEntries();
    mTree->Write();
    mFile->Close();
    // this is a feature of ROOT, the tree belongs to the file and will be deleted
    // automatically
    mTree.release();
    mFile.reset(nullptr);
  }

  bool isClosed() const
  {
    return mIsClosed;
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

  template <typename T = char>
  using BinaryBranchStoreType = std::tuple<std::vector<T>, TBranch*, size_t>;

  /// one element in the tree structure object
  /// it contains the previous element as base class and is bound to a data type.
  template <typename DataT, typename BASE>
  class TreeStructureElement : public BASE
  {
   public:
    using PrevT = BASE;
    using value_type = DataT;
    using store_type = typename std::conditional<std::is_same<DataT, const char*>::value,
                                                 BinaryBranchStoreType<char>,
                                                 typename std::conditional<has_root_dictionary<value_type>::value,
                                                                           value_type*,
                                                                           value_type>::type>::type;
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

    // the default method creates branch using address to store variable
    // Note: the type of the store variable is pointer to object for objects with a ROOT TClass
    // interface, or plain type for all others
    template <typename T, typename std::enable_if_t<!std::is_same<T, const char*>::value, int> = 0>
    TBranch* createBranch(TTree* tree, const char* name, size_t branchIdx)
    {
      return tree->Branch(name, &(mStore.at(branchIdx)));
    }

    // specialization for binary buffers indicated by const char*
    // a variable binary data branch is written, the size of each entry is stored in a size branch
    template <typename T, typename std::enable_if_t<std::is_same<T, const char*>::value, int> = 0>
    TBranch* createBranch(TTree* tree, const char* name, size_t branchIdx)
    {
      // size variable to write the size of the data branch
      std::get<2>(mStore.at(branchIdx)) = 1;
      // branch name of the size branch
      std::string sizeBranchName = std::string(name) + "Size";
      // leaf list: one leaf of type unsinged int
      std::string leafList = sizeBranchName + "/i";
      std::get<1>(mStore.at(branchIdx)) = tree->Branch(sizeBranchName.c_str(), &(std::get<2>(mStore.at(branchIdx))), leafList.c_str());
      return tree->Branch(name, &(std::get<0>(mStore.at(branchIdx))));
    }

    /// setup this instance and recurse to the parent one
    void setupInstance(std::vector<BranchSpec>& specs, TTree* tree)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      PrevT::setupInstance(specs, tree);
      constexpr size_t SpecIndex = STAGE - 1;
      specs[SpecIndex].classinfo = TClass::GetClass(typeid(value_type));
      if (std::is_same<value_type, const char*>::value == false && std::is_fundamental<value_type>::value == false &&
          specs[SpecIndex].classinfo == nullptr) {
        // for all non-fundamental types but the special case for binary chunks, a dictionary is required
        // FIXME: find a reliable way to check that the type has been specified in the LinkDef
        // Only then the required functionality for streaming the type to the branch is available.
        // If e.g. a standard container of some ROOT serializable type has not been specified in
        // LinkDef, the functionality is not available and the attempt to stream will simply crash.
        // Unfortunately, a class info object can be extracted for the type, so this check does not help
        throw std::runtime_error(std::to_string(SpecIndex) + ": no dictionary available for non-fundamental type " + typeid(value_type).name());
      }
      size_t branchIdx = 0;
      mStore.resize(specs[SpecIndex].names.size());
      for (auto const& name : specs[SpecIndex].names) {
        specs[SpecIndex].branches.at(branchIdx) = createBranch<value_type>(tree, name.c_str(), branchIdx);
        LOG(INFO) << SpecIndex << ": branch  " << name << " set up";
        branchIdx++;
      }
    }

    // specialization for trivial structs or serialized objects without a TClass interface
    // the extracted object is copied to store variable
    template <typename T, typename std::enable_if_t<std::is_same<T, value_type>::value, int> = 0>
    void fillData(InputContext& context, const char* key, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<typename std::add_pointer<value_type>::type>(key);
      mStore[branchIdx] = *data;
      branch->Fill();
    }

    // specialization for objects with ROOT dictionary
    // for non-trivial structs, the address of the pointer to the objects needs to be used
    // in order to directly use the pointer to extracted object
    // store is a pointer to object
    template <typename T, typename std::enable_if_t<std::is_same<T, value_type*>::value, int> = 0>
    void fillData(InputContext& context, const char* key, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<typename std::add_pointer<value_type>::type>(key);
      // this is ugly but necessary because of the TTree API does not allow a const
      // object as input. Have to rely on that ROOT treats the object as const
      mStore[branchIdx] = const_cast<value_type*>(data.get());
      branch->Fill();
    }

    // specialization for binary buffers using const char*
    // this writes both the data branch and a size branch
    template <typename T, typename std::enable_if_t<std::is_same<T, BinaryBranchStoreType<char>>::value, int> = 0>
    void fillData(InputContext& context, const char* key, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<gsl::span<char>>(key);
      std::get<2>(mStore.at(branchIdx)) = data.size();
      std::get<1>(mStore.at(branchIdx))->Fill();
      std::get<0>(mStore.at(branchIdx)).resize(data.size());
      memcpy(std::get<0>(mStore.at(branchIdx)).data(), data.data(), data.size());
      branch->Fill();
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
        fillData<store_type>(context, key.c_str(), spec.branches.at(branchIdx), branchIdx);
      }
    }

   private:
    /// internal store variable of the type wrapped by this instance
    std::vector<store_type> mStore;
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

  /// the output file
  std::unique_ptr<TFile> mFile;
  /// the output tree
  std::unique_ptr<TTree> mTree;
  /// definitions of branch specs
  std::vector<BranchSpec> mBranchSpecs;
  /// the underlying tree structure
  std::unique_ptr<TreeStructureInterface> mTreeStructure;
  /// indicate that the writer has been closed
  bool mIsClosed = false;
};

} // namespace framework
} // namespace o2
#endif
