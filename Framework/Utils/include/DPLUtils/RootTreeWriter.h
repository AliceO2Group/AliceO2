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

#include "Framework/RootSerializationSupport.h"
#include "Framework/InputRecord.h"
#include "Framework/DataRef.h"
#include "Framework/Logger.h"
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
#include <variant>

namespace o2
{
namespace framework
{

/// @class RootTreeWriter
/// @brief A generic writer interface for ROOT TTree objects.
///
/// The writer class is configured with the file name, the tree name and a variable list
/// of branch definitions passed to the constructor.
///
/// The class is currently fixed to the DPL ProcessingContext and InputRecord for
/// reading the input, but the implementation has been kept open for other interfaces.
///
/// Usage:
///
///     RootTreeWriter("file_name",
///                    "tree_name",
///                    BranchDef<type>{ "key", "branchname" },
///                    ...// further input and branch config
///                   ) writer;
///     writer(processingContext);
///
/// @note
/// See also the MakeRootTreeWriterSpec helper class for easy generation of a
/// processor spec using RootTreeWriter.
///
/// \par Using branch definition \c BranchDef:
/// The branch definition describes the mapping of inputs referenced by \a keys
/// to outputs, i.e. the branches. Each branch definition holds the data type as template
/// parameter, as well as input key definition and a branch name for the output.
/// A variable list of branch definition parameters can be given to the constructor.
/// See \ref BranchDef structure for more details.
///
/// \par Multiple inputs and outputs:
/// One branch definition can handle multiple branches as output for the same data type
/// and a single input or list of inputs. \ref BranchDef needs to be configured with the
/// number \a n of output branches, a callback to retrieve an index in the range [0, n-1],
/// and a callback creating the branch name for base name and index. A single input can
/// also be distributed to multiple branches if the callback calculates the index
/// from another piece of information, e.g. from information in the header stack.
///
/// \par Writing binary data:
/// While the generic writer is primarily intended for ROOT serializable objects, a special
/// case is the writing of binary data when <tt>const char*</tt> is used as type. Data is written
/// as a \c std::vector<char>, this ensures separation on event basis as well as having binary
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
  // a custom callback to close the writer
  // writing of tree and closing of file need to be implemented, but the pointers
  // remain owned by the tool
  using CustomClose = std::function<void(TFile* file, TTree* tree)>;

  /// DefaultKeyExtractor maps a data type used as key in the branch definition
  /// to the default internal key type std::string
  struct DefaultKeyExtractor {
    template <typename T>
    static key_type asString(T const& arg)
    {
      return arg;
    }
  };

  enum struct BranchVerbosity {
    Quiet,
    Medium,
    High,
  };

  /// @struct BranchDef Definition of branch specification for the RootTreeWriter
  /// @brief BranchDef is used to define the mapping between inputs and branches.

  /// A branch definition is always bound to a particular data type of the object to be
  /// written to tree branch. The type must be provided as template parameter.
  ///
  /// \par \a KeyType and \a KeyExtractor:
  /// Each branch definition is identified by a \c key which describes the input binding, i.e.
  /// it is used as argument in the input function. The RootTreeWriter uses \c std::string as
  /// internal key type to store a map of all dranch definitions. An extractor must be defined
  /// for the key type provided to BranchDef.
  /// In simple cases, defaults RootTreeWriter::key_type and RootTreeWriter::DefaultKeyExtractor
  /// can be used directly and are thus default template parameters.
  ///
  /// \par Multiple branches:
  /// The same definition can handle more than one branch as target for writing the objects,
  /// which is indicated by specifying the number of branches as parameter. The mapping of
  /// input objects to branch names is provided by the two callbacks \c getIndex and \c getName.
  /// The \c getIndex callback may extract the relavent information from the data object e.g.
  /// from the header stack and returns an index. The \c getName callback must return the
  /// branch name for writing based on this index.
  ///
  /// \par Multiple branches of identical data type:
  /// Multiple branches of identical data type can be served by one branch definition simply
  /// using a vector of inputs. Again, number of branches and \c getIndex and \c getName
  /// callbacks need to be provided
  ///
  /// \par Multiple inputs:
  /// The ability to serve more than one input can be used to write all data to the same branch,
  /// the exact behavior is controlled by the callbacks.
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

    using Fill = std::function<void(TBranch& branch, T const&)>;
    using FillExt = std::function<void(TBranch& branch, T const&, DataRef const&)>;
    using Spectator = std::function<void(T const&)>;
    using SpectatorExt = std::function<void(T const&, DataRef const&)>;
    using BranchCallback = std::variant<std::monostate, Fill, FillExt, Spectator, SpectatorExt>;
    BranchCallback callback;

    /// simple constructor for single input and one branch
    /// the definition is ignored if number of branches is zero
    /// @param key          input key
    /// @param _branchName  name of the target branch
    /// @param _nofBranches number of branches
    BranchDef(key_type key, std::string _branchName, size_t _nofBranches = 1) : keys({key}), branchName(_branchName), nofBranches(_nofBranches) {}

    /// constructor for single input and multiple output branches
    /// the definition is ignored if number of branches is zero
    /// @param key          input key
    /// @param _branchName  name of the target branch
    /// @param _nofBranches the number of target branches
    /// @param args         parameter pack can contain the following argument types:
    ///                     IndexExtractor: index callback
    ///                     BranchNameMapper: branch name callback
    ///                     Fill: fill handler
    ///                     Spectator: spectator handler
    template <typename... Args>
    BranchDef(key_type key, std::string _branchName, Args&&... args) : keys({key}), branchName(_branchName), nofBranches(1)
    {
      init(std::forward<Args>(args)...);
    }

    /// constructor for multiple inputs and multiple output branches
    /// the definition is ignored if number of branches is zero
    /// @param key          vector of input keys
    /// @param _branchName  name of the target branch
    /// @param _nofBranches the number of target branches
    /// @param args         parameter pack can contain the following argument types:
    ///                     IndexExtractor: index callback
    ///                     BranchNameMapper: branch name callback
    ///                     Fill: fill handler
    ///                     Spectator: spectator handler
    template <typename... Args>
    BranchDef(std::vector<key_type> vec, std::string _branchName, Args&&... args) : keys(vec), branchName(_branchName), nofBranches(1)
    {
      init(std::forward<Args>(args)...);
    }

    /// recursively init member variables from parameter pack
    template <typename Arg, typename... Args>
    void init(Arg&& arg, Args&&... args)
    {
      using Type = std::decay_t<Arg>;
      if constexpr (can_assign<Type, IndexExtractor>::value) {
        getIndex = arg;
      } else if constexpr (can_assign<Type, BranchNameMapper>::value) {
        getName = arg;
      } else if constexpr (can_assign<Type, Fill>::value) {
        callback = arg;
      } else if constexpr (can_assign<Type, FillExt>::value) {
        callback = arg;
      } else if constexpr (can_assign<Type, Spectator>::value) {
        callback = arg;
      } else if constexpr (can_assign<Type, SpectatorExt>::value) {
        callback = arg;
      } else if constexpr (std::is_integral<Type>::value) {
        nofBranches = arg;
      } else {
        assertNoMatchingType(std::forward<Arg>(arg));
      }
      if constexpr (sizeof...(args) > 0) {
        init(std::forward<Args>(args)...);
      }
    }

    // wrap the non matching type assert into a function to better see type of argument
    // in the compiler error
    template <typename Arg>
    void assertNoMatchingType(Arg&& arg)
    {
      static_assert(always_static_assert_v<Arg>, "no matching function signature for passed object. Please check:\n- Is it a callable object?\n- Does it have correct parameters and return type?\n- Are all type correctly qualified");
    }
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
    parseConstructorArgs(std::forward<Args>(args)...);
    if (!mTreeStructure) {
      std::runtime_error("Failed to create the branch configuration");
    }
    if (filename && treename) {
      init(filename, treename);
    }
  }

  /// Init the output file and tree.
  /// @param filename output file
  /// @param treename output tree
  ///
  /// After setting up the tree, the branches will be created according to the
  /// branch definition provided to the constructor.
  void init(const char* filename, const char* treename, const char* treetitle = nullptr)
  {
    mFile = std::make_unique<TFile>(filename, "RECREATE");
    mTree = std::make_unique<TTree>(treename, treetitle != nullptr ? treetitle : treename);
    mTreeStructure->setup(mBranchSpecs, mTree.get());
  }

  /// Set the branch name for a branch definition from the constructor argument list
  /// @param index       position in the argument list
  /// @param branchName  (base)branch name
  ///
  /// If the branch definition handles multiple output branches, the getName callback
  /// of the definition is used to build the names of the output branches
  void setBranchName(size_t index, const char* branchName)
  {
    auto& spec = mBranchSpecs.at(index);
    if (spec.names.size() > 1 && spec.getName) {
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
    if (mCustomClose) {
      mCustomClose(mFile.get(), mTree.get());
    } else {
      // set the number of elements according to branch content and write tree
      mTree->SetEntries();
      mTree->Write();
      mFile->Close();
    }
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

  /// A static helper function to change the type of a (yet unused) branch.
  /// Removes the old branch from the TTree system and creates a new branch.
  /// The function is useful for situations in which we want to transform data after
  /// the RootTreeWriter was created and change the type of the branch - such as in user callbacks.
  /// The function needs to be used with care. The user should ensure that "branch" is no longer used
  /// after a call to this function.
  template <typename T>
  static TBranch* remapBranch(TBranch& branchRef, T** newdata)
  {
    auto tree = branchRef.GetTree();
    auto name = branchRef.GetName();
    auto branch = tree->GetBranch(name); // the input branch might actually no belong to the tree but to TreeWriter cache
    assert(branch);
    if (branch->GetEntries()) { // if it has entries, then it was already remapped/filled at prevous event
      branch->SetAddress(newdata);
      return branch;
    }
    auto branchleaves = branch->GetListOfLeaves();
    branch->DropBaskets("all");
    branch->DeleteBaskets("all");
    tree->GetListOfBranches()->Remove(branch);
    for (auto entry : *branchleaves) {
      tree->GetListOfLeaves()->Remove(entry);
    }
    // ROOT leaves holes in the arrays so we need to compress as well
    tree->GetListOfBranches()->Compress();
    tree->GetListOfLeaves()->Compress();
    // create a new branch with the same original name but attached to the new data
    return tree->Branch(name, newdata);
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

  // type trait to determine the type of the storage
  // the storage type is different depending on the type, because
  // - binary branches are supported
  // - the ROOT TTree API makes a difference between fundamental types where the pointer to variable is
  //   passed and complex objects with dictionary where the pointer to a pointer variable needs to be
  //   passed. The former case will allways include a copy of the value, the latter uses the deserialized
  //   object by pointer
  // It appears, that also vectors of fundamental and also messageable tyapes can be treated in the same
  // way as fundamental types, as long as there is a corresponding TClass implementation. ROOT creates those
  // for vectors of fundamental types, for all other types, ClassDefNV macro is needed in the class
  // declaration together with entry in LinkDef.
  template <typename T, typename _ = void>
  struct StructureElementTypeTrait {
  };

  using BinaryBranchSpecialization = std::integral_constant<char, 0>;
  using MessageableTypeSpecialization = std::integral_constant<char, 1>;
  using MessageableVectorSpecialization = std::integral_constant<char, 2>;
  using ROOTTypeSpecialization = std::integral_constant<char, 3>;

  // the binary branch format is chosen for const char*
  // using internally a vector<char>, this involves for the moment a copy, investigate if ROOT
  // can write a simple array of chars and use a pointer
  template <typename T>
  struct StructureElementTypeTrait<T, std::enable_if_t<std::is_same<T, const char*>::value>> {
    using value_type = T;
    using store_type = BinaryBranchStoreType<char>;
    using specialization_id = BinaryBranchSpecialization;
  };

  // all messageable types
  // using an internal variable of the given type, involves a copy
  template <typename T>
  struct StructureElementTypeTrait<T, std::enable_if_t<is_messageable<T>::value>> {
    using value_type = T;
    using store_type = value_type;
    using specialization_id = MessageableTypeSpecialization;
  };

  // vectors of messageable types
  template <typename T>
  struct StructureElementTypeTrait<T, std::enable_if_t<has_messageable_value_type<T>::value &&
                                                       is_specialization<T, std::vector>::value>> {
    using value_type = T;
    using store_type = value_type*;
    using specialization_id = MessageableVectorSpecialization;
  };

  // types with root dictionary but non-messageable and not vectors of messageable type
  template <typename T>
  struct StructureElementTypeTrait<T, std::enable_if_t<has_root_dictionary<T>::value &&
                                                       is_messageable<T>::value == false &&
                                                       has_messageable_value_type<T>::value == false>> {
    using value_type = T;
    using store_type = value_type*;
    using specialization_id = ROOTTypeSpecialization;
  };

  // types marked as ROOT serialized
  template <typename T>
  struct StructureElementTypeTrait<T, std::enable_if_t<is_specialization<T, ROOTSerialized>::value == true>> {
    using value_type = typename T::wrapped_type;
    using store_type = value_type*;
    using specialization_id = ROOTTypeSpecialization;
  };

  /// one element in the tree structure object
  /// it contains the previous element as base class and is bound to a data type.
  template <typename DataT, typename BASE>
  class TreeStructureElement : public BASE
  {
   public:
    using PrevT = BASE;
    using value_type = typename StructureElementTypeTrait<DataT>::value_type;
    using store_type = typename StructureElementTypeTrait<DataT>::store_type;
    using specialization_id = typename StructureElementTypeTrait<DataT>::specialization_id;
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
    // specialized on specialization_id
    template <typename S, typename std::enable_if_t<!std::is_same<S, BinaryBranchSpecialization>::value, int> = 0>
    TBranch* createBranch(TTree* tree, const char* name, size_t branchIdx)
    {
      return tree->Branch(name, &(mStore.at(branchIdx)));
    }

    // specialization for binary buffers indicated by const char*
    // a variable binary data branch is written, the size of each entry is stored in a size branch
    // specialized on specialization_id
    template <typename S, typename std::enable_if_t<std::is_same<S, BinaryBranchSpecialization>::value, int> = 0>
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
      if (specs[SpecIndex].branches.size() == 0) {
        // this definition is disabled
        return;
      }
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
        specs[SpecIndex].branches.at(branchIdx) = createBranch<specialization_id>(tree, name.c_str(), branchIdx);
        if (specs[SpecIndex].branches.at(branchIdx) == nullptr) {
          throw std::runtime_error(std::to_string(SpecIndex) + ": can not create branch " + name + " for type " + typeid(value_type).name() + " - LinkDef entry missing?");
        }
        LOG(INFO) << SpecIndex << ": branch  " << name << " set up";
        branchIdx++;
      }
    }

    /// check the alternatives for the callback and run if there are any
    /// @return true if branch has been filled, false if still to be filled
    template <typename DataType>
    bool runCallback(TBranch* branch, DataType const& data, DataRef const& ref)
    {
      if (std::holds_alternative<typename BranchDef<value_type>::Spectator>(mCallback)) {
        std::get<typename BranchDef<value_type>::Spectator>(mCallback)(data);
      }
      if (std::holds_alternative<typename BranchDef<value_type>::SpectatorExt>(mCallback)) {
        std::get<typename BranchDef<value_type>::SpectatorExt>(mCallback)(data, ref);
      }
      if (std::holds_alternative<typename BranchDef<value_type>::Fill>(mCallback)) {
        std::get<typename BranchDef<value_type>::Fill>(mCallback)(*branch, data);
        return true;
      }
      if (std::holds_alternative<typename BranchDef<value_type>::FillExt>(mCallback)) {
        std::get<typename BranchDef<value_type>::FillExt>(mCallback)(*branch, data, ref);
        return true;
      }
      return false;
    }

    // specialization for trivial structs or serialized objects without a TClass interface
    // the extracted object is copied to store variable
    template <typename S, typename std::enable_if_t<std::is_same<S, MessageableTypeSpecialization>::value, int> = 0>
    void fillData(InputContext& context, DataRef const& ref, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<value_type>(ref);
      if (!runCallback(branch, data, ref)) {
        mStore[branchIdx] = data;
        branch->Fill();
      }
    }

    // specialization for non-messageable types with ROOT dictionary
    // for non-trivial structs, the address of the pointer to the objects needs to be used
    // in order to directly use the pointer to extracted object
    // store is a pointer to object
    template <typename S, typename std::enable_if_t<std::is_same<S, ROOTTypeSpecialization>::value, int> = 0>
    void fillData(InputContext& context, DataRef const& ref, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<typename std::add_pointer<value_type>::type>(ref);
      if (!runCallback(branch, *data, ref)) {
        // this is ugly but necessary because of the TTree API does not allow a const
        // object as input. Have to rely on that ROOT treats the object as const
        mStore[branchIdx] = const_cast<value_type*>(data.get());
        branch->Fill();
      }
    }

    // specialization for binary buffers using const char*
    // this writes both the data branch and a size branch
    template <typename S, typename std::enable_if_t<std::is_same<S, BinaryBranchSpecialization>::value, int> = 0>
    void fillData(InputContext& context, DataRef const& ref, TBranch* branch, size_t branchIdx)
    {
      auto data = context.get<gsl::span<char>>(ref);
      std::get<2>(mStore.at(branchIdx)) = data.size();
      std::get<1>(mStore.at(branchIdx))->Fill();
      std::get<0>(mStore.at(branchIdx)).resize(data.size());
      memcpy(std::get<0>(mStore.at(branchIdx)).data(), data.data(), data.size());
      branch->Fill();
    }

    // specialization for vectors of messageable types
    template <typename S, typename std::enable_if_t<std::is_same<S, MessageableVectorSpecialization>::value, int> = 0>
    void fillData(InputContext& context, DataRef const& ref, TBranch* branch, size_t branchIdx)
    {
      using ElementType = typename value_type::value_type;
      static_assert(is_messageable<ElementType>::value, "logical error: should be correctly selected by StructureElementTypeTrait");

      // A helper struct mimicking data layout of std::vector containers
      // We assume a standard layout of begin, end, end_capacity
      struct VecBase {
        VecBase() = default;
        const ElementType* start = nullptr;
        const ElementType* end = nullptr;
        const ElementType* cap = nullptr;
      };

      // a low level hack to make a gsl::span appear as a std::vector so that ROOT serializes the correct type
      // but without the need for an extra copy
      auto adopt = [](auto const& data, value_type& v) {
        static_assert(sizeof(v) == 24);
        if (data.size() == 0) {
          return;
        }
        VecBase impl;
        impl.start = &(data[0]);
        impl.end = &(data[data.size() - 1]) + 1; // end pointer (beyond last element)
        impl.cap = impl.end;
        std::memcpy(&v, &impl, sizeof(VecBase));
      };

      // if the value type is messagable and has a ROOT dictionary, two serialization methods are possible
      // for the moment, the InputRecord API can not handle both with the same call
      try {
        // try extracting from message with serialization method NONE, throw runtime error
        // if message is serialized
        auto data = context.get<gsl::span<ElementType>>(ref);
        // take an ordinary std::vector "view" on the data
        auto* dataview = new value_type;
        adopt(data, *dataview);
        if (!runCallback(branch, *dataview, ref)) {
          mStore[branchIdx] = dataview;
          branch->Fill();
        }
        // we delete JUST the view without deleting the data (which is handled by DPL)
        auto ptr = (VecBase*)dataview;
        if (ptr) {
          delete ptr;
        }
      } catch (RuntimeErrorRef e) {
        if constexpr (has_root_dictionary<value_type>::value == true) {
          // try extracting from message with serialization method ROOT
          auto data = context.get<typename std::add_pointer<value_type>::type>(ref);
          if (!runCallback(branch, *data, ref)) {
            mStore[branchIdx] = const_cast<value_type*>(data.get());
            branch->Fill();
          }
        } else {
          // the type has no ROOT dictionary, re-throw exception
          throw e;
        }
      }
    }

    // process previous stage and this stage
    void process(InputContext& context, std::vector<BranchSpec>& specs)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      PrevT::process(context, specs);
      constexpr size_t SpecIndex = STAGE - 1;
      BranchSpec const& spec = specs[SpecIndex];
      if (spec.branches.size() == 0) {
        // this definition is disabled
        return;
      }
      // loop over all defined inputs
      for (auto const& key : spec.keys) {
        auto keypos = context.getPos(key.c_str());
        auto parts = context.getNofParts(keypos);
        for (decltype(parts) part = 0; part < parts; part++) {
          auto dataref = context.get(key.c_str(), part);
          size_t branchIdx = 0;
          if (spec.getIndex) {
            branchIdx = spec.getIndex(dataref);
            if (branchIdx == ~(size_t)0) {
              // this indicates skipping
              continue;
            }
          }
          fillData<specialization_id>(context, dataref, spec.branches.at(branchIdx), branchIdx);
        }
      }
    }

    // helper function to get Nth argument from the argument pack
    template <size_t N, typename Arg, typename... Args>
    auto getArg(Arg&& arg, Args&&... args)
    {
      if constexpr (N == 0) {
        return arg;
      } else if constexpr (sizeof...(Args) > 0) {
        return getArg<N - 1>(std::forward<Args>(args)...);
      }
    }

    // recursively set optional callback from argument pack and recurse to all folded types
    template <typename... Args>
    void setCallback(Args&&... args)
    {
      // recursing through the tree structure by simply using method of the previous type,
      // i.e. the base class method.
      if constexpr (STAGE > 1) {
        PrevT::setCallback(std::forward<Args>(args)...);
      }

      // now set this instance, the argument position is determined by the STAGE counter
      auto arg = getArg<STAGE - 1>(std::forward<Args>(args)...);
      if constexpr (std::is_same<value_type, typename decltype(arg)::type>::value) {
        if (not std::holds_alternative<std::monostate>(arg.callback)) {
          mCallback = std::move(arg.callback);
        }
      }
    }

   private:
    /// internal store variable of the type wrapped by this instance
    std::vector<store_type> mStore;
    /// an optional callback to either customize the filling or just spectate on the data
    typename BranchDef<value_type>::BranchCallback mCallback;
  };

  /// recursive parsing of constructor arguments, all branch definitions come at the end of the pack
  template <typename Arg, typename... Args>
  void parseConstructorArgs(Arg&& arg, Args&&... args)
  {
    using Type = std::decay_t<Arg>;
    if constexpr (can_assign<Type, CustomClose>::value) {
      mCustomClose = arg;
    } else {
      mTreeStructure = createTreeStructure<0, TreeStructureInterface>(std::forward<Arg>(arg), std::forward<Args>(args)...);
      return;
    }
    if constexpr (sizeof...(Args) > 0) {
      parseConstructorArgs(std::forward<Args>(args)...);
    }
  }

  /// recursively step through all members of the store and set up corresponding branch
  template <size_t N, typename BASE, typename T, typename... Args>
  auto createTreeStructure(T&& def, Args&&... args)
  {
    // Note: a branch definition can be disabled by setting nofBranches to zero
    // an entry of the definition is kept in the registry, but the branches vector
    // is of size zero, which will make the writer to always skip this definition
    mBranchSpecs.push_back({{}, {def.branchName}});
    auto& spec = mBranchSpecs.back();

    // extract the internal keys for the list of provided input definitions
    size_t idx = 0;
    spec.keys.resize(def.keys.size());
    std::generate(spec.keys.begin(), spec.keys.end(), [&def, &idx] { return T::key_extractor::asString(def.keys[idx++]); });
    mBranchSpecs.back().branches.resize(def.nofBranches, nullptr);
    // a branch definition might be disabled by setting number of branches to 0; if enabled,
    // the number of branches has to match the number of inputs but can be larger depending
    // on the exact functionality provided with the getIndex callback. In any case, the
    // callbacks only need to be propagated if multiple branches are defined
    assert(def.nofBranches == 0 || def.nofBranches >= spec.keys.size());
    // a getIndex function makes only sense if there are multiple branches
    assert(def.nofBranches <= 1 || def.getIndex);
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
    if constexpr (N == 0) {
      // in the upper most call we have the concrete type of the mixin object
      // and call its method to set the callbacks from the argument pack, it
      // recurses to all folded types
      auto instance = createTreeStructure<N + 1, type>(std::forward<Args>(args)...);
      instance->setCallback(std::move(def), std::forward<Args>(args)...);
      // the upper most call returns the unique pointer of the virtual interface
      std::unique_ptr<TreeStructureInterface> ret(instance.release());
      return ret;
    } else {
      return std::move(createTreeStructure<N + 1, type>(std::forward<Args>(args)...));
    }
  }

  // last iteration, returning unique pointer of the concrete mixin
  template <size_t N, typename T>
  std::unique_ptr<T> createTreeStructure()
  {
    static_assert(N > 0, "The writer does not make sense without branch definitions");
    return std::make_unique<T>();
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
  /// custom close handler, optional
  CustomClose mCustomClose;
};

} // namespace framework
} // namespace o2
#endif
