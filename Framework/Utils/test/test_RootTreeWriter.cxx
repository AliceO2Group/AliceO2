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

#include <catch_amalgamated.hpp>
#include "Framework/RootSerializationSupport.h"
#include <iomanip>
#include "Headers/DataHeader.h"
#include <fairmq/Message.h>
#include <fairmq/TransportFactory.h>
#include "Framework/DataProcessingHeader.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "DPLUtils/RootTreeWriter.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "../../Core/test/TestClasses.h"
#include <vector>
#include <memory>
#include <iostream>
#include <type_traits> // std::is_fundamental
#include <TClass.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TSystem.h>

#define CHECK_MESSAGE(cond, msg) \
  do {                           \
    INFO(msg);                   \
    CHECK(cond);                 \
  } while ((void)0, 0)
#define REQUIRE_MESSAGE(cond, msg) \
  do {                             \
    INFO(msg);                     \
    REQUIRE(cond);                 \
  } while ((void)0, 0)

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

namespace o2::test
{

/// test struct without ROOT dictionary, used to provoke raising of runtime_error
struct TrivialStruct {
  int x;
  int y;
};

template <typename T>
struct BranchContent {
  using ref_type = T;
  const char* branchName;
  ref_type reference;
};

template <typename T>
bool checkBranch(TTree& tree, BranchContent<T>&& content)
{
  TBranch* branch = tree.GetBranch(content.branchName);
  REQUIRE(branch != nullptr);
  T store;
  T* pointer = &store;
  // in general, pointer to pointer has to be used for setting the branch
  // address to a store object, however this does not work for fundamental
  // types, there the address to the variable has to be used in order
  // to read back the value. Why? no clue.
  if (std::is_fundamental<T>::value) {
    branch->SetAddress(&store);
  } else {
    branch->SetAddress(&pointer);
  }
  branch->GetEntry(0);
  CHECK_MESSAGE(store == content.reference, "mismatch for branch " << content.branchName);
  return store == content.reference;
}

template <typename T, typename... Args>
bool checkBranch(TTree& tree, BranchContent<T>&& content, Args&&... args)
{
  return checkBranch(tree, std::forward<BranchContent<T>>(content)) && checkBranch(tree, std::forward<Args>(args)...);
}

template <typename... Args>
bool checkTree(const char* filename, const char* treename, Args&&... args)
{
  TFile* file = TFile::Open(filename);
  REQUIRE(file != nullptr);
  auto* tree = reinterpret_cast<TTree*>(file->GetObjectChecked(treename, "TTree"));
  REQUIRE(tree != nullptr);
  return checkBranch(*tree, std::forward<Args>(args)...);
}

TEST_CASE("test_RootTreeWriter")
{
  std::string filename = "test_RootTreeWriter.root";
  const char* treename = "testtree";

  using Container = std::vector<o2::test::Polymorphic>;
  // setting up the writer with two branch definitions
  // first definition is for a single input and simple type written to one branch
  // second branch handles two inputs of the same data type, the mapping of the
  // input data to the target branch is taken from the sub specification
  auto getIndex = [](o2::framework::DataRef const& ref) -> size_t {
    auto const* dataHeader = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    return dataHeader->subSpecification;
  };
  auto getName = [](std::string base, size_t i) -> std::string {
    return base + "_" + std::to_string(i);
  };
  auto customClose = [](TFile* file, TTree* tree) {
    // branches are filled independently of the tree, so the tree state needs to be
    // synchronized with the branch states
    tree->SetEntries();
    INFO("Custom close, tree has " << tree->GetEntries() << " entries");
    // there was one write cycle and each branch should have one entry
    CHECK(tree->GetEntries() == 1);
    tree->Write();
    file->Close();
  };
  RootTreeWriter writer(filename.c_str(), treename, // file and tree name
                        customClose,
                        RootTreeWriter::BranchDef<int>{"input1", "intbranch"},
                        RootTreeWriter::BranchDef<Container>{
                          std::vector<std::string>({"input2", "input3"}), "containerbranch",
                          // define two target branches (this matches the input list)
                          2,
                          // the callback extracts the sub specification from the DataHeader as index
                          getIndex,
                          // the branch names are simply built by adding the index
                          getName},
                        RootTreeWriter::BranchDef<const char*>{"input4", "binarybranch"},
                        RootTreeWriter::BranchDef<o2::test::TriviallyCopyable>{"input6", "msgablebranch"},
                        RootTreeWriter::BranchDef<std::vector<int>>{"input6", "intvecbranch"},
                        RootTreeWriter::BranchDef<std::vector<o2::test::TriviallyCopyable>>{"input7", "trivvecbranch"},
                        // TriviallyCopyable can be sent with either serialization methods NONE or ROOT
                        RootTreeWriter::BranchDef<std::vector<o2::test::TriviallyCopyable>>{"input8", "srlzdvecbranch"});

  CHECK(writer.getStoreSize() == 7);

  // need to mimic a context to actually call the processing
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::vector<fair::mq::MessagePtr> store;

  auto createPlainMessage = [&transport, &store](DataHeader&& dh, auto& data) {
    dh.payloadSize = sizeof(data);
    dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    DataProcessingHeader dph{0, 1};
    o2::header::Stack stack{dh, dph};
    fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
    fair::mq::MessagePtr payload = transport->CreateMessage(sizeof(data));
    memcpy(header->GetData(), stack.data(), stack.size());
    memcpy(payload->GetData(), &data, sizeof(data));
    store.emplace_back(std::move(header));
    store.emplace_back(std::move(payload));
  };

  auto createVectorMessage = [&transport, &store](DataHeader&& dh, auto& data) {
    dh.payloadSize = data.size() * sizeof(typename std::remove_reference<decltype(data)>::type::value_type);
    dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    DataProcessingHeader dph{0, 1};
    o2::header::Stack stack{dh, dph};
    fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
    fair::mq::MessagePtr payload = transport->CreateMessage(dh.payloadSize);
    memcpy(header->GetData(), stack.data(), stack.size());
    memcpy(payload->GetData(), data.data(), dh.payloadSize);
    store.emplace_back(std::move(header));
    store.emplace_back(std::move(payload));
  };

  auto createSerializedMessage = [&transport, &store](DataHeader&& dh, auto& data) {
    fair::mq::MessagePtr payload = transport->CreateMessage();
    auto* cl = TClass::GetClass(typeid(decltype(data)));
    TMessageSerializer().Serialize(*payload, &data, cl);
    dh.payloadSize = payload->GetSize();
    dh.payloadSerializationMethod = o2::header::gSerializationMethodROOT;
    DataProcessingHeader dph{0, 1};
    o2::header::Stack stack{dh, dph};
    fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
    memcpy(header->GetData(), stack.data(), stack.size());
    store.emplace_back(std::move(header));
    store.emplace_back(std::move(payload));
  };

  int a = 23;
  Container b{{0}};
  Container c{{21}};
  o2::test::TriviallyCopyable msgable{10, 21, 42};
  std::vector<int> intvec{10, 21, 42};
  std::vector<o2::test::TriviallyCopyable> trivvec{{10, 21, 42}, {1, 2, 3}};
  createPlainMessage(o2::header::DataHeader{"INT", "TST", 0}, a);
  createSerializedMessage(o2::header::DataHeader{"CONTAINER", "TST", 0}, b);
  createSerializedMessage(o2::header::DataHeader{"CONTAINER", "TST", 1}, c);
  createPlainMessage(o2::header::DataHeader{"BINARY", "TST", 0}, a);
  createPlainMessage(o2::header::DataHeader{"MSGABLE", "TST", 0}, msgable);
  createVectorMessage(o2::header::DataHeader{"FDMTLVEC", "TST", 0}, intvec);
  createVectorMessage(o2::header::DataHeader{"TRIV_VEC", "TST", 0}, trivvec);
  createSerializedMessage(o2::header::DataHeader{"SRLZDVEC", "TST", 0}, trivvec);

  // Note: InputRecord works on references to the schema and the message vector
  // so we can not specify the schema definition directly in the definition of
  // the InputRecord. Intrestingly enough, the compiler does not complain about
  // getting reference to temporary rvalue argument. So it might work if the
  // temporary argument is still in memory
  // FIXME: check why the compiler does not detect this
  std::vector<InputRoute> schema = {
    {InputSpec{"input1", "TST", "INT"}, 0, "input1", 0},       //
    {InputSpec{"input2", "TST", "CONTAINER"}, 1, "input2", 0}, //
    {InputSpec{"input3", "TST", "CONTAINER"}, 2, "input3", 0}, //
    {InputSpec{"input4", "TST", "BINARY"}, 3, "input4", 0},    //
    {InputSpec{"input5", "TST", "MSGABLE"}, 4, "input5", 0},   //
    {InputSpec{"input6", "TST", "FDMTLVEC"}, 5, "input6", 0},  //
    {InputSpec{"input7", "TST", "TRIV_VEC"}, 6, "input7", 0},  //
    {InputSpec{"input8", "TST", "SRLZDVEC"}, 7, "input8", 0},  //
  };

  auto getter = [&store](size_t i) -> DataRef {
    return DataRef{nullptr, static_cast<char const*>(store[2 * i]->GetData()), static_cast<char const*>(store[2 * i + 1]->GetData())};
  };
  InputSpan span{getter, store.size() / 2};
  ServiceRegistry registry;
  InputRecord inputs{
    schema,
    span,
    registry};

  writer(inputs);
  writer.close();

  checkTree(filename.c_str(), treename,
            BranchContent<decltype(a)>{"intbranch", a},
            BranchContent<decltype(b)>{"containerbranch_0", b},
            BranchContent<decltype(c)>{"containerbranch_1", c},
            BranchContent<decltype(msgable)>{"msgablebranch", msgable},
            BranchContent<decltype(intvec)>{"intvecbranch", intvec},
            BranchContent<decltype(trivvec)>{"trivvecbranch", trivvec},
            BranchContent<decltype(trivvec)>{"srlzdvecbranch", trivvec});
}

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

TEST_CASE("test_MakeRootTreeWriterSpec")
{
  // setup the spec helper and retrieve the spec by calling the operator
  struct Printer {
    Printer()
    {
      // TODO: to be fully correct we need to check at exit if we have been here instead
      // of a thumb log message
      std::cout << "Setting up a spectator" << std::endl;
    }
  };
  auto logger = [printer = std::make_shared<Printer>()](float const&) {
  };
  MakeRootTreeWriterSpec("writer-process",                                                          //
                         BranchDefinition<int>{InputSpec{"input1", "TST", "INTDATA"}, "intbranch"}, //
                         BranchDefinition<float>{InputSpec{"input2", "TST", "FLOATDATA"},           //
                                                 "floatbranch", "floatbranchname",                  //
                                                 1, logger}                                         //
                         )();
}

TEST_CASE("test_ThrowOnMissingDictionary")
{
  // trying to set up a branch for a collection class without dictionary which must throw
  const char* filename = "test_RootTreeWriterTrow.root";
  const char* treename = "testtree";
  RootTreeWriter writer(nullptr, nullptr, RootTreeWriter::BranchDef<std::vector<TrivialStruct>>{"input1", "vecbranch"});
  REQUIRE_THROWS(writer.init(filename, treename));
  // we print this note to explain the error message in the log
  INFO("Note: This error has been provoked by the configuration, the exception has been handled");
}

template <typename T>
using Trait = RootTreeWriter::StructureElementTypeTrait<T>;
template <typename T>
using BinaryBranchStoreType = RootTreeWriter::BinaryBranchStoreType<T>;
TEST_CASE("test_RootTreeWriterSpec_store_types")
{
  using TriviallyCopyable = o2::test::TriviallyCopyable;
  using Polymorphic = o2::test::Polymorphic;

  // simple fundamental type
  // type itself used as store type
  static_assert(std::is_same<Trait<int>::store_type, int>::value == true);

  // messageable type with or without ROOT dictionary
  // type itself used as store type
  static_assert(std::is_same<Trait<TriviallyCopyable>::store_type, TriviallyCopyable>::value == true);

  // non-messageable type with ROOT dictionary
  // pointer type used as store type
  static_assert(std::is_same<Trait<Polymorphic>::store_type, Polymorphic*>::value == true);

  // binary branch indicated through const char*
  // BinaryBranchStoreType is used
  static_assert(std::is_same<Trait<const char*>::store_type, BinaryBranchStoreType<char>>::value == true);

  // vectors of fundamental types
  // type itself (the vector) is used
  static_assert(std::is_same<Trait<std::vector<int>>::store_type, std::vector<int>*>::value == true);

  // vector of messageable type with or without ROOT dictionary
  // type itself (the vector) is used
  static_assert(std::is_same<Trait<std::vector<TriviallyCopyable>>::store_type, std::vector<TriviallyCopyable>*>::value == true);

  // vector of non-messageable type with ROOT dictionary
  // pointer type used as store type
  static_assert(std::is_same<Trait<std::vector<Polymorphic>>::store_type, std::vector<Polymorphic>*>::value == true);
}

TEST_CASE("TestCanAssign")
{
  using Callback = std::function<bool(int, float)>;
  auto matching = [](int, float) -> bool {
    return true;
  };
  auto otherReturn = [](int, float) -> int {
    return 0;
  };
  auto otherParam = [](int, int) -> bool {
    return true;
  };
  REQUIRE((can_assign<decltype(matching), Callback>::value == true));
  REQUIRE((can_assign<decltype(otherReturn), Callback>::value == false));
  REQUIRE((can_assign<decltype(otherParam), Callback>::value == false));
}
} // namespace o2::test
