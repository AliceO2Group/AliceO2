// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Utils RootTreeWriter
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "Headers/DataHeader.h"
#include <fairmq/FairMQMessage.h>
#include <fairmq/FairMQTransportFactory.h>
#include "Framework/DataProcessingHeader.h"
#include "Framework/InputRecord.h"
#include "Utils/RootTreeWriter.h"
#include "Utils/MakeRootTreeWriterSpec.h"
#include "../../Core/test/TestClasses.h"
#include <vector>
#include <memory>
#include <TClass.h>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

BOOST_AUTO_TEST_CASE(test_RootTreeWriter)
{
  // need to mimic a context to actually call the processing
  // for now just test the besic compilation and setup
  using Container = std::vector<o2::test::Polymorphic>;
  RootTreeWriter writer("test.root", "testtree",                                            // file and tree name
                        RootTreeWriter::BranchDef<int>{ "input1", "intbranch" },            // branch definition
                        RootTreeWriter::BranchDef<Container>{ "input2", "containerbranch" } // branch definition
                        );

  BOOST_CHECK(writer.getStoreSize() == 2);

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  std::vector<FairMQMessagePtr> messages;

  auto createPlainMessage = [&transport, &messages](DataHeader&& dh, auto& data) {
    dh.payloadSize = sizeof(data);
    dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    DataProcessingHeader dph{ 0, 1 };
    o2::header::Stack stack{ dh, dph };
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(sizeof(data));
    memcpy(header->GetData(), stack.data(), stack.size());
    memcpy(payload->GetData(), &data, sizeof(data));
    messages.emplace_back(std::move(header));
    messages.emplace_back(std::move(payload));
  };

  auto createSerializedMessage = [&transport, &messages](DataHeader&& dh, auto& data) {
    FairMQMessagePtr payload = transport->CreateMessage();
    auto* cl = TClass::GetClass(typeid(decltype(data)));
    TMessageSerializer().Serialize(*payload, &data, cl);
    dh.payloadSize = payload->GetSize();
    dh.payloadSerializationMethod = o2::header::gSerializationMethodROOT;
    DataProcessingHeader dph{ 0, 1 };
    o2::header::Stack stack{ dh, dph };
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    memcpy(header->GetData(), stack.data(), stack.size());
    messages.emplace_back(std::move(header));
    messages.emplace_back(std::move(payload));
  };

  int a = 23;
  Container b{ { 0 } };
  createPlainMessage(o2::header::DataHeader{ "INT", "TST", 0 }, a);
  createSerializedMessage(o2::header::DataHeader{ "CONTAINER", "TST", 0 }, b);

  // Note: InputRecord works on references to the schema and the message vector
  // so we can not specify the schema definition directly in the definition of
  // the InputRecord. Intrestingly enough, the compiler does not complain about
  // getting reference to temporary rvalue argument. So it might work if the
  // temporary argument is still in memory
  // FIXME: check why the compiler does not detect this
  std::vector<InputRoute> schema = {
    { InputSpec{ "input1", "TST", "INT" }, "input1", 0 },      //
    { InputSpec{ "input2", "TST", "CONTAINER" }, "input2", 0 } //
  };

  InputRecord inputs{
    schema,
    messages
  };

  writer(inputs);
  writer.close();
}

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

BOOST_AUTO_TEST_CASE(test_MakeRootTreeWriterSpec)
{
  // setup the spec helper and retrieve the spec by calling the operator
  MakeRootTreeWriterSpec("writer-process",                                                                   //
                         BranchDefinition<int>{ InputSpec{ "input1", "TST", "INTDATA" }, "intbranch" },      //
                         BranchDefinition<float>{ InputSpec{ "input2", "TST", "FLOATDATA" }, "floatbranch" } //
                         )();
}
