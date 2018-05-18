// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/SerializationMethods.h"
#include "Headers/DataHeader.h"
#include "TestClasses.h"
#include "FairMQLogger.h"
#include <vector>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
  }

DataProcessorSpec getTimeoutSpec()
{
  // a timer process to terminate the workflow after a timeout
  auto processingFct = [](ProcessingContext& pc) {
    static int counter = 0;
    pc.outputs().snapshot(Output{ "TST", "TIMER", 0, Lifetime::Timeframe }, counter);

    sleep(1);
    if (counter++ > 10) {
      LOG(ERROR) << "Timeout reached, the workflow seems to be broken";
      pc.services().get<ControlService>().readyToQuit(true);
    }
  };

  return DataProcessorSpec{ "timer",  // name of the processor
                            Inputs{}, // inputs empty
                            { OutputSpec{ "TST", "TIMER", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(processingFct) };
}

DataProcessorSpec getSourceSpec()
{
  auto processingFct = [](ProcessingContext& pc) {
    static int counter = 0;
    o2::test::TriviallyCopyable a(42, 23, 0xdead);
    o2::test::Polymorphic b(0xbeef);
    std::vector<o2::test::Polymorphic> c{ { 0xaffe }, { 0xd00f } };
    // class TriviallyCopyable is both messageable and has a dictionary, the default
    // picked by the framework is no serialization
    pc.outputs().snapshot(Output{ "TST", "MESSAGEABLE", 0, Lifetime::Timeframe }, a);
    pc.outputs().snapshot(Output{ "TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe },
                          o2::framework::ROOTSerialized<decltype(a)>(a));
    // class Polymorphic is not messageable, so the serialization type is deduced
    // from the fact that the type has a dictionary and can be ROOT-serialized.
    pc.outputs().snapshot(Output{ "TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe }, b);
    // vector of ROOT serializable class
    pc.outputs().snapshot(Output{ "TST", "ROOTVECTOR", 0, Lifetime::Timeframe }, c);
    // likewise, passed anonymously with char type and class name
    o2::framework::ROOTSerialized<char, const char> d(*((char*)&c), "vector<o2::test::Polymorphic>");
    pc.outputs().snapshot(Output{ "TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe }, d);
    // vector of ROOT serializable class wrapped with TClass info as hint
    auto* cl = TClass::GetClass(typeid(decltype(c)));
    ASSERT_ERROR(cl != nullptr);
    o2::framework::ROOTSerialized<char, TClass> e(*((char*)&c), cl);
    pc.outputs().snapshot(Output{ "TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe }, e);
  };

  return DataProcessorSpec{ "source", // name of the processor
                            { InputSpec{ "timer", "TST", "TIMER", 0, Lifetime::Timeframe } },
                            { OutputSpec{ "TST", "MESSAGEABLE", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "ROOTVECTOR", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(processingFct) };
}

DataProcessorSpec getSinkSpec()
{
  auto processingFct = [](ProcessingContext& pc) {
    using DataHeader = o2::header::DataHeader;
    for (auto& input : pc.inputs()) {
      auto dh = o2::header::get<const DataHeader*>(input.header);
      LOG(INFO) << dh->dataOrigin.str << " " << dh->dataDescription.str << " " << dh->payloadSize;
    }
    // plain, unserialized object in input1 channel
    auto object1 = pc.inputs().get<o2::test::TriviallyCopyable>("input1");
    ASSERT_ERROR(object1 != nullptr);
    ASSERT_ERROR(*object1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    // ROOT-serialized messageable object in input2 channel
    auto object2 = pc.inputs().get<o2::test::TriviallyCopyable>("input2");
    ASSERT_ERROR(object2 != nullptr);
    ASSERT_ERROR(*object2 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    // ROOT-serialized, non-messageable object in input3 channel
    auto object3 = pc.inputs().get<o2::test::Polymorphic>("input3");
    ASSERT_ERROR(object3 != nullptr);
    ASSERT_ERROR(*object3 == o2::test::Polymorphic(0xbeef));

    // container of objects
    auto object4 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input4");
    ASSERT_ERROR(object4.size() == 2);
    ASSERT_ERROR(object4[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object4[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    auto object5 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input5");
    ASSERT_ERROR(object5.size() == 2);
    ASSERT_ERROR(object5[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object5[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    auto object6 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input6");
    ASSERT_ERROR(object6.size() == 2);
    ASSERT_ERROR(object6[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object6[1] == o2::test::Polymorphic(0xd00f));

    // checking retrieving buffer as raw char*, and checking content by cast
    auto rawchar = pc.inputs().get<const char*>("input1");
    const auto& data1 = *reinterpret_cast<const o2::test::TriviallyCopyable*>(rawchar);
    ASSERT_ERROR(data1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    pc.services().get<ControlService>().readyToQuit(true);
  };

  return DataProcessorSpec{ "sink", // name of the processor
                            { InputSpec{ "input1", "TST", "MESSAGEABLE", 0, Lifetime::Timeframe },
                              InputSpec{ "input2", "TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe },
                              InputSpec{ "input3", "TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe },
                              InputSpec{ "input4", "TST", "ROOTVECTOR", 0, Lifetime::Timeframe },
                              InputSpec{ "input5", "TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe },
                              InputSpec{ "input6", "TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe } },
                            Outputs{},
                            AlgorithmSpec(processingFct) };
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    getTimeoutSpec(),
    getSourceSpec(),
    getSinkSpec()
  };
}
