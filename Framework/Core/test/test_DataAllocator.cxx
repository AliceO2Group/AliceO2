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
#include "Framework/RawDeviceService.h"
#include "Framework/SerializationMethods.h"
#include "Framework/OutputRoute.h"
#include "Headers/DataHeader.h"
#include "TestClasses.h"
#include "Framework/Logger.h"
#include <fairmq/FairMQDevice.h>
#include <vector>
#include <chrono>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
  }

// this function is only used to do the static checks for API return types
void doTypeChecks()
{
  TimingInfo* timingInfo = nullptr;
  ContextRegistry* contextes = nullptr;
  std::vector<OutputRoute> routes;
  DataAllocator allocator(timingInfo, contextes, routes);
  const Output output{ "TST", "DUMMY", 0, Lifetime::Timeframe };
  // we require references to objects owned by allocator context
  static_assert(std::is_lvalue_reference<decltype(allocator.make<int>(output))>::value);
  static_assert(std::is_lvalue_reference<decltype(allocator.make<std::string>(output, "test"))>::value);
  static_assert(std::is_lvalue_reference<decltype(allocator.make<std::vector<int>>(output))>::value);
}

namespace test
{
struct MetaHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  static const o2::header::HeaderType sHeaderType;
  static const uint32_t sVersion = 1;

  MetaHeader(uint32_t v)
    : BaseHeader(sizeof(MetaHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), secret(v)
  {
  }

  uint64_t secret;
};
constexpr o2::header::HeaderType MetaHeader::sHeaderType = "MetaHead";
}

DataProcessorSpec getTimeoutSpec()
{
  // a timer process to terminate the workflow after a timeout
  auto processingFct = [](ProcessingContext& pc) {
    static int counter = 0;
    pc.outputs().snapshot(Output{ "TEST", "TIMER", 0, Lifetime::Timeframe }, counter);

    // terminate if WaitFor was not interrupted
    if (pc.services().get<RawDeviceService>().device()->WaitFor(std::chrono::seconds(1)) && (counter++ > 10)) {
      LOG(ERROR) << "Timeout reached, the workflow seems to be broken";
      pc.services().get<ControlService>().readyToQuit(true);
    }
  };

  return DataProcessorSpec{ "timer",  // name of the processor
                            Inputs{}, // inputs empty
                            { OutputSpec{ "TEST", "TIMER", 0, Lifetime::Timeframe } },
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
    test::MetaHeader meta1{ 42 };
    test::MetaHeader meta2{ 23 };
    pc.outputs().snapshot(Output{ "TST", "MESSAGEABLE", 0, Lifetime::Timeframe, { meta1, meta2 } }, a);
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
    // test the 'make' methods
    pc.outputs().make<o2::test::TriviallyCopyable>(OutputRef{ "makesingle", 0 }) = a;
    auto& multi = pc.outputs().make<o2::test::TriviallyCopyable>(OutputRef{ "makespan", 0 }, 3);
    ASSERT_ERROR(multi.size() == 3);
    for (auto& object : multi) {
      object = a;
    }
    // test the adopt method
    auto freefct = [](void* data, void* hint) {}; // simply ignore the cleanup for the test
    static std::string teststring = "adoptchunk";
    pc.outputs().adoptChunk(Output{ "TST", "ADOPTCHUNK", 0, Lifetime::Timeframe }, teststring.data(), teststring.length(), freefct, nullptr);
    // test resizable data chunk, initial size 0 and grow
    auto& growchunk = pc.outputs().newChunk(OutputRef{ "growchunk", 0 }, 0);
    growchunk.resize(sizeof(o2::test::TriviallyCopyable));
    memcpy(growchunk.data(), &a, sizeof(o2::test::TriviallyCopyable));
    // test resizable data chunk, large initial size and shrink
    auto& shrinkchunk = pc.outputs().newChunk(OutputRef{ "shrinkchunk", 0 }, 1000000);
    shrinkchunk.resize(sizeof(o2::test::TriviallyCopyable));
    memcpy(shrinkchunk.data(), &a, sizeof(o2::test::TriviallyCopyable));
    auto& messageablevector = pc.outputs().make<std::vector<o2::test::TriviallyCopyable>>(OutputRef{ "messageablevector", 0 });
    ASSERT_ERROR(messageablevector.size() == 0);
    messageablevector.push_back(a);
    messageablevector.emplace_back(10, 20, 0xacdc);
  };

  return DataProcessorSpec{ "source", // name of the processor
                            { InputSpec{ "timer", "TEST", "TIMER", 0, Lifetime::Timeframe } },
                            { OutputSpec{ "TST", "MESSAGEABLE", 0, Lifetime::Timeframe },
                              OutputSpec{ { "makesingle" }, "TST", "MAKESINGLE", 0, Lifetime::Timeframe },
                              OutputSpec{ { "makespan" }, "TST", "MAKESPAN", 0, Lifetime::Timeframe },
                              OutputSpec{ { "growchunk" }, "TST", "GROWCHUNK", 0, Lifetime::Timeframe },
                              OutputSpec{ { "shrinkchunk" }, "TST", "SHRINKCHUNK", 0, Lifetime::Timeframe },
                              OutputSpec{ { "messageablevector" }, "TST", "MSGABLVECTOR", 0, Lifetime::Timeframe },
                              OutputSpec{ "TST", "ADOPTCHUNK", 0, Lifetime::Timeframe },
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
      auto* dh = o2::header::get<const DataHeader*>(input.header);
      LOG(INFO) << dh->dataOrigin.str << " " << dh->dataDescription.str << " " << dh->payloadSize;

      using DumpStackFctType = std::function<void(const o2::header::BaseHeader*)>;
      DumpStackFctType dumpStack = [&](const o2::header::BaseHeader* h) {
        o2::header::hexDump("", h, h->size());
        if (h->flagsNextHeader) {
          auto next = reinterpret_cast<const o2::byte*>(h) + h->size();
          dumpStack(reinterpret_cast<const o2::header::BaseHeader*>(next));
        }
      };

      dumpStack(dh);
    }
    // plain, unserialized object in input1 channel
    LOG(INFO) << "extracting o2::test::TriviallyCopyable from input1";
    auto object1 = pc.inputs().get<o2::test::TriviallyCopyable>("input1");
    ASSERT_ERROR(object1 == o2::test::TriviallyCopyable(42, 23, 0xdead));
    LOG(INFO) << "extracting span of o2::test::TriviallyCopyable from input1";
    auto object1span = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("input1");
    ASSERT_ERROR(object1span.size() == 1);
    ASSERT_ERROR(sizeof(typename decltype(object1span)::value_type) == sizeof(o2::test::TriviallyCopyable));
    // check the additional header on the stack
    auto* metaHeader1 = DataRefUtils::getHeader<test::MetaHeader*>(pc.inputs().get("input1"));
    // check if there are more of the same type
    auto* metaHeader2 = metaHeader1 ? o2::header::get<test::MetaHeader*>(metaHeader1->next()) : nullptr;
    ASSERT_ERROR(metaHeader1 != nullptr);
    ASSERT_ERROR(metaHeader1->secret == 42);
    ASSERT_ERROR(metaHeader2 != nullptr && metaHeader2->secret == 23);

    // ROOT-serialized messageable object in input2 channel
    LOG(INFO) << "extracting o2::test::TriviallyCopyable pointer from input2";
    auto object2 = pc.inputs().get<o2::test::TriviallyCopyable*>("input2");
    ASSERT_ERROR(object2 != nullptr);
    ASSERT_ERROR(*object2 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    // ROOT-serialized, non-messageable object in input3 channel
    LOG(INFO) << "extracting o2::test::Polymorphic pointer from input3";
    auto object3 = pc.inputs().get<o2::test::Polymorphic*>("input3");
    ASSERT_ERROR(object3 != nullptr);
    ASSERT_ERROR(*object3 == o2::test::Polymorphic(0xbeef));

    // container of objects
    LOG(INFO) << "extracting vector of o2::test::Polymorphic from input4";
    auto object4 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input4");
    ASSERT_ERROR(object4.size() == 2);
    ASSERT_ERROR(object4[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object4[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    LOG(INFO) << "extracting vector of o2::test::Polymorphic from input5";
    auto object5 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input5");
    ASSERT_ERROR(object5.size() == 2);
    ASSERT_ERROR(object5[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object5[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    LOG(INFO) << "extracting vector of o2::test::Polymorphic from input6";
    auto object6 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input6");
    ASSERT_ERROR(object6.size() == 2);
    ASSERT_ERROR(object6[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object6[1] == o2::test::Polymorphic(0xd00f));

    // checking retrieving buffer as raw char*, and checking content by cast
    LOG(INFO) << "extracting raw char* from input1";
    auto rawchar = pc.inputs().get<const char*>("input1");
    const auto& data1 = *reinterpret_cast<const o2::test::TriviallyCopyable*>(rawchar);
    ASSERT_ERROR(data1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(INFO) << "extracting o2::test::TriviallyCopyable from input7";
    auto object7 = pc.inputs().get<o2::test::TriviallyCopyable>("input7");
    ASSERT_ERROR(object1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(INFO) << "extracting span of o2::test::TriviallyCopyable from input8";
    auto objectspan8 = DataRefUtils::as<o2::test::TriviallyCopyable>(pc.inputs().get("input8"));
    ASSERT_ERROR(objectspan8.size() == 3);
    for (auto const& object8 : objectspan8) {
      ASSERT_ERROR(object8 == o2::test::TriviallyCopyable(42, 23, 0xdead));
    }

    LOG(INFO) << "extracting std::string from input9";
    auto object9 = pc.inputs().get<std::string>("input9");
    ASSERT_ERROR(object9 == "adoptchunk");

    LOG(INFO) << "extracting o2::test::TriviallyCopyable from input10";
    auto object10 = pc.inputs().get<o2::test::TriviallyCopyable>("input10");
    ASSERT_ERROR(object10 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(INFO) << "extracting o2::test::TriviallyCopyable from input11";
    auto object11 = pc.inputs().get<o2::test::TriviallyCopyable>("input11");
    ASSERT_ERROR(object11 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(INFO) << "extracting the original std::vector<o2::test::TriviallyCopyable> as span from input12";
    auto object12 = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("input12");
    ASSERT_ERROR(object12.size() == 2);
    ASSERT_ERROR((object12[0] == o2::test::TriviallyCopyable{ 42, 23, 0xdead }));
    ASSERT_ERROR((object12[1] == o2::test::TriviallyCopyable{ 10, 20, 0xacdc }));

    pc.services().get<ControlService>().readyToQuit(true);
  };

  return DataProcessorSpec{ "sink", // name of the processor
                            { InputSpec{ "input1", "TST", "MESSAGEABLE", 0, Lifetime::Timeframe },
                              InputSpec{ "input2", "TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe },
                              InputSpec{ "input3", "TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe },
                              InputSpec{ "input4", "TST", "ROOTVECTOR", 0, Lifetime::Timeframe },
                              InputSpec{ "input5", "TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe },
                              InputSpec{ "input6", "TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe },
                              InputSpec{ "input7", "TST", "MAKESINGLE", 0, Lifetime::Timeframe },
                              InputSpec{ "input8", "TST", "MAKESPAN", 0, Lifetime::Timeframe },
                              InputSpec{ "input9", "TST", "ADOPTCHUNK", 0, Lifetime::Timeframe },
                              InputSpec{ "input10", "TST", "GROWCHUNK", 0, Lifetime::Timeframe },
                              InputSpec{ "input11", "TST", "SHRINKCHUNK", 0, Lifetime::Timeframe },
                              InputSpec{ "input12", "TST", "MSGABLVECTOR", 0, Lifetime::Timeframe } },
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
