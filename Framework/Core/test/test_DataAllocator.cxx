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

#include "Framework/RootSerializationSupport.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/RootMessageContext.h"
#include "Framework/runDataProcessing.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/RawDeviceService.h"
#include "Framework/SerializationMethods.h"
#include "Framework/OutputRoute.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "TestClasses.h"
#include "Framework/Logger.h"
#include <fairmq/Device.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <deque>
#include <utility> // std::declval
#include <TNamed.h>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

// this function is only used to do the static checks for API return types
void doTypeChecks()
{
  const Output output{"TST", "DUMMY", 0};
  // we require references to objects owned by allocator context
  static_assert(std::is_lvalue_reference<decltype(std::declval<DataAllocator>().make<int>(output))>::value);
  static_assert(std::is_lvalue_reference<decltype(std::declval<DataAllocator>().make<std::string>(output, "test"))>::value);
  static_assert(std::is_lvalue_reference<decltype(std::declval<DataAllocator>().make<std::vector<int>>(output))>::value);
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
} // namespace test

DataProcessorSpec getSourceSpec()
{
  static_assert(enable_root_serialization<o2::test::Polymorphic>::value, "enable_root_serialization<o2::test::Polymorphic> must be true");
  auto processingFct = [](ProcessingContext& pc) {
    static int counter = 0;
    o2::test::TriviallyCopyable a(42, 23, 0xdead);
    o2::test::Polymorphic b(0xbeef);
    std::vector<o2::test::Polymorphic> c{{0xaffe}, {0xd00f}};
    std::vector<o2::test::Base*> ptrVec{new o2::test::Polymorphic{0xaffe}, new o2::test::Polymorphic{0xd00f}};
    std::deque<int> testDequePayload{10, 20, 30};

    // class TriviallyCopyable is both messageable and has a dictionary, the default
    // picked by the framework is no serialization
    test::MetaHeader meta1{42};
    test::MetaHeader meta2{23};
    pc.outputs().snapshot(Output{"TST", "MESSAGEABLE", 0, {meta1, meta2}}, a);
    pc.outputs().snapshot(Output{"TST", "MSGBLEROOTSRLZ", 0},
                          o2::framework::ROOTSerialized<decltype(a)>(a));
    // class Polymorphic is not messageable, so the serialization type is deduced
    // from the fact that the type has a dictionary and can be ROOT-serialized.
    pc.outputs().snapshot(Output{"TST", "ROOTNONTOBJECT", 0}, b);
    // vector of ROOT serializable class
    pc.outputs().snapshot(Output{"TST", "ROOTVECTOR", 0}, c);
    // deque of simple types
    pc.outputs().snapshot(Output{"TST", "DEQUE", 0}, testDequePayload);
    // likewise, passed anonymously with char type and class name
    o2::framework::ROOTSerialized<char, const char> d(*((char*)&c), "vector<o2::test::Polymorphic>");
    pc.outputs().snapshot(Output{"TST", "ROOTSERLZDVEC", 0}, d);
    // vector of ROOT serializable class wrapped with TClass info as hint
    auto* cl = TClass::GetClass(typeid(decltype(c)));
    ASSERT_ERROR(cl != nullptr);
    o2::framework::ROOTSerialized<char, TClass> e(*((char*)&c), cl);
    pc.outputs().snapshot(Output{"TST", "ROOTSERLZDVEC2", 0}, e);
    // test the 'make' methods
    pc.outputs().make<o2::test::TriviallyCopyable>(OutputRef{"makesingle", 0}) = a;
    auto& multi = pc.outputs().make<o2::test::TriviallyCopyable>(OutputRef{"makespan", 0}, 3);
    ASSERT_ERROR(multi.size() == 3);
    for (auto& object : multi) {
      object = a;
    }
    // test the adopt method
    auto freefct = [](void* data, void* hint) {}; // simply ignore the cleanup for the test
    static std::string teststring = "adoptchunk";
    pc.outputs().adoptChunk(Output{"TST", "ADOPTCHUNK", 0}, teststring.data(), teststring.length(), freefct, nullptr);
    // test resizable data chunk, initial size 0 and grow
    auto& growchunk = pc.outputs().newChunk(OutputRef{"growchunk", 0}, 0);
    growchunk.resize(sizeof(o2::test::TriviallyCopyable));
    memcpy(growchunk.data(), &a, sizeof(o2::test::TriviallyCopyable));
    // test resizable data chunk, large initial size and shrink
    auto& shrinkchunk = pc.outputs().newChunk(OutputRef{"shrinkchunk", 0}, 1000000);
    shrinkchunk.resize(sizeof(o2::test::TriviallyCopyable));
    memcpy(shrinkchunk.data(), &a, sizeof(o2::test::TriviallyCopyable));
    // make Root-serializable object derived from TObject
    auto& rootobject = pc.outputs().make<TNamed>(OutputRef{"maketobject", 0}, "a_name", "a_title");
    // make Root-serializable object Non-TObject
    auto& rootpolymorphic = pc.outputs().make<o2::test::Polymorphic>(OutputRef{"makerootserlzblobj", 0}, b);
    // make vector of Root-serializable objects
    auto& rootserlzblvector = pc.outputs().make<std::vector<o2::test::Polymorphic>>(OutputRef{"rootserlzblvector", 0});
    rootserlzblvector.emplace_back(0xacdc);
    rootserlzblvector.emplace_back(0xbeef);
    // make vector of messagable objects
    auto& messageablevector = pc.outputs().make<std::vector<o2::test::TriviallyCopyable>>(OutputRef{"messageablevector", 0});
    ASSERT_ERROR(messageablevector.size() == 0);
    messageablevector.push_back(a);
    messageablevector.emplace_back(10, 20, 0xacdc);

    // create multiple parts matching the same output spec with subspec wildcard
    // we are using ConcreteDataTypeMatcher to define the output spec matcher independent
    // of subspec (i.a. a wildcard), all data blcks will go on the same channel regardless
    // of sebspec.
    pc.outputs().make<int>(OutputRef{"multiparts", 0}) = 10;
    pc.outputs().make<int>(OutputRef{"multiparts", 1}) = 20;
    pc.outputs().make<int>(OutputRef{"multiparts", 2}) = 30;

    // make a PMR std::vector, make it large to test the auto transport buffer resize funtionality as well
    Output pmrOutputSpec{"TST", "PMRTESTVECTOR", 0};
    auto pmrvec = o2::pmr::vector<o2::test::TriviallyCopyable>(pc.outputs().getMemoryResource(pmrOutputSpec));
    pmrvec.reserve(100);
    pmrvec.emplace_back(o2::test::TriviallyCopyable{1, 2, 3});
    pc.outputs().adoptContainer(pmrOutputSpec, std::move(pmrvec));

    // make a vector of POD and set some data
    pc.outputs().make<std::vector<int>>(OutputRef{"podvector"}) = {10, 21, 42};

    // vector of pointers to ROOT serializable objects
    pc.outputs().snapshot(Output{"TST", "ROOTSERLZDPTRVEC", 0}, ptrVec);

    // now we are done and signal this downstream
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    ASSERT_ERROR(pc.outputs().isAllowed({"TST", "MESSAGEABLE", 0}) == true);
    ASSERT_ERROR(pc.outputs().isAllowed({"TST", "MESSAGEABLE", 1}) == false);
    ASSERT_ERROR(pc.outputs().isAllowed({"TST", "NOWAY", 0}) == false);
    for (auto ptr : ptrVec) {
      delete ptr;
    }
  };

  return DataProcessorSpec{"source", // name of the processor
                           {},
                           {OutputSpec{"TST", "MESSAGEABLE", 0, Lifetime::Timeframe},
                            OutputSpec{{"makesingle"}, "TST", "MAKESINGLE", 0, Lifetime::Timeframe},
                            OutputSpec{{"makespan"}, "TST", "MAKESPAN", 0, Lifetime::Timeframe},
                            OutputSpec{{"growchunk"}, "TST", "GROWCHUNK", 0, Lifetime::Timeframe},
                            OutputSpec{{"shrinkchunk"}, "TST", "SHRINKCHUNK", 0, Lifetime::Timeframe},
                            OutputSpec{{"maketobject"}, "TST", "MAKETOBJECT", 0, Lifetime::Timeframe},
                            OutputSpec{{"makerootserlzblobj"}, "TST", "ROOTSERLZBLOBJ", 0, Lifetime::Timeframe},
                            OutputSpec{{"rootserlzblvector"}, "TST", "ROOTSERLZBLVECT", 0, Lifetime::Timeframe},
                            OutputSpec{{"messageablevector"}, "TST", "MSGABLVECTOR", 0, Lifetime::Timeframe},
                            OutputSpec{{"multiparts"}, "TST", "MULTIPARTS", 0, Lifetime::Timeframe},
                            OutputSpec{{"multiparts"}, "TST", "MULTIPARTS", 1, Lifetime::Timeframe},
                            OutputSpec{{"multiparts"}, "TST", "MULTIPARTS", 2, Lifetime::Timeframe},
                            OutputSpec{"TST", "ADOPTCHUNK", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "ROOTVECTOR", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "DEQUE", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe},
                            OutputSpec{"TST", "PMRTESTVECTOR", 0, Lifetime::Timeframe},
                            OutputSpec{{"podvector"}, "TST", "PODVECTOR", 0, Lifetime::Timeframe},
                            OutputSpec{{"inputPtrVec"}, "TST", "ROOTSERLZDPTRVEC", 0, Lifetime::Timeframe}},
                           AlgorithmSpec(processingFct)};
}

DataProcessorSpec getSinkSpec()
{
  auto processingFct = [](ProcessingContext& pc) {
    using DataHeader = o2::header::DataHeader;
    for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
      auto const& input = *iit;
      LOG(info) << (*iit).spec->binding << " " << (iit.isValid() ? "is valid" : "is not valid");
      if (iit.isValid() == false) {
        continue;
      }
      auto* dh = DataRefUtils::getHeader<const DataHeader*>(input);
      LOG(info) << "{" << dh->dataOrigin.str << ":" << dh->dataDescription.str << ":" << dh->subSpecification << "}"
                << " payload size " << dh->payloadSize;

      using DumpStackFctType = std::function<void(const o2::header::BaseHeader*)>;
      DumpStackFctType dumpStack = [&](const o2::header::BaseHeader* h) {
        o2::header::hexDump("", h, h->size());
        if (h->flagsNextHeader) {
          auto next = reinterpret_cast<const std::byte*>(h) + h->size();
          dumpStack(reinterpret_cast<const o2::header::BaseHeader*>(next));
        }
      };

      dumpStack(dh);

      if ((*iit).spec->binding == "inputMP") {
        LOG(info) << "inputMP with " << iit.size() << " part(s)";
        int nPart = 0;
        for (auto const& ref : iit) {
          LOG(info) << "accessing part " << nPart++ << " of input slot 'inputMP':"
                    << pc.inputs().get<int>(ref);
          ASSERT_ERROR(pc.inputs().get<int>(ref) == nPart * 10);
        }
        ASSERT_ERROR(nPart == 3);
      }
    }
    // plain, unserialized object in input1 channel
    LOG(info) << "extracting o2::test::TriviallyCopyable from input1";
    auto object1 = pc.inputs().get<o2::test::TriviallyCopyable>("input1");
    ASSERT_ERROR(object1 == o2::test::TriviallyCopyable(42, 23, 0xdead));
    LOG(info) << "extracting span of o2::test::TriviallyCopyable from input1";
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
    LOG(info) << "extracting o2::test::TriviallyCopyable pointer from input2";
    auto object2 = pc.inputs().get<o2::test::TriviallyCopyable*>("input2");
    ASSERT_ERROR(object2 != nullptr);
    ASSERT_ERROR(*object2 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    // ROOT-serialized, non-messageable object in input3 channel
    LOG(info) << "extracting o2::test::Polymorphic pointer from input3";
    auto object3 = pc.inputs().get<o2::test::Polymorphic*>("input3");
    ASSERT_ERROR(object3 != nullptr);
    ASSERT_ERROR(*object3 == o2::test::Polymorphic(0xbeef));

    // container of objects
    LOG(info) << "extracting vector of o2::test::Polymorphic from input4";
    auto object4 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input4");
    ASSERT_ERROR(object4.size() == 2);
    ASSERT_ERROR(object4[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object4[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    LOG(info) << "extracting vector of o2::test::Polymorphic from input5";
    auto object5 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input5");
    ASSERT_ERROR(object5.size() == 2);
    ASSERT_ERROR(object5[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object5[1] == o2::test::Polymorphic(0xd00f));

    // container of objects
    LOG(info) << "extracting vector of o2::test::Polymorphic from input6";
    auto object6 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input6");
    ASSERT_ERROR(object6.size() == 2);
    ASSERT_ERROR(object6[0] == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(object6[1] == o2::test::Polymorphic(0xd00f));

    // checking retrieving buffer as raw char*, and checking content by cast
    LOG(info) << "extracting raw char* from input1";
    auto rawchar = pc.inputs().get<const char*>("input1");
    const auto& data1 = *reinterpret_cast<const o2::test::TriviallyCopyable*>(rawchar);
    ASSERT_ERROR(data1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(info) << "extracting o2::test::TriviallyCopyable from input7";
    auto object7 = pc.inputs().get<o2::test::TriviallyCopyable>("input7");
    ASSERT_ERROR(object1 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(info) << "extracting span of o2::test::TriviallyCopyable from input8";
    auto objectspan8 = DataRefUtils::as<o2::test::TriviallyCopyable>(pc.inputs().get("input8"));
    ASSERT_ERROR(objectspan8.size() == 3);
    for (auto const& object8 : objectspan8) {
      ASSERT_ERROR(object8 == o2::test::TriviallyCopyable(42, 23, 0xdead));
    }

    LOG(info) << "extracting std::string from input9";
    auto object9 = pc.inputs().get<std::string>("input9");
    ASSERT_ERROR(object9 == "adoptchunk");

    LOG(info) << "extracting o2::test::TriviallyCopyable from input10";
    auto object10 = pc.inputs().get<o2::test::TriviallyCopyable>("input10");
    ASSERT_ERROR(object10 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(info) << "extracting o2::test::TriviallyCopyable from input11";
    auto object11 = pc.inputs().get<o2::test::TriviallyCopyable>("input11");
    ASSERT_ERROR(object11 == o2::test::TriviallyCopyable(42, 23, 0xdead));

    LOG(info) << "extracting the original std::vector<o2::test::TriviallyCopyable> as span from input12";
    auto object12 = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("input12");
    ASSERT_ERROR(object12.size() == 2);
    ASSERT_ERROR((object12[0] == o2::test::TriviallyCopyable{42, 23, 0xdead}));
    ASSERT_ERROR((object12[1] == o2::test::TriviallyCopyable{10, 20, 0xacdc}));
    // forward the read-only span on a different route
    pc.outputs().snapshot(Output{"TST", "MSGABLVECTORCPY", 0}, object12);

    LOG(info) << "extracting TNamed object from input13";
    auto object13 = pc.inputs().get<TNamed*>("input13");
    ASSERT_ERROR(strcmp(object13->GetName(), "a_name") == 0);
    ASSERT_ERROR(strcmp(object13->GetTitle(), "a_title") == 0);

    LOG(info) << "extracting Root-serialized Non-TObject from input14";
    auto object14 = pc.inputs().get<o2::test::Polymorphic*>("input14");
    ASSERT_ERROR(*object14 == o2::test::Polymorphic{0xbeef});

    LOG(info) << "extracting Root-serialized vector from input15";
    auto object15 = pc.inputs().get<std::vector<o2::test::Polymorphic>>("input15");
    ASSERT_ERROR(object15[0] == o2::test::Polymorphic{0xacdc});
    ASSERT_ERROR(object15[1] == o2::test::Polymorphic{0xbeef});

    LOG(info) << "extracting deque to vector from input16";
    auto object16 = pc.inputs().get<std::vector<int>>("input16");
    LOG(info) << "object16.size() = " << object16.size() << std::endl;
    ASSERT_ERROR(object16.size() == 3);
    ASSERT_ERROR(object16[0] == 10 && object16[1] == 20 && object16[2] == 30);

    LOG(info) << "extracting PMR vector";
    auto pmrspan = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("inputPMR");
    ASSERT_ERROR((pmrspan[0] == o2::test::TriviallyCopyable{1, 2, 3}));
    auto dataref = pc.inputs().get<DataRef>("inputPMR");
    auto header = DataRefUtils::getHeader<const o2::header::DataHeader*>(dataref);
    ASSERT_ERROR((header->payloadSize == sizeof(o2::test::TriviallyCopyable)));

    LOG(info) << "extracting POD vector";
    // TODO: use the ReturnType helper once implemented
    decltype(std::declval<InputRecord>().get<std::vector<int>>(DataRef{nullptr, nullptr, nullptr})) podvector;
    podvector = pc.inputs().get<std::vector<int>>("inputPODvector");
    ASSERT_ERROR(podvector.size() == 3);
    ASSERT_ERROR(podvector[0] == 10 && podvector[1] == 21 && podvector[2] == 42);

    LOG(info) << "extracting vector of o2::test::Base* from inputPtrVec";
    auto ptrVec = pc.inputs().get<std::vector<o2::test::Base*>>("inputPtrVec");
    ASSERT_ERROR(ptrVec.size() == 2);
    auto ptrVec0 = dynamic_cast<o2::test::Polymorphic*>(ptrVec[0]);
    auto ptrVec1 = dynamic_cast<o2::test::Polymorphic*>(ptrVec[1]);
    ASSERT_ERROR(ptrVec0 != nullptr);
    ASSERT_ERROR(ptrVec1 != nullptr);
    ASSERT_ERROR(*ptrVec0 == o2::test::Polymorphic(0xaffe));
    ASSERT_ERROR(*ptrVec1 == o2::test::Polymorphic(0xd00f));
    delete ptrVec[0];
    delete ptrVec[1];

    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };

  return DataProcessorSpec{"sink", // name of the processor
                           {InputSpec{"input1", "TST", "MESSAGEABLE", 0, Lifetime::Timeframe},
                            InputSpec{"input2", "TST", "MSGBLEROOTSRLZ", 0, Lifetime::Timeframe},
                            InputSpec{"input3", "TST", "ROOTNONTOBJECT", 0, Lifetime::Timeframe},
                            InputSpec{"input4", "TST", "ROOTVECTOR", 0, Lifetime::Timeframe},
                            InputSpec{"input5", "TST", "ROOTSERLZDVEC", 0, Lifetime::Timeframe},
                            InputSpec{"input6", "TST", "ROOTSERLZDVEC2", 0, Lifetime::Timeframe},
                            InputSpec{"input7", "TST", "MAKESINGLE", 0, Lifetime::Timeframe},
                            InputSpec{"input8", "TST", "MAKESPAN", 0, Lifetime::Timeframe},
                            InputSpec{"input9", "TST", "ADOPTCHUNK", 0, Lifetime::Timeframe},
                            InputSpec{"input10", "TST", "GROWCHUNK", 0, Lifetime::Timeframe},
                            InputSpec{"input11", "TST", "SHRINKCHUNK", 0, Lifetime::Timeframe},
                            InputSpec{"input12", "TST", "MSGABLVECTOR", 0, Lifetime::Timeframe},
                            InputSpec{"input13", "TST", "MAKETOBJECT", 0, Lifetime::Timeframe},
                            InputSpec{"input14", "TST", "ROOTSERLZBLOBJ", 0, Lifetime::Timeframe},
                            InputSpec{"input15", "TST", "ROOTSERLZBLVECT", 0, Lifetime::Timeframe},
                            InputSpec{"input16", "TST", "DEQUE", 0, Lifetime::Timeframe},
                            InputSpec{"inputPMR", "TST", "PMRTESTVECTOR", 0, Lifetime::Timeframe},
                            InputSpec{"inputPODvector", "TST", "PODVECTOR", 0, Lifetime::Timeframe},
                            InputSpec{"inputMP", ConcreteDataTypeMatcher{"TST", "MULTIPARTS"}, Lifetime::Timeframe},
                            InputSpec{"inputPtrVec", "TST", "ROOTSERLZDPTRVEC", 0, Lifetime::Timeframe}},
                           Outputs{OutputSpec{"TST", "MSGABLVECTORCPY", 0, Lifetime::Timeframe}},
                           AlgorithmSpec(processingFct)};
}

// a second spec subscribing to some of the same data to test forwarding of messages
DataProcessorSpec getSpectatorSinkSpec()
{
  auto processingFct = [](ProcessingContext& pc) {
    using DataHeader = o2::header::DataHeader;
    int nPart = 0;
    for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
      auto const& input = *iit;
      LOG(info) << (*iit).spec->binding << " " << (iit.isValid() ? "is valid" : "is not valid");
      if (iit.isValid() == false) {
        continue;
      }
      auto* dh = DataRefUtils::getHeader<const DataHeader*>(input);
      LOG(info) << "{" << dh->dataOrigin.str << ":" << dh->dataDescription.str << ":" << dh->subSpecification << "}"
                << " payload size " << dh->payloadSize;

      if ((*iit).spec->binding == "inputMP") {
        LOG(info) << "inputMP with " << iit.size() << " part(s)";
        for (auto const& ref : iit) {
          LOG(info) << "accessing part " << nPart << " of input slot 'inputMP':"
                    << pc.inputs().get<int>(ref);
          nPart++;
          ASSERT_ERROR(pc.inputs().get<int>(ref) == nPart * 10);
        }
      }
    }
    ASSERT_ERROR(nPart == 3);
    LOG(info) << "extracting the forwarded gsl::span<o2::test::TriviallyCopyable> as span from input12";
    auto object12 = pc.inputs().get<gsl::span<o2::test::TriviallyCopyable>>("input12");
    ASSERT_ERROR(object12.size() == 2);
    ASSERT_ERROR((object12[0] == o2::test::TriviallyCopyable{42, 23, 0xdead}));
    ASSERT_ERROR((object12[1] == o2::test::TriviallyCopyable{10, 20, 0xacdc}));

    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };

  return DataProcessorSpec{"spectator-sink", // name of the processor
                           {InputSpec{"inputMP", ConcreteDataTypeMatcher{"TST", "MULTIPARTS"}, Lifetime::Timeframe},
                            InputSpec{"input12", ConcreteDataTypeMatcher{"TST", "MSGABLVECTORCPY"}, Lifetime::Timeframe}},
                           Outputs{},
                           AlgorithmSpec(processingFct)};
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    getSourceSpec(),
    getSinkSpec(),
    getSpectatorSinkSpec()};
}
