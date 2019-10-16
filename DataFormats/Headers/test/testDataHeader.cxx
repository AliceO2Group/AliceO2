// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Headers DataHeaderTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "Headers/DataHeader.h"
#include "Headers/NameHeader.h"
#include "Headers/Stack.h"

#include <chrono>

using system_clock = std::chrono::system_clock;
using TimeScale = std::chrono::nanoseconds;

namespace o2
{
namespace header
{
namespace test
{
struct MetaHeader : public BaseHeader {
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
} // namespace header
} // namespace o2

namespace o2
{
namespace header
{

BOOST_AUTO_TEST_CASE(Descriptor_test)
{
  // test for the templated Descriptor struct
  constexpr int descriptorSize = 8;
  using TestDescriptorT = Descriptor<descriptorSize>;
  BOOST_CHECK(TestDescriptorT::size == descriptorSize);
  BOOST_CHECK(TestDescriptorT::bitcount == descriptorSize * 8);
  BOOST_CHECK(sizeof(TestDescriptorT::ItgType) * TestDescriptorT::arraySize == descriptorSize);
  static_assert(TestDescriptorT::size == sizeof(TestDescriptorT),
                "Descriptor must have size of the underlying data member");

  // the Descriptor allows to define an integral value from
  // a char sequence so that the in-memory representation shows
  // a readable pattern
  constexpr char readable[] = "DESCRPTR";
  // the inverse sequence as integer value:       R T P R C S E D
  constexpr TestDescriptorT::ItgType itgvalue = 0x5254505243534544;

  TestDescriptorT testDescriptor(readable);
  TestDescriptorT anotherDescriptor("ANOTHER");
  BOOST_CHECK(TestDescriptorT::size == sizeof(testDescriptor));

  // copy constructor and comparison operator
  BOOST_CHECK(testDescriptor == TestDescriptorT(testDescriptor));

  // comparison operator
  BOOST_CHECK(testDescriptor != anotherDescriptor);

  // type cast operator
  BOOST_CHECK(itgvalue == testDescriptor);

  // assignment operator
  anotherDescriptor = testDescriptor;
  BOOST_CHECK(testDescriptor == anotherDescriptor);
  anotherDescriptor = TestDescriptorT("SOMEMORE");
  BOOST_CHECK(itgvalue != anotherDescriptor);

  // assignment using implicit conversion with constructor
  anotherDescriptor = itgvalue;
  BOOST_CHECK(testDescriptor == anotherDescriptor);

  // check with runtime string
  std::string runtimeString = "RUNTIMES";
  TestDescriptorT runtimeDescriptor;
  runtimeDescriptor.runtimeInit(runtimeString.c_str());
  BOOST_CHECK(runtimeDescriptor == TestDescriptorT("RUNTIMES"));

  BOOST_CHECK(testDescriptor.as<std::string>().length() == 8);
}

BOOST_AUTO_TEST_CASE(DataDescription_test)
{
  char test[16] = "ITSRAW*********";
  test[6] = 0;

  DataDescription desc("ITSRAW");
  BOOST_CHECK(strcmp(desc.str, "ITSRAW") == 0);

  // checking the corresponding integer value
  // the upper part must be 0 since the string has only up tp 8 chars
  // lower part corresponds to reverse ITSRAW
  //                       W A R S T I
  uint64_t itgDesc = 0x0000574152535449;
  BOOST_CHECK(desc.itg[0] == itgDesc);
  BOOST_CHECK(desc.itg[1] == 0);

  BOOST_CHECK(desc == DataDescription("ITSRAW"));

  DataDescription desc2(test);
  BOOST_CHECK(strcmp(desc2.str, "ITSRAW") == 0);
  // the upper part must be 0 since the string has only up tp 8 chars
  BOOST_CHECK(desc2.itg[1] == 0);

  BOOST_CHECK(desc2 == DataDescription("ITSRAW"));

  std::string runtimeString = "DATA_DESCRIPTION";
  DataDescription runtimeDesc;
  runtimeDesc.runtimeInit(runtimeString.c_str());
  BOOST_CHECK(runtimeDesc == DataDescription("DATA_DESCRIPTION"));

  BOOST_CHECK(desc.as<std::string>().length() == 6);
  BOOST_CHECK(runtimeDesc.as<std::string>().length() == 16);
  BOOST_CHECK(DataDescription("INVALIDDATA").as<std::string>().length() == 11);

  BOOST_CHECK(DataDescription("A") < DataDescription("B"));
  BOOST_CHECK(DataDescription("AA") < DataDescription("AB"));
  BOOST_CHECK(DataDescription("AAA") < DataDescription("AAB"));
  BOOST_CHECK(DataDescription("AAA") < DataDescription("ABA"));
}

BOOST_AUTO_TEST_CASE(DataOrigin_test)
{
  // test for the templated Descriptor struct
  constexpr int descriptorSize = 4;
  using TestDescriptorT = Descriptor<descriptorSize>;
  BOOST_CHECK(TestDescriptorT::size == descriptorSize);
  BOOST_CHECK(TestDescriptorT::bitcount == descriptorSize * 8);
  BOOST_CHECK(sizeof(TestDescriptorT::ItgType) * TestDescriptorT::arraySize == descriptorSize);
  BOOST_CHECK(TestDescriptorT::size == sizeof(DataOrigin));

  // we want to explicitely have the size of DataOrigin to be 4
  static_assert(sizeof(DataOrigin) == 4,
                "DataOrigin struct must be of size 4");

  // Check that ordering works.
  BOOST_CHECK(DataOrigin("A") < DataOrigin("B"));
  BOOST_CHECK(DataOrigin("AA") < DataOrigin("AB"));
  BOOST_CHECK(DataOrigin("AAA") < DataOrigin("AAB"));
  BOOST_CHECK(DataOrigin("AAA") < DataOrigin("ABA"));
  std::vector<DataOrigin> v1 = {DataOrigin("B"), DataOrigin("C"), DataOrigin("A")};
  std::sort(v1.begin(), v1.end());
  BOOST_CHECK_EQUAL(v1[0], DataOrigin("A"));
  BOOST_CHECK_EQUAL(v1[1], DataOrigin("B"));
  BOOST_CHECK_EQUAL(v1[2], DataOrigin("C"));
  std::vector<DataOrigin> v2 = {DataOrigin("A"), DataOrigin("B")};
  std::sort(v2.begin(), v2.end());
  BOOST_CHECK_EQUAL(v2[0], DataOrigin("A"));
  BOOST_CHECK_EQUAL(v2[1], DataOrigin("B"));

  using CustomHeader = std::tuple<DataOrigin, DataDescription>;
  std::vector<CustomHeader> v3{CustomHeader{"TST", "B"}, CustomHeader{"TST", "A"}};
  std::sort(v3.begin(), v3.end());
  auto h0 = CustomHeader{"TST", "A"};
  auto h1 = CustomHeader{"TST", "B"};
  BOOST_CHECK(v3[0] == h0);
  BOOST_CHECK(v3[1] == h1);

  using CustomHeader2 = std::tuple<DataOrigin, DataDescription, int>;
  std::vector<CustomHeader2> v4{CustomHeader2{"TST", "A", 1}, CustomHeader2{"TST", "A", 0}};
  std::sort(v4.begin(), v4.end());
  auto hh0 = CustomHeader2{"TST", "A", 0};
  auto hh1 = CustomHeader2{"TST", "A", 1};
  BOOST_CHECK(v4[0] == hh0);
  BOOST_CHECK(v4[1] == hh1);

  struct CustomHeader3 {
    DataOrigin origin;
    DataDescription desc;
    uint64_t subSpec;
    int isOut;
  };
  std::vector<CustomHeader3> v5{CustomHeader3{"TST", "A", 0, 1}, CustomHeader3{"TST", "A", 0, 0}};
  std::sort(v5.begin(), v5.end(), [](CustomHeader3 const& lhs, CustomHeader3 const& rhs) {
    return std::tie(lhs.origin, lhs.desc, rhs.subSpec, lhs.isOut) < std::tie(rhs.origin, rhs.desc, rhs.subSpec, rhs.isOut);
  });
  BOOST_CHECK(v5[0].isOut == 0);
  BOOST_CHECK(v5[1].isOut == 1);
}

BOOST_AUTO_TEST_CASE(BaseHeader_test)
{
  static_assert(sizeof(HeaderType) == 8,
                "HeaderType struct must be of size 8");
  static_assert(sizeof(SerializationMethod) == 8,
                "SerializationMethod struct must be of size 8");
  static_assert(sizeof(BaseHeader) == 32,
                "BaseHeader struct must be of size 32");
}

BOOST_AUTO_TEST_CASE(DataHeader_test)
{
  DataHeader dh;
  bool verbose = false;
  if (verbose) {
    std::cout << "size of BaseHeader: " << sizeof(BaseHeader) << std::endl;
    std::cout << "size of DataHeader: " << sizeof(DataHeader) << std::endl;

    std::cout << "dataDescription:             "
              << "size " << std::setw(2) << sizeof(dh.dataDescription) << " at " << (char*)(&dh.dataDescription) - (char*)(&dh) << std::endl;
    std::cout << "dataOrigin:                  "
              << "size " << std::setw(2) << sizeof(dh.dataOrigin) << " at " << (char*)(&dh.dataOrigin) - (char*)(&dh) << std::endl;
    std::cout << "splitPayloadParts:          "
              << "size " << std::setw(2) << sizeof(dh.splitPayloadParts) << " at "
              << (char*)(&dh.splitPayloadParts) - (char*)(&dh) << std::endl;
    std::cout << "payloadSerializationMethod:  "
              << "size " << std::setw(2) << sizeof(dh.payloadSerializationMethod) << " at " << (char*)(&dh.payloadSerializationMethod) - (char*)(&dh) << std::endl;
    std::cout << "subSpecification:            "
              << "size " << std::setw(2) << sizeof(dh.subSpecification) << " at " << (char*)(&dh.subSpecification) - (char*)(&dh) << std::endl;
    std::cout << "splitPayloadIndex:           "
              << "size " << std::setw(2) << sizeof(dh.splitPayloadIndex) << " at "
              << (char*)(&dh.splitPayloadIndex) - (char*)(&dh) << std::endl;
    std::cout << "payloadSize                  "
              << "size " << std::setw(2) << sizeof(dh.payloadSize) << " at " << (char*)(&dh.payloadSize) - (char*)(&dh) << std::endl;
  }

  // DataHeader must have size 80
  static_assert(sizeof(DataHeader) == 80,
                "DataHeader struct must be of size 80");
  DataHeader dh2;
  BOOST_CHECK(dh == dh2);
  DataHeader dh3{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}, 0};
  BOOST_CHECK(dh == dh3);
  DataHeader dh4{gDataDescriptionAny, gDataOriginAny, DataHeader::SubSpecificationType{1}, 1};
  BOOST_CHECK(!(dh4 == dh));
  dh4 = dh;
  BOOST_CHECK(dh4 == dh);
}

BOOST_AUTO_TEST_CASE(headerStack_test)
{

  DataHeader dh1{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}, 0};

  Stack s1{DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}, 0},
           NameHeader<9>{"somename"}};

  const DataHeader* h1 = get<DataHeader*>(s1.data());
  BOOST_CHECK(h1 != nullptr);
  BOOST_CHECK(*h1 == dh1);
  BOOST_CHECK(h1->flagsNextHeader == 1);
  const NameHeader<0>* h2 = get<NameHeader<0>*>(s1.data());
  BOOST_CHECK(h2 != nullptr);
  BOOST_CHECK(0 == std::strcmp(h2->getName(), "somename"));
  BOOST_CHECK(h2->description == NameHeader<0>::sHeaderType);
  BOOST_CHECK(h2->serialization == gSerializationMethodNone);
  BOOST_CHECK(h2->size() == sizeof(NameHeader<9>));
  BOOST_CHECK(h2->getNameLength() == 9);

  // create new stack from stack and additional header
  auto meta = test::MetaHeader{42};
  Stack s2{s1, meta};
  BOOST_CHECK(s2.size() == s1.size() + sizeof(decltype(meta)));

  auto* h3 = get<test::MetaHeader*>(s1.data());
  BOOST_CHECK(h3 == nullptr);
  h3 = get<test::MetaHeader*>(s2.data());
  BOOST_REQUIRE(h3 != nullptr);
  BOOST_CHECK(h3->flagsNextHeader == false);
  h1 = get<DataHeader*>(s2.data());
  BOOST_REQUIRE(h1 != nullptr);
  BOOST_CHECK(h1->flagsNextHeader == true);

  // create stack from header and empty stack
  Stack s3{meta, Stack{}};
  BOOST_CHECK(s3.size() == sizeof(meta));
  h3 = get<test::MetaHeader*>(s3.data());
  BOOST_REQUIRE(h3 != nullptr);
  // no next header to be flagged as the stack was empty
  BOOST_CHECK(h3->flagsNextHeader == false);

  // create new stack from stack, empty stack, and header
  Stack s4{s1, Stack{}, meta};
  BOOST_CHECK(s4.size() == s1.size() + sizeof(meta));
  // check if we can find the header even though there was an empty stack in the middle
  h3 = get<test::MetaHeader*>(s4.data());
  BOOST_REQUIRE(h3 != nullptr);
  BOOST_CHECK(h3->secret == 42);

  //test constructing from a buffer and an additional header
  using namespace boost::container::pmr;
  Stack s5(new_delete_resource(), s1.data(), Stack{}, meta);
  BOOST_CHECK(s5.size() == s1.size() + sizeof(meta));
  // check if we can find the header even though there was an empty stack in the middle
  h3 = get<test::MetaHeader*>(s5.data());
  BOOST_REQUIRE(h3 != nullptr);
  BOOST_CHECK(h3->secret == 42);
  auto* h4 = Stack::lastHeader(s5.data());
  auto* h5 = Stack::firstHeader(s5.data());
  auto* h6 = get<DataHeader*>(s5.data());
  BOOST_REQUIRE(h5 == h6);
  BOOST_REQUIRE(h5 != nullptr);
  BOOST_CHECK(h4 == h3);

  // let's assume we have some stack that is missing the required DataHeader at the beginning:
  Stack s6{new_delete_resource(), DataHeader{}, s1.data()};
  BOOST_CHECK(s6.size() == sizeof(DataHeader) + s1.size());
}

BOOST_AUTO_TEST_CASE(Descriptor_benchmark)
{
  using TestDescriptor = Descriptor<8>;
  TestDescriptor a("TESTDESC");
  TestDescriptor b(a);

  auto refTime = system_clock::now();
  const int nrolls = 1000000;
  for (auto count = 0; count < nrolls; ++count) {
    if (a == b) {
      ++a.itg[0];
      ++b.itg[0];
    }
  }
  auto duration = std::chrono::duration_cast<TimeScale>(std::chrono::system_clock::now() - refTime);
  std::cout << nrolls << " operation(s): " << duration.count() << " ns" << std::endl;
  // there is not really a check at the moment
}
} // namespace header
} // namespace o2
