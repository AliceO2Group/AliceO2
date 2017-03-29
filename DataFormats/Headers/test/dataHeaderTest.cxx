#define BOOST_TEST_MODULE Test Headers DataHeaderTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "Headers/DataHeader.h"

namespace AliceO2 {
  namespace Header {

    BOOST_AUTO_TEST_CASE(Description_test)
    {
      // test for the templated Descriptor struct
      constexpr int descriptorSize = 8;
      using TestDescriptorT = Descriptor<descriptorSize>;
      BOOST_CHECK(TestDescriptorT::size == descriptorSize);
      BOOST_CHECK(TestDescriptorT::bitcount == descriptorSize * 8);
      BOOST_CHECK(sizeof(TestDescriptorT::ItgType) == descriptorSize);

      // the Descriptor allows to define an integral value from
      // a char sequence so that the in-memory representation shows
      // a readable pattern
      constexpr char readable[] = "DESCRPTR";
      // the inverse sequence as integer value:       R T P R C S E D
      constexpr TestDescriptorT::ItgType itgvalue = 0x5254505243534544;

      TestDescriptorT testDescriptor(readable);
      TestDescriptorT anotherDescriptor("ANOTHER");

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
    }

    BOOST_AUTO_TEST_CASE(DataDescription_test)
    {
      char test[16]="ITSRAW*********";
      test[6] = 0;

      DataDescription desc("ITSRAW");
      BOOST_CHECK(strcmp(desc.str, "ITSRAW")==0);

      BOOST_CHECK(desc == "ITSRAW");

      DataDescription desc2(test);
      BOOST_CHECK(strcmp(desc2.str, "ITSRAW")==0);

      BOOST_CHECK(desc2 == "ITSRAW");

      std::string runtimeString = "DATA_DESCRIPTION";
      DataDescription runtimeDesc;
      runtimeDesc.runtimeInit(runtimeString.c_str());
      BOOST_CHECK(runtimeDesc == DataDescription("DATA_DESCRIPTION"));
    }
  } 
}

