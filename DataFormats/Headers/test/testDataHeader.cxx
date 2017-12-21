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

#include <chrono>

using system_clock = std::chrono::system_clock;
using TimeScale = std::chrono::nanoseconds;

namespace o2 {
  namespace header {

    BOOST_AUTO_TEST_CASE(Descriptor_test)
    {
      // test for the templated Descriptor struct
      constexpr int descriptorSize = 8;
      using TestDescriptorT = Descriptor<descriptorSize>;
      BOOST_CHECK(TestDescriptorT::size == descriptorSize);
      BOOST_CHECK(TestDescriptorT::bitcount == descriptorSize * 8);
      BOOST_CHECK(sizeof(TestDescriptorT::ItgType)*TestDescriptorT::arraySize == descriptorSize);
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
    }

    BOOST_AUTO_TEST_CASE(DataDescription_test)
    {
      char test[16]="ITSRAW*********";
      test[6] = 0;

      DataDescription desc("ITSRAW");
      BOOST_CHECK(strcmp(desc.str, "ITSRAW")==0);

      // checking the corresponding integer value
      // the upper part must be 0 since the string has only up tp 8 chars
      // lower part corresponds to reverse ITSRAW
      //                       W A R S T I
      uint64_t itgDesc = 0x0000574152535449;
      BOOST_CHECK(desc.itg[0] == itgDesc);
      BOOST_CHECK(desc.itg[1] == 0);

      BOOST_CHECK(desc == DataDescription("ITSRAW"));

      DataDescription desc2(test);
      BOOST_CHECK(strcmp(desc2.str, "ITSRAW")==0);
      // the upper part must be 0 since the string has only up tp 8 chars
      BOOST_CHECK(desc2.itg[1] == 0);

      BOOST_CHECK(desc2 == DataDescription("ITSRAW"));

      std::string runtimeString = "DATA_DESCRIPTION";
      DataDescription runtimeDesc;
      runtimeDesc.runtimeInit(runtimeString.c_str());
      BOOST_CHECK(runtimeDesc == DataDescription("DATA_DESCRIPTION"));
    }

    BOOST_AUTO_TEST_CASE(DataOrigin_test)
    {
      // test for the templated Descriptor struct
      constexpr int descriptorSize = 4;
      using TestDescriptorT = Descriptor<descriptorSize>;
      BOOST_CHECK(TestDescriptorT::size == descriptorSize);
      BOOST_CHECK(TestDescriptorT::bitcount == descriptorSize * 8);
      BOOST_CHECK(sizeof(TestDescriptorT::ItgType)*TestDescriptorT::arraySize == descriptorSize);
      BOOST_CHECK(TestDescriptorT::size == sizeof(DataOrigin));

      // we want to explicitely have the size of DataOrigin to be 4
      static_assert(sizeof(DataOrigin) == 4,
                    "DataOrigin struct must be of size 4");
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

        std::cout << "dataDescription:             " << "size " << std::setw(2) << sizeof(dh.dataDescription)            << " at " << (char *)(&dh.dataDescription) - (char *)(&dh) << std::endl;
        std::cout << "dataOrigin:                  " << "size " << std::setw(2) << sizeof(dh.dataOrigin)                 << " at " << (char *)(&dh.dataOrigin) - (char *)(&dh) << std::endl;
        std::cout << "reserved:                    " << "size " << std::setw(2) << sizeof(dh.reserved)                   << " at " << (char *)(&dh.reserved) - (char *)(&dh) << std::endl;
        std::cout << "payloadSerializationMethod:  " << "size " << std::setw(2) << sizeof(dh.payloadSerializationMethod) << " at " << (char *)(&dh.payloadSerializationMethod) - (char *)(&dh) << std::endl;
        std::cout << "subSpecification:            " << "size " << std::setw(2) << sizeof(dh.subSpecification)           << " at " << (char *)(&dh.subSpecification) - (char *)(&dh) << std::endl;
        std::cout << "payloadSize                  " << "size " << std::setw(2) << sizeof(dh.payloadSize)                << " at " << (char *)(&dh.payloadSize) - (char *)(&dh) << std::endl;
      }

      // DataHeader must have size 80
      static_assert(sizeof(DataHeader) == 80,
                    "DataHeader struct must be of size 80");
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
  }
}
