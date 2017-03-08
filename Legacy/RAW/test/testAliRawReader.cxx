#define BOOST_TEST_MODULE Test AliRawReader legacy interface
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../src/AliRawReader.h"

namespace AliceO2 {
  namespace Header {

    BOOST_AUTO_TEST_CASE(AliRawReader_test)
    {
      AliRawReader* rawreader = AliRawReader::Create("raw.root");
      BOOST_CHECK(rawreader != nullptr);

      rawreader->RewindEvents();
      int eventCount=0;
      if (!rawreader->NextEvent()) {
        std::cout << "info: no events found in raw.root" << std::endl;
      } else {
        do {
          std::cout << " event " << eventCount << std::endl;
          ++eventCount;
        } while (rawreader->NextEvent());
      }
    }
  }
}
