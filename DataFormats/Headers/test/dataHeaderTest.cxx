#define BOOST_TEST_MODULE Test Headers DataHeaderTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "Headers/DataHeader.h"

namespace AliceO2 {
  namespace Header {

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
    }
  } 
}

