#define BOOST_TEST_MODULE Test TPC Base
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCBase/Point3D.h"

namespace o2 {
  namespace TPC {
    
    BOOST_AUTO_TEST_CASE(Point3D_test)
    {
      Point3D<double> testpoint(2.,3.,4.);
      BOOST_CHECK_CLOSE(testpoint.getX(),2.,1E-12);
      BOOST_CHECK_CLOSE(testpoint.getY(),3.,1E-12);
      BOOST_CHECK_CLOSE(testpoint.getZ(),4.,1E-12);
    }
  }
} 