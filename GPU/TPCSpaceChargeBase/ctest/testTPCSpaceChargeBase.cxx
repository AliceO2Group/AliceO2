#define BOOST_TEST_MODULE Test TPC Space-Charge Base Class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "AliTPCSpaceCharge3DCalc.h"

/// @brief Basic test if we can create the method class
BOOST_AUTO_TEST_CASE(TPCSpaceChargeBase_test1)
{
  auto spacecharge = new AliTPCSpaceCharge3DCalc;
  delete spacecharge;
}
