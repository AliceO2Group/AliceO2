#define BOOST_TEST_MODULE Test TPC CA GPU Tracking
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "AliHLTTPCCAO2Interface.h"

/// @brief Basic test if we can create the interface
BOOST_AUTO_TEST_CASE(CATracking_test1)
{
  auto interface = new AliHLTTPCCAO2Interface;
  interface->Initialize();
  delete interface;
}
