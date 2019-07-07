#define BOOST_TEST_MODULE Ex4
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "Ex4/A.h"

#include <iostream>

BOOST_AUTO_TEST_CASE(AValueShouldBeTheAnswer)
{
  ex4::A a;
  BOOST_CHECK_EQUAL(a.value(), 42);
}

