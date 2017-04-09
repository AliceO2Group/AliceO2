///
/// \author Barthelemy von Haller
///

#include "../include/ExampleModule1/Foo.h"

#define BOOST_TEST_MODULE MO test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cassert>
#include "ExampleModule1/Foo.h"


namespace o2 {
namespace Examples {
namespace ExampleModule1 {

BOOST_AUTO_TEST_CASE(testFoo)
{
  Foo foo;
  foo.greet();
}

} /* namespace ExampleModule1 */
} /* namespace Examples */
} /* namespace AliceO2 */