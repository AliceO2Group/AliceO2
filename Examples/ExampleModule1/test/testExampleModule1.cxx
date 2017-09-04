// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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