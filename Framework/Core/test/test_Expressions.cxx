// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Expressions
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "../src/ExpressionHelpers.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework::expressions;

namespace nodes
{
static BindingNode pt{"pt"};
static BindingNode phi{"phi"};
static BindingNode eta{"eta"};
} // namespace nodes

BOOST_AUTO_TEST_CASE(TestTreeParsing)
{
  Filter f = ((nodes::phi > 1) && (nodes::phi < 2)) && (nodes::eta < 1);
  auto specs = createKernelsFromFilter(f);
  BOOST_REQUIRE_EQUAL(specs[0].left, DatumSpec{1u});
  BOOST_REQUIRE_EQUAL(specs[0].right, DatumSpec{2u});
  BOOST_REQUIRE_EQUAL(specs[0].result, DatumSpec{0u});

  BOOST_REQUIRE_EQUAL(specs[1].left, DatumSpec{std::string{"eta"}});
  BOOST_REQUIRE_EQUAL(specs[1].right, DatumSpec{LiteralNode::var_t{1}});
  BOOST_REQUIRE_EQUAL(specs[1].result, DatumSpec{2u});

  BOOST_REQUIRE_EQUAL(specs[2].left, DatumSpec{3u});
  BOOST_REQUIRE_EQUAL(specs[2].right, DatumSpec{4u});
  BOOST_REQUIRE_EQUAL(specs[2].result, DatumSpec{1u});

  BOOST_REQUIRE_EQUAL(specs[3].left, DatumSpec{std::string{"phi"}});
  BOOST_REQUIRE_EQUAL(specs[3].right, DatumSpec{LiteralNode::var_t{2}});
  BOOST_REQUIRE_EQUAL(specs[3].result, DatumSpec{4u});

  BOOST_REQUIRE_EQUAL(specs[4].left, DatumSpec{std::string{"phi"}});
  BOOST_REQUIRE_EQUAL(specs[4].right, DatumSpec{LiteralNode::var_t{1}});
  BOOST_REQUIRE_EQUAL(specs[4].result, DatumSpec{3u});
}
