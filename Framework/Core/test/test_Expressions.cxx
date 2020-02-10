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

using namespace o2::framework;
using namespace o2::framework::expressions;

namespace nodes
{
static BindingNode pt{"pt", atype::FLOAT};
static BindingNode phi{"phi", atype::FLOAT};
static BindingNode eta{"eta", atype::FLOAT};
} // namespace nodes

BOOST_AUTO_TEST_CASE(TestTreeParsing)
{
  expressions::Filter f = ((nodes::phi > 1) && (nodes::phi < 2)) && (nodes::eta < 1);
  auto specs = createOperations(f);
  BOOST_REQUIRE_EQUAL(specs[0].left, (DatumSpec{1u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(specs[0].right, (DatumSpec{2u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(specs[0].result, (DatumSpec{0u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(specs[1].left, (DatumSpec{std::string{"eta"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(specs[1].right, (DatumSpec{LiteralNode::var_t{1}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(specs[1].result, (DatumSpec{2u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(specs[2].left, (DatumSpec{3u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(specs[2].right, (DatumSpec{4u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(specs[2].result, (DatumSpec{1u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(specs[3].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(specs[3].right, (DatumSpec{LiteralNode::var_t{2}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(specs[3].result, (DatumSpec{4u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(specs[4].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(specs[4].right, (DatumSpec{LiteralNode::var_t{1}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(specs[4].result, (DatumSpec{3u, atype::BOOL}));

  expressions::Filter g = ((nodes::eta + 2.f) > 0.5) || ((nodes::phi - M_PI) < 3);
  auto gspecs = createOperations(g);
  BOOST_REQUIRE_EQUAL(gspecs[0].left, (DatumSpec{1u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(gspecs[0].right, (DatumSpec{2u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(gspecs[0].result, (DatumSpec{0u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(gspecs[1].left, (DatumSpec{3u, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(gspecs[1].right, (DatumSpec{LiteralNode::var_t{3}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(gspecs[1].result, (DatumSpec{2u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(gspecs[2].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(gspecs[2].right, (DatumSpec{LiteralNode::var_t{M_PI}, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(gspecs[2].result, (DatumSpec{3u, atype::DOUBLE}));

  BOOST_REQUIRE_EQUAL(gspecs[3].left, (DatumSpec{4u, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(gspecs[3].right, (DatumSpec{LiteralNode::var_t{0.5}, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(gspecs[3].result, (DatumSpec{1u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(gspecs[4].left, (DatumSpec{std::string{"eta"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(gspecs[4].right, (DatumSpec{LiteralNode::var_t{2.f}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(gspecs[4].result, (DatumSpec{4u, atype::FLOAT}));

  expressions::Filter h = (nodes::phi == 0) || (nodes::phi == 3);
  auto hspecs = createOperations(h);

  BOOST_REQUIRE_EQUAL(hspecs[0].left, (DatumSpec{1u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(hspecs[0].right, (DatumSpec{2u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(hspecs[0].result, (DatumSpec{0u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(hspecs[1].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(hspecs[1].right, (DatumSpec{LiteralNode::var_t{3}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(hspecs[1].result, (DatumSpec{2u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(hspecs[2].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(hspecs[2].right, (DatumSpec{LiteralNode::var_t{0}, atype::INT32}));
  BOOST_REQUIRE_EQUAL(hspecs[2].result, (DatumSpec{1u, atype::BOOL}));

  expressions::Filter u = nabs(nodes::eta) < 1.0 && nexp(nodes::phi + 2.0 * M_PI) > 3.0;
  auto uspecs = createOperations(u);
  BOOST_REQUIRE_EQUAL(uspecs[0].left, (DatumSpec{1u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(uspecs[0].right, (DatumSpec{2u, atype::BOOL}));
  BOOST_REQUIRE_EQUAL(uspecs[0].result, (DatumSpec{0u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(uspecs[1].left, (DatumSpec{3u, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(uspecs[1].right, (DatumSpec{LiteralNode::var_t{3.0}, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(uspecs[1].result, (DatumSpec{2u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(uspecs[2].left, (DatumSpec{4u, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(uspecs[2].right, (DatumSpec{}));
  BOOST_REQUIRE_EQUAL(uspecs[2].result, (DatumSpec{3u, atype::DOUBLE}));

  BOOST_REQUIRE_EQUAL(uspecs[3].left, (DatumSpec{std::string{"phi"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(uspecs[3].right, (DatumSpec{LiteralNode::var_t{2.0 * M_PI}, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(uspecs[3].result, (DatumSpec{4u, atype::DOUBLE}));

  BOOST_REQUIRE_EQUAL(uspecs[4].left, (DatumSpec{5u, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(uspecs[4].right, (DatumSpec{LiteralNode::var_t{1.0}, atype::DOUBLE}));
  BOOST_REQUIRE_EQUAL(uspecs[4].result, (DatumSpec{1u, atype::BOOL}));

  BOOST_REQUIRE_EQUAL(uspecs[5].left, (DatumSpec{std::string{"eta"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(uspecs[5].right, (DatumSpec{}));
  BOOST_REQUIRE_EQUAL(uspecs[5].result, (DatumSpec{5u, atype::FLOAT}));
}
