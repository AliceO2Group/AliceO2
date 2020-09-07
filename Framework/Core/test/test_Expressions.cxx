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

#include "Framework/Configurable.h"
#include "../src/ExpressionHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AODReaderHelpers.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace o2::framework::expressions;

namespace nodes
{
static BindingNode pt{"pt", atype::FLOAT};
static BindingNode phi{"phi", atype::FLOAT};
static BindingNode eta{"eta", atype::FLOAT};

static BindingNode tgl{"tgl", atype::FLOAT};
static BindingNode signed1Pt{"signed1Pt", atype::FLOAT};
} // namespace nodes

namespace o2::aod::track
{
DECLARE_SOA_EXPRESSION_COLUMN(Pze, pz, float, o2::aod::track::tgl*(1.f / o2::aod::track::signed1Pt));
}

BOOST_AUTO_TEST_CASE(TestTreeParsing)
{
  expressions::Filter f = ((nodes::phi > 1) && (nodes::phi < 2)) && (nodes::eta < 1);
  auto specs = createOperations(std::move(f));
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
  auto gspecs = createOperations(std::move(g));
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
  auto hspecs = createOperations(std::move(h));

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
  auto uspecs = createOperations(std::move(u));
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

  Configurable<float> pTCut{"pTCut", 0.5f, "Lower pT limit"};
  Filter ptfilter = o2::aod::track::pt > pTCut;
  BOOST_REQUIRE_EQUAL(ptfilter.node->self.index(), 2);
  BOOST_REQUIRE_EQUAL(ptfilter.node->left->self.index(), 1);
  BOOST_REQUIRE_EQUAL(ptfilter.node->right->self.index(), 3);
  auto ptfilterspecs = createOperations(ptfilter);
  BOOST_REQUIRE_EQUAL(ptfilterspecs[0].left, (DatumSpec{std::string{"fPt"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(ptfilterspecs[0].right, (DatumSpec{LiteralNode::var_t{0.5f}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(ptfilterspecs[0].result, (DatumSpec{0u, atype::BOOL}));
}

BOOST_AUTO_TEST_CASE(TestGandivaTreeCreation)
{
  Projector pze = o2::aod::track::Pze::Projector();
  auto pzspecs = createOperations(std::move(pze));
  BOOST_REQUIRE_EQUAL(pzspecs[0].left, (DatumSpec{std::string{"fTgl"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(pzspecs[0].right, (DatumSpec{1u, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(pzspecs[0].result, (DatumSpec{0u, atype::FLOAT}));

  BOOST_REQUIRE_EQUAL(pzspecs[1].left, (DatumSpec{LiteralNode::var_t{1.f}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(pzspecs[1].right, (DatumSpec{std::string{"fSigned1Pt"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(pzspecs[1].result, (DatumSpec{1u, atype::FLOAT}));
  auto infield1 = o2::aod::track::Signed1Pt::asArrowField();
  auto infield2 = o2::aod::track::Tgl::asArrowField();
  auto resfield = o2::aod::track::Pze::asArrowField();
  auto schema = std::make_shared<arrow::Schema>(std::vector{infield1, infield2, resfield});
  auto gandiva_tree = createExpressionTree(pzspecs, schema);

  auto gandiva_expression = makeExpression(gandiva_tree, resfield);
  BOOST_CHECK_EQUAL(gandiva_expression->ToString(), "float multiply((float) fTgl, float divide((const float) 1 raw(3f800000), (float) fSigned1Pt))");
  auto projector = createProjector(schema, pzspecs, resfield);

  Projector pte = o2::aod::track::Pt::Projector();
  auto ptespecs = createOperations(std::move(pte));
  BOOST_REQUIRE_EQUAL(ptespecs[0].left, (DatumSpec{1u, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(ptespecs[0].right, (DatumSpec{}));
  BOOST_REQUIRE_EQUAL(ptespecs[0].result, (DatumSpec{0u, atype::FLOAT}));

  BOOST_REQUIRE_EQUAL(ptespecs[1].left, (DatumSpec{LiteralNode::var_t{1.f}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(ptespecs[1].right, (DatumSpec{std::string{"fSigned1Pt"}, atype::FLOAT}));
  BOOST_REQUIRE_EQUAL(ptespecs[1].result, (DatumSpec{1u, atype::FLOAT}));

  auto infield3 = o2::aod::track::Signed1Pt::asArrowField();
  auto resfield2 = o2::aod::track::Pt::asArrowField();
  auto schema2 = std::make_shared<arrow::Schema>(std::vector{infield3, resfield2});
  auto gandiva_tree2 = createExpressionTree(ptespecs, schema2);

  auto gandiva_expression2 = makeExpression(gandiva_tree2, resfield2);
  BOOST_CHECK_EQUAL(gandiva_expression2->ToString(), "float absf(float divide((const float) 1 raw(3f800000), (float) fSigned1Pt))");

  auto projector_b = createProjector(schema2, ptespecs, resfield2);
  auto schema_p = o2::soa::createSchemaFromColumns(o2::aod::Tracks::persistent_columns_t{});
  auto projector_alt = o2::framework::expressions::createProjectors(o2::framework::pack<o2::aod::track::Pt>{}, schema_p);
}
