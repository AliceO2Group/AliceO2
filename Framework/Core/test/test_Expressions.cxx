// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Configurable.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "../src/ExpressionJSONHelpers.h"
#include <catch_amalgamated.hpp>
#include <arrow/util/config.h>
#include <iostream>

using namespace o2::framework;
using namespace o2::framework::expressions;

namespace nodes
{
static BindingNode pt{"pt", 1, atype::FLOAT};
static BindingNode phi{"phi", 2, atype::FLOAT};
static BindingNode eta{"eta", 3, atype::FLOAT};

static BindingNode tgl{"tgl", 4, atype::FLOAT};
static BindingNode signed1Pt{"signed1Pt", 5, atype::FLOAT};
static BindingNode testInt{"testInt", 6, atype::INT32};
} // namespace nodes

namespace o2::aod::track
{
DECLARE_SOA_EXPRESSION_COLUMN(Pze, pz, float, o2::aod::track::tgl * (1.f / o2::aod::track::signed1Pt));
} // namespace o2::aod::track

TEST_CASE("TestTreeParsing")
{
  expressions::Filter f = ((nodes::phi > 1) && (nodes::phi < 2)) && (nodes::eta < 1);
  auto specs = createOperations(f);
  REQUIRE(specs[0].left == (DatumSpec{1u, atype::BOOL}));
  REQUIRE(specs[0].right == (DatumSpec{2u, atype::BOOL}));
  REQUIRE(specs[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(specs[1].left == (DatumSpec{std::string{"eta"}, 3, atype::FLOAT}));
  REQUIRE(specs[1].right == (DatumSpec{LiteralNode::var_t{1}, atype::INT32}));
  REQUIRE(specs[1].result == (DatumSpec{2u, atype::BOOL}));

  REQUIRE(specs[2].left == (DatumSpec{3u, atype::BOOL}));
  REQUIRE(specs[2].right == (DatumSpec{4u, atype::BOOL}));
  REQUIRE(specs[2].result == (DatumSpec{1u, atype::BOOL}));

  REQUIRE(specs[3].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(specs[3].right == (DatumSpec{LiteralNode::var_t{2}, atype::INT32}));
  REQUIRE(specs[3].result == (DatumSpec{4u, atype::BOOL}));

  REQUIRE(specs[4].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(specs[4].right == (DatumSpec{LiteralNode::var_t{1}, atype::INT32}));
  REQUIRE(specs[4].result == (DatumSpec{3u, atype::BOOL}));

  expressions::Filter g = ((nodes::eta + 2.f) > 0.5) || ((nodes::phi - M_PI) < 3);
  auto gspecs = createOperations(g);
  REQUIRE(gspecs[0].left == (DatumSpec{1u, atype::BOOL}));
  REQUIRE(gspecs[0].right == (DatumSpec{2u, atype::BOOL}));
  REQUIRE(gspecs[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(gspecs[1].left == (DatumSpec{3u, atype::DOUBLE}));
  REQUIRE(gspecs[1].right == (DatumSpec{LiteralNode::var_t{3}, atype::INT32}));
  REQUIRE(gspecs[1].result == (DatumSpec{2u, atype::BOOL}));

  REQUIRE(gspecs[2].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(gspecs[2].right == (DatumSpec{LiteralNode::var_t{M_PI}, atype::DOUBLE}));
  REQUIRE(gspecs[2].result == (DatumSpec{3u, atype::DOUBLE}));

  REQUIRE(gspecs[3].left == (DatumSpec{4u, atype::FLOAT}));
  REQUIRE(gspecs[3].right == (DatumSpec{LiteralNode::var_t{0.5}, atype::DOUBLE}));
  REQUIRE(gspecs[3].result == (DatumSpec{1u, atype::BOOL}));

  REQUIRE(gspecs[4].left == (DatumSpec{std::string{"eta"}, 3, atype::FLOAT}));
  REQUIRE(gspecs[4].right == (DatumSpec{LiteralNode::var_t{2.f}, atype::FLOAT}));
  REQUIRE(gspecs[4].result == (DatumSpec{4u, atype::FLOAT}));

  expressions::Filter h = (nodes::phi == 0) || (nodes::phi == 3);
  auto hspecs = createOperations(h);

  REQUIRE(hspecs[0].left == (DatumSpec{1u, atype::BOOL}));
  REQUIRE(hspecs[0].right == (DatumSpec{2u, atype::BOOL}));
  REQUIRE(hspecs[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(hspecs[1].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(hspecs[1].right == (DatumSpec{LiteralNode::var_t{3}, atype::INT32}));
  REQUIRE(hspecs[1].result == (DatumSpec{2u, atype::BOOL}));

  REQUIRE(hspecs[2].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(hspecs[2].right == (DatumSpec{LiteralNode::var_t{0}, atype::INT32}));
  REQUIRE(hspecs[2].result == (DatumSpec{1u, atype::BOOL}));

  expressions::Filter u = nabs(nodes::eta) < 1.0 && nexp(nodes::phi + 2.0 * M_PI) > 3.0;
  auto uspecs = createOperations(std::move(u));
  REQUIRE(uspecs[0].left == (DatumSpec{1u, atype::BOOL}));
  REQUIRE(uspecs[0].right == (DatumSpec{2u, atype::BOOL}));
  REQUIRE(uspecs[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(uspecs[1].left == (DatumSpec{3u, atype::DOUBLE}));
  REQUIRE(uspecs[1].right == (DatumSpec{LiteralNode::var_t{3.0}, atype::DOUBLE}));
  REQUIRE(uspecs[1].result == (DatumSpec{2u, atype::BOOL}));

  REQUIRE(uspecs[2].left == (DatumSpec{4u, atype::DOUBLE}));
  REQUIRE(uspecs[2].right == (DatumSpec{}));
  REQUIRE(uspecs[2].result == (DatumSpec{3u, atype::DOUBLE}));

  REQUIRE(uspecs[3].left == (DatumSpec{std::string{"phi"}, 2, atype::FLOAT}));
  REQUIRE(uspecs[3].right == (DatumSpec{LiteralNode::var_t{2.0 * M_PI}, atype::DOUBLE}));
  REQUIRE(uspecs[3].result == (DatumSpec{4u, atype::DOUBLE}));

  REQUIRE(uspecs[4].left == (DatumSpec{5u, atype::FLOAT}));
  REQUIRE(uspecs[4].right == (DatumSpec{LiteralNode::var_t{1.0}, atype::DOUBLE}));
  REQUIRE(uspecs[4].result == (DatumSpec{1u, atype::BOOL}));

  REQUIRE(uspecs[5].left == (DatumSpec{std::string{"eta"}, 3, atype::FLOAT}));
  REQUIRE(uspecs[5].right == (DatumSpec{}));
  REQUIRE(uspecs[5].result == (DatumSpec{5u, atype::FLOAT}));

  Configurable<float> pTCut{"pTCut", 0.5f, "Lower pT limit"};
  Filter ptfilter = o2::aod::track::pt > pTCut;
  REQUIRE(ptfilter.node->self.index() == 2);
  REQUIRE(ptfilter.node->left->self.index() == 1);
  REQUIRE(ptfilter.node->right->self.index() == 3);
  auto ptfilterspecs = createOperations(ptfilter);
  REQUIRE(ptfilterspecs[0].left == (DatumSpec{std::string{"fPt"}, "o2::aod::track::pt"_h, atype::FLOAT}));
  REQUIRE(ptfilterspecs[0].right == (DatumSpec{LiteralNode::var_t{0.5f}, atype::FLOAT}));
  REQUIRE(ptfilterspecs[0].result == (DatumSpec{0u, atype::BOOL}));

  struct : ConfigurableGroup {
    std::string prefix = "prefix";
    Configurable<float> pTCut{"pTCut", 1.0f, "Lower pT limit"};
  } group;
  Filter ptfilter2 = o2::aod::track::pt > group.pTCut;
  group.pTCut.name.insert(0, 1, '.');
  group.pTCut.name.insert(0, group.prefix);
  REQUIRE(ptfilter2.node->self.index() == 2);
  REQUIRE(ptfilter2.node->left->self.index() == 1);
  REQUIRE(ptfilter2.node->right->self.index() == 3);
  REQUIRE(std::get<PlaceholderNode>(ptfilter2.node->right->self).name == "prefix.pTCut");
  auto ptfilterspecs2 = createOperations(ptfilter2);
  REQUIRE(ptfilterspecs2[0].left == (DatumSpec{std::string{"fPt"}, "o2::aod::track::pt"_h, atype::FLOAT}));
  REQUIRE(ptfilterspecs2[0].right == (DatumSpec{LiteralNode::var_t{1.0f}, atype::FLOAT}));
  REQUIRE(ptfilterspecs2[0].result == (DatumSpec{0u, atype::BOOL}));

  Configurable<int> cvalue{"cvalue", 1, "test value"};
  Filter testFilter = o2::aod::track::tpcNClsShared < as<uint8_t>(cvalue);
  REQUIRE(testFilter.node->self.index() == 2);
  REQUIRE(testFilter.node->left->self.index() == 1);
  REQUIRE(testFilter.node->right->self.index() == 3);
  REQUIRE(std::get<PlaceholderNode>(testFilter.node->right->self).name == "cvalue");
  auto testSpecs = createOperations(testFilter);
  REQUIRE(testSpecs[0].right == (DatumSpec{LiteralNode::var_t{(uint8_t)1}, atype::UINT8}));
}

TEST_CASE("TestGandivaTreeCreation")
{
  Projector pze = o2::aod::track::Pze::Projector();
  auto pzspecs = createOperations(pze);
  REQUIRE(pzspecs[0].left == (DatumSpec{std::string{"fTgl"}, "o2::aod::track::tgl"_h, atype::FLOAT}));
  REQUIRE(pzspecs[0].right == (DatumSpec{1u, atype::FLOAT}));
  REQUIRE(pzspecs[0].result == (DatumSpec{0u, atype::FLOAT}));

  REQUIRE(pzspecs[1].left == (DatumSpec{LiteralNode::var_t{1.f}, atype::FLOAT}));
  REQUIRE(pzspecs[1].right == (DatumSpec{std::string{"fSigned1Pt"}, "o2::aod::track::signed1Pt"_h, atype::FLOAT}));
  REQUIRE(pzspecs[1].result == (DatumSpec{1u, atype::FLOAT}));
  auto infield1 = o2::aod::track::Signed1Pt::asArrowField();
  auto infield2 = o2::aod::track::Tgl::asArrowField();
  auto resfield = o2::aod::track::Pze::asArrowField();
  auto schema = std::make_shared<arrow::Schema>(std::vector{infield1, infield2, resfield});
  auto gandiva_tree = createExpressionTree(pzspecs, schema);

  auto gandiva_expression = makeExpression(gandiva_tree, resfield);
  REQUIRE(std::string(gandiva_expression->ToString()) == std::string("float multiply((float) fTgl, float divide((const float) 1 raw(3f800000), (float) fSigned1Pt))"));
  auto projector = createProjector(schema, pzspecs, resfield);

  Projector pte = o2::aod::track::Pt::Projector();
  auto ptespecs = createOperations(pte);

  auto infield3 = o2::aod::track::Signed1Pt::asArrowField();
  auto resfield2 = o2::aod::track::Pt::asArrowField();
  auto schema2 = std::make_shared<arrow::Schema>(std::vector{infield3, resfield2});
  auto gandiva_tree2 = createExpressionTree(ptespecs, schema2);

  auto gandiva_expression2 = makeExpression(gandiva_tree2, resfield2);
  REQUIRE(gandiva_expression2->ToString() == "if (bool less_than_or_equal_to(float absf((float) fSigned1Pt), (const float) 1.17549e-38 raw(800000))) { (const float) 8.50706e+37 raw(7e800000) } else { float absf(float divide((const float) 1 raw(3f800000), (float) fSigned1Pt)) }");

  auto projector_b = createProjector(schema2, ptespecs, resfield2);
  auto fields = o2::soa::createFieldsFromColumns(o2::aod::Tracks::persistent_columns_t{});
  auto schema_p = std::make_shared<arrow::Schema>(fields);
  auto projector_alt = o2::framework::expressions::createProjectors(o2::framework::pack<o2::aod::track::Pt>{}, {resfield2}, schema_p);

  Filter bitwiseFilter = (o2::aod::track::flags & static_cast<uint32_t>(o2::aod::track::TPCrefit)) != 0u;
  auto bwf = createOperations(bitwiseFilter);
  REQUIRE(bwf[0].left == (DatumSpec{1u, atype::UINT32}));
  REQUIRE(bwf[0].right == (DatumSpec{LiteralNode::var_t{0u}, atype::UINT32}));
  REQUIRE(bwf[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(bwf[1].left == (DatumSpec{std::string{"fFlags"}, "o2::aod::track::flags"_h, atype::UINT32}));
  REQUIRE(bwf[1].right == (DatumSpec{LiteralNode::var_t{static_cast<uint32_t>(o2::aod::track::TPCrefit)}, atype::UINT32}));
  REQUIRE(bwf[1].result == (DatumSpec{1u, atype::UINT32}));

  auto infield4 = o2::aod::track::Flags::asArrowField();
  auto resfield3 = std::make_shared<arrow::Field>("out", arrow::boolean());
  auto schema_b = std::make_shared<arrow::Schema>(std::vector{infield4, resfield3});
  auto gandiva_tree3 = createExpressionTree(bwf, schema_b);
  REQUIRE(gandiva_tree3->ToString() == "bool not_equal(uint32 bitwise_and((uint32) fFlags, (const uint32) 2), (const uint32) 0)");
  auto condition = expressions::makeCondition(gandiva_tree3);
  std::shared_ptr<gandiva::Filter> flt;
  auto s = gandiva::Filter::Make(schema_b, condition, &flt);
  REQUIRE(s.ok());

  Filter rounding = nround(o2::aod::track::pt) > 0.1f;
  auto rf = createOperations(rounding);
  REQUIRE(rf[0].left == (DatumSpec{1u, atype::FLOAT}));
  REQUIRE(rf[0].right == (DatumSpec{LiteralNode::var_t{0.1f}, atype::FLOAT}));
  REQUIRE(rf[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(rf[1].left == (DatumSpec{std::string{"fPt"}, "o2::aod::track::pt"_h, atype::FLOAT}));
  REQUIRE(rf[1].right == (DatumSpec{}));
  REQUIRE(rf[1].result == (DatumSpec{1u, atype::FLOAT}));

  auto infield5 = o2::aod::track::Pt::asArrowField();
  auto resfield4 = std::make_shared<arrow::Field>("out", arrow::boolean());
  auto schema_c = std::make_shared<arrow::Schema>(std::vector{infield5, resfield4});
  auto gandiva_tree4 = createExpressionTree(rf, schema_c);
  REQUIRE(gandiva_tree4->ToString() == "bool greater_than(float round((float) fPt), (const float) 0.1 raw(3dcccccd))");
  auto condition2 = expressions::makeCondition(gandiva_tree4);
  std::shared_ptr<gandiva::Filter> flt2;
  auto s2 = gandiva::Filter::Make(schema_c, condition2, &flt2);
  REQUIRE(s2.ok());
}

TEST_CASE("TestConditionalExpressions")
{
  // simple conditional
  Filter cf = nabs(o2::aod::track::eta) < 1.0f && ifnode((o2::aod::track::pt < 1.0f), (o2::aod::track::phi > (float)(M_PI / 2.)), (o2::aod::track::phi < (float)(M_PI / 2.)));
  auto cfspecs = createOperations(cf);
  REQUIRE(cfspecs[0].left == (DatumSpec{1u, atype::BOOL}));
  REQUIRE(cfspecs[0].right == (DatumSpec{2u, atype::BOOL}));
  REQUIRE(cfspecs[0].result == (DatumSpec{0u, atype::BOOL}));

  REQUIRE(cfspecs[1].left == (DatumSpec{3u, atype::BOOL}));
  REQUIRE(cfspecs[1].right == (DatumSpec{4u, atype::BOOL}));
  REQUIRE(cfspecs[1].condition == (DatumSpec{5u, atype::BOOL}));
  REQUIRE(cfspecs[1].result == (DatumSpec{2u, atype::BOOL}));

  REQUIRE(cfspecs[2].left == (DatumSpec{std::string{"fPt"}, "o2::aod::track::pt"_h, atype::FLOAT}));
  REQUIRE(cfspecs[2].right == (DatumSpec{LiteralNode::var_t{1.0f}, atype::FLOAT}));
  REQUIRE(cfspecs[2].result == (DatumSpec{5u, atype::BOOL}));

  REQUIRE(cfspecs[3].left == (DatumSpec{std::string{"fPhi"}, "o2::aod::track::phi"_h, atype::FLOAT}));
  REQUIRE(cfspecs[3].right == (DatumSpec{LiteralNode::var_t{(float)(M_PI / 2.)}, atype::FLOAT}));
  REQUIRE(cfspecs[3].result == (DatumSpec{4u, atype::BOOL}));

  REQUIRE(cfspecs[4].left == (DatumSpec{std::string{"fPhi"}, "o2::aod::track::phi"_h, atype::FLOAT}));
  REQUIRE(cfspecs[4].right == (DatumSpec{LiteralNode::var_t{(float)(M_PI / 2.)}, atype::FLOAT}));
  REQUIRE(cfspecs[4].result == (DatumSpec{3u, atype::BOOL}));

  REQUIRE(cfspecs[5].left == (DatumSpec{6u, atype::FLOAT}));
  REQUIRE(cfspecs[5].right == (DatumSpec{LiteralNode::var_t{1.0f}, atype::FLOAT}));
  REQUIRE(cfspecs[5].result == (DatumSpec{1u, atype::BOOL}));

  REQUIRE(cfspecs[6].left == (DatumSpec{std::string{"fEta"}, "o2::aod::track::eta"_h, atype::FLOAT}));
  REQUIRE(cfspecs[6].right == (DatumSpec{}));
  REQUIRE(cfspecs[6].result == (DatumSpec{6u, atype::FLOAT}));

  auto infield1 = o2::aod::track::Pt::asArrowField();
  auto infield2 = o2::aod::track::Eta::asArrowField();
  auto infield3 = o2::aod::track::Phi::asArrowField();
  auto schema = std::make_shared<arrow::Schema>(std::vector{infield1, infield2, infield3});
  auto gandiva_tree = createExpressionTree(cfspecs, schema);
  auto gandiva_condition = makeCondition(gandiva_tree);
  auto gandiva_filter = createFilter(schema, gandiva_condition);

  REQUIRE(gandiva_tree->ToString() == "bool less_than(float absf((float) fEta), (const float) 1 raw(3f800000)) && if (bool less_than((float) fPt, (const float) 1 raw(3f800000))) { bool greater_than((float) fPhi, (const float) 1.5708 raw(3fc90fdb)) } else { bool less_than((float) fPhi, (const float) 1.5708 raw(3fc90fdb)) }");

  // nested conditional
  Filter cfn = o2::aod::track::signed1Pt > 0.f && ifnode(std::move(*cf.node), nabs(o2::aod::track::x) > 1.0f, nabs(o2::aod::track::y) > 1.0f);
  auto cfnspecs = createOperations(cfn);
  auto infield4 = o2::aod::track::Signed1Pt::asArrowField();
  auto infield5 = o2::aod::track::X::asArrowField();
  auto infield6 = o2::aod::track::Y::asArrowField();
  auto schema2 = std::make_shared<arrow::Schema>(std::vector{infield1, infield2, infield3, infield4, infield5, infield6});
  auto gandiva_tree2 = createExpressionTree(cfnspecs, schema2);
  auto gandiva_condition2 = makeCondition(gandiva_tree2);
  auto gandiva_filter2 = createFilter(schema2, gandiva_condition2);
  REQUIRE(gandiva_tree2->ToString() == "bool greater_than((float) fSigned1Pt, (const float) 0 raw(0)) && if (bool less_than(float absf((float) fEta), (const float) 1 raw(3f800000)) && if (bool less_than((float) fPt, (const float) 1 raw(3f800000))) { bool greater_than((float) fPhi, (const float) 1.5708 raw(3fc90fdb)) } else { bool less_than((float) fPhi, (const float) 1.5708 raw(3fc90fdb)) }) { bool greater_than(float absf((float) fX), (const float) 1 raw(3f800000)) } else { bool greater_than(float absf((float) fY), (const float) 1 raw(3f800000)) }");

  // clamp
  Projector clp = clamp(o2::aod::track::pt, 1.0f, 10.f);
  auto clpspecs = createOperations(clp);
  auto schemaclp = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Pt::asArrowField()});
  auto gandiva_tree_clp = createExpressionTree(clpspecs, schemaclp);
  REQUIRE(gandiva_tree_clp->ToString() == "if (bool less_than((float) fPt, (const float) 1 raw(3f800000))) { (const float) 1 raw(3f800000) } else { if (bool greater_than((float) fPt, (const float) 10 raw(41200000))) { (const float) 10 raw(41200000) } else { (float) fPt } }");
}

TEST_CASE("TestBinnedExpressions")
{
  std::vector<float> bins{0.5, 1.5, 2.5, 3.5, 4.5};
  std::vector<float> params{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3, 4.0, 4.1, 4.2, 4.3};
  Projector p = binned(bins, params, o2::aod::track::pt, par(0) * o2::aod::track::x + par(1) * o2::aod::track::y + par(2) * o2::aod::track::z + par(3) * o2::aod::track::phi, LiteralNode{0.f});
  auto pspecs = createOperations(p);
  auto schema = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Pt::asArrowField(), o2::aod::track::X::asArrowField(), o2::aod::track::Y::asArrowField(), o2::aod::track::Z::asArrowField(), o2::aod::track::Phi::asArrowField()});
  auto tree = createExpressionTree(pspecs, schema);
  REQUIRE(tree->ToString() == "if (bool less_than((float) fPt, (const float) 0.5 raw(3f000000))) { (const float) 0 raw(0) } else { if (bool less_than((float) fPt, (const float) 1.5 raw(3fc00000))) { float add(float add(float add(float multiply((const float) 1 raw(3f800000), (float) fX), float multiply((const float) 2 raw(40000000), (float) fY)), float multiply((const float) 3 raw(40400000), (float) fZ)), float multiply((const float) 4 raw(40800000), (float) fPhi)) } else { if (bool less_than((float) fPt, (const float) 2.5 raw(40200000))) { float add(float add(float add(float multiply((const float) 1.1 raw(3f8ccccd), (float) fX), float multiply((const float) 2.1 raw(40066666), (float) fY)), float multiply((const float) 3.1 raw(40466666), (float) fZ)), float multiply((const float) 4.1 raw(40833333), (float) fPhi)) } else { if (bool less_than((float) fPt, (const float) 3.5 raw(40600000))) { float add(float add(float add(float multiply((const float) 1.2 raw(3f99999a), (float) fX), float multiply((const float) 2.2 raw(400ccccd), (float) fY)), float multiply((const float) 3.2 raw(404ccccd), (float) fZ)), float multiply((const float) 4.2 raw(40866666), (float) fPhi)) } else { if (bool less_than((float) fPt, (const float) 4.5 raw(40900000))) { float add(float add(float add(float multiply((const float) 1.3 raw(3fa66666), (float) fX), float multiply((const float) 2.3 raw(40133333), (float) fY)), float multiply((const float) 3.3 raw(40533333), (float) fZ)), float multiply((const float) 4.3 raw(4089999a), (float) fPhi)) } else { (const float) 0 raw(0) } } } } }");

  std::vector<float> binning{0, o2::constants::math::PIHalf, o2::constants::math::PI, o2::constants::math::PI + o2::constants::math::PIHalf, o2::constants::math::TwoPI};
  std::vector<float> parameters{1.0, 1.1, 1.2, 1.3,  // par 0
                                2.0, 2.1, 2.2, 2.3,  // par 1
                                3.0, 3.1, 3.2, 3.3,  // par 2
                                4.0, 4.1, 4.2, 4.3}; // par 3

  Projector p2 = binned((std::vector<float>)binning,
                        (std::vector<float>)parameters,
                        o2::aod::track::phi, par(0) * o2::aod::track::x * o2::aod::track::x + par(1) * o2::aod::track::y * o2::aod::track::y + par(2) * o2::aod::track::z * o2::aod::track::z,
                        LiteralNode{-1.f});
  auto p2specs = createOperations(p2);
  auto schema2 = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Phi::asArrowField(), o2::aod::track::X::asArrowField(), o2::aod::track::Y::asArrowField(), o2::aod::track::Z::asArrowField()});
  auto tree2 = createExpressionTree(p2specs, schema2);
  REQUIRE(tree2->ToString() == "if (bool less_than((float) fPhi, (const float) 0 raw(0))) { (const float) -1 raw(bf800000) } else { if (bool less_than((float) fPhi, (const float) 1.5708 raw(3fc90fdb))) { float add(float add(float multiply(float multiply((const float) 1 raw(3f800000), (float) fX), (float) fX), float multiply(float multiply((const float) 2 raw(40000000), (float) fY), (float) fY)), float multiply(float multiply((const float) 3 raw(40400000), (float) fZ), (float) fZ)) } else { if (bool less_than((float) fPhi, (const float) 3.14159 raw(40490fdb))) { float add(float add(float multiply(float multiply((const float) 1.1 raw(3f8ccccd), (float) fX), (float) fX), float multiply(float multiply((const float) 2.1 raw(40066666), (float) fY), (float) fY)), float multiply(float multiply((const float) 3.1 raw(40466666), (float) fZ), (float) fZ)) } else { if (bool less_than((float) fPhi, (const float) 4.71239 raw(4096cbe4))) { float add(float add(float multiply(float multiply((const float) 1.2 raw(3f99999a), (float) fX), (float) fX), float multiply(float multiply((const float) 2.2 raw(400ccccd), (float) fY), (float) fY)), float multiply(float multiply((const float) 3.2 raw(404ccccd), (float) fZ), (float) fZ)) } else { if (bool less_than((float) fPhi, (const float) 6.28319 raw(40c90fdb))) { float add(float add(float multiply(float multiply((const float) 1.3 raw(3fa66666), (float) fX), (float) fX), float multiply(float multiply((const float) 2.3 raw(40133333), (float) fY), (float) fY)), float multiply(float multiply((const float) 3.3 raw(40533333), (float) fZ), (float) fZ)) } else { (const float) -1 raw(bf800000) } } } } }");
}

void printTokens(Tokenizer& t)
{
  int token;
  while ((token = t.nextToken()) && (token != Token::EoL)) {
    std::cout << t.TokenStr << " ";
  };
  std::cout << std::endl;
}

TEST_CASE("TestStringExpressionsParsing")
{
  Filter f = (o2::aod::track::flags & 1u) != 0u && (o2::aod::track::pt <= 10.f);
  std::string input = "(o2::aod::track::flags & 1u) != 0u && (o2::aod::track::pt <= 10.f)";

  auto t1 = createOperations(f);
  Filter ff = Parser::parse(input);
  auto t2 = createOperations(ff);

  auto schema = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Flags::asArrowField(), o2::aod::track::Pt::asArrowField()});
  auto tree1 = createExpressionTree(t1, schema);
  auto tree2 = createExpressionTree(t2, schema);

  REQUIRE(tree1->ToString() == tree2->ToString());

  Projector p = -1.f * nlog(ntan(o2::constants::math::PIQuarter - 0.5f * natan(o2::aod::fwdtrack::tgl)));
  input = "-1.f * nlog(ntan(PIQuarter - 0.5f * natan(o2::aod::fwdtrack::tgl)))";

  auto tp1 = createOperations(p);
  Projector pp = Parser::parse(input);
  auto tp2 = createOperations(pp);

  schema = std::make_shared<arrow::Schema>(std::vector{o2::aod::fwdtrack::Tgl::asArrowField()});
  auto treep1 = createExpressionTree(tp1, schema);
  auto treep2 = createExpressionTree(tp2, schema);

  REQUIRE(treep1->ToString() == treep2->ToString());

  Filter f2 = o2::aod::track::signed1Pt > 0.f && ifnode(nabs(o2::aod::track::eta) < 1.0f, nabs(o2::aod::track::x) > 2.0f, nabs(o2::aod::track::y) > 3.0f);
  input = "o2::aod::track::signed1Pt > 0.f && ifnode(nabs(o2::aod::track::eta) < 1.0f, nabs(o2::aod::track::x) > 2.0f, nabs(o2::aod::track::y) > 3.0f)";

  auto tf1 = createOperations(f2);
  Filter ff2 = Parser::parse(input);
  auto tf2 = createOperations(ff2);

  schema = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Eta::asArrowField(), o2::aod::track::Signed1Pt::asArrowField(), o2::aod::track::X::asArrowField(), o2::aod::track::Y::asArrowField()});
  auto treef1 = createExpressionTree(tf1, schema);
  auto treef2 = createExpressionTree(tf2, schema);

  REQUIRE(treef1->ToString() == treef2->ToString());

  Configurable<float> pTCut{"pTCut", 0.5f, "Lower pT limit"};
  Filter pcfg1 = o2::aod::track::pt > pTCut;
  Filter pcfg2 = Parser::parse("o2::aod::track::pt > ncfg(float, 0.5, \"pTCut\")");
  auto pcfg1specs = createOperations(pcfg1);
  auto pcfg2specs = createOperations(pcfg2);

  REQUIRE(pcfg2.node->right->self.index() == 3);
  REQUIRE(pcfg2specs[0].right == (DatumSpec{LiteralNode::var_t{0.5f}, atype::FLOAT}));

  schema = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Pt::asArrowField()});
  auto tree1c = createExpressionTree(pcfg1specs, schema);
  auto tree2c = createExpressionTree(pcfg2specs, schema);

  REQUIRE(tree1c->ToString() == tree2c->ToString());
}

TEST_CASE("TestExpressionSerialization")
{
  Filter f = o2::aod::track::signed1Pt > 0.f && ifnode(nabs(o2::aod::track::eta) < 1.0f, nabs(o2::aod::track::x) > 2.0f, nabs(o2::aod::track::y) > 3.0f);
  Projector p = -1.f * nlog(ntan(o2::constants::math::PIQuarter - 0.5f * natan(o2::aod::fwdtrack::tgl)));
  Projector p1 = ifnode(o2::aod::track::itsClusterSizes > (uint32_t)0, static_cast<uint8_t>(o2::aod::track::ITS), (uint8_t)0x0) |
                 ifnode(o2::aod::track::tpcNClsFindable > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TPC), (uint8_t)0x0) |
                 ifnode(o2::aod::track::trdPattern > (uint8_t)0, static_cast<uint8_t>(o2::aod::track::TRD), (uint8_t)0x0) |
                 ifnode((o2::aod::track::tofChi2 >= 0.f) && (o2::aod::track::tofExpMom > 0.f), static_cast<uint8_t>(o2::aod::track::TOF), (uint8_t)0x0);

  std::vector<Projector> projectors;
  projectors.emplace_back(std::move(f));
  projectors.emplace_back(std::move(p));
  projectors.emplace_back(std::move(p1));

  std::stringstream osm;
  ExpressionJSONHelpers::write(osm, projectors);

  std::stringstream ism;
  ism.str(osm.str());
  auto ps = ExpressionJSONHelpers::read(ism);

  auto s1 = createOperations(projectors[0]);
  auto s2 = createOperations(ps[0]);
  auto schemaf = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::Eta::asArrowField(), o2::aod::track::Signed1Pt::asArrowField(), o2::aod::track::X::asArrowField(), o2::aod::track::Y::asArrowField()});
  auto t1 = createExpressionTree(s1, schemaf);
  auto t2 = createExpressionTree(s2, schemaf);
  REQUIRE(t1->ToString() == t2->ToString());

  auto s12 = createOperations(projectors[1]);
  auto s22 = createOperations(ps[1]);
  auto schemap = std::make_shared<arrow::Schema>(std::vector{o2::aod::fwdtrack::Tgl::asArrowField()});
  auto t12 = createExpressionTree(s12, schemap);
  auto t22 = createExpressionTree(s22, schemap);
  REQUIRE(t12->ToString() == t22->ToString());

  auto s13 = createOperations(projectors[2]);
  auto s23 = createOperations(ps[2]);
  auto schemap1 = std::make_shared<arrow::Schema>(std::vector{o2::aod::track::ITSClusterSizes::asArrowField(), o2::aod::track::TPCNClsFindable::asArrowField(),
                                                              o2::aod::track::TRDPattern::asArrowField(), o2::aod::track::TOFChi2::asArrowField(),
                                                              o2::aod::track::TOFExpMom::asArrowField()});
  auto t13 = createExpressionTree(s13, schemap1);
  auto t23 = createExpressionTree(s23, schemap1);
  REQUIRE(t13->ToString() == t23->ToString());

  osm.clear();
  osm.str("");
  ArrowJSONHelpers::write(osm, schemaf);

  ism.clear();
  ism.str(osm.str());
  auto newSchemaf = ArrowJSONHelpers::read(ism);
  REQUIRE(schemaf->ToString() == newSchemaf->ToString());

  osm.clear();
  osm.str("");
  ArrowJSONHelpers::write(osm, schemap);

  ism.clear();
  ism.str(osm.str());
  auto newSchemap = ArrowJSONHelpers::read(ism);
  REQUIRE(schemap->ToString() == newSchemap->ToString());
}
