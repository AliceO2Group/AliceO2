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
/// @author  Laurent Aphecetche

#define BOOST_TEST_MODULE Test MCHContour SegmentTree
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "../include/MCHContour/SegmentTree.h"

using namespace o2::mch::contour::impl;

struct YPOS {

  YPOS()
  {

    auto* left = new Node<int>{Interval<int>{0, 4}, 2};
    auto* right = new Node<int>{Interval<int>{4, 8}, 6};

    testNode.setLeft(left).setRight(right);

    left->setCardinality(dummyCardinality);
    right->setCardinality(dummyCardinality);
  }

  std::vector<int> yposInt{0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> yposDouble{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  Node<int> node{Interval<int>{0, 8}, 4};
  Node<int> testNode{Interval<int>{0, 8}, 4};
  int dummyCardinality{3};
};

BOOST_AUTO_TEST_SUITE(o2_mch_contour)

BOOST_FIXTURE_TEST_SUITE(segmenttree, YPOS)

BOOST_AUTO_TEST_CASE(NeedAtLeastTwoValuesToBuildASegmentTree)
{
  std::vector<int> onlyOneElement{0};
  BOOST_CHECK_THROW(createSegmentTree(onlyOneElement), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(NodeInsertAndDeleteIntVersion)
{
  std::unique_ptr<Node<int>> t{createSegmentTree(yposInt)};

  t->insertInterval(Interval<int>{1, 5});
  t->insertInterval(Interval<int>{5, 8});
  t->deleteInterval(Interval<int>{6, 7});

  std::ostringstream os;

  os << '\n'
     << (*t);

  std::string expectedOutput =
    R"(
[0,8] potent
     [0,4] potent
           [0,2] potent
                 [0,1]
                 [1,2] C=1
           [2,4] C=1
                 [2,3]
                 [3,4]
     [4,8] potent
           [4,6] C=1
                 [4,5]
                 [5,6]
           [6,8] potent
                 [6,7]
                 [7,8] C=1
)";

  BOOST_CHECK_EQUAL(os.str(), expectedOutput);
}

BOOST_AUTO_TEST_CASE(NodeInsertAndDeleteDoubleVersion)
{
  std::unique_ptr<Node<double>> t{createSegmentTree(yposDouble)};

  t->insertInterval(Interval<double>{0.1, 0.5});
  t->insertInterval(Interval<double>{0.5, 0.8});
  t->deleteInterval(Interval<double>{0.6, 0.7});

  std::ostringstream os;

  os << '\n'
     << (*t);

  std::string expectedOutput =
    R"(
[0,0.8] potent
     [0,0.4] potent
           [0,0.2] potent
                 [0,0.1]
                 [0.1,0.2] C=1
           [0.2,0.4] C=1
                 [0.2,0.3]
                 [0.3,0.4]
     [0.4,0.8] potent
           [0.4,0.6] C=1
                 [0.4,0.5]
                 [0.5,0.6]
           [0.6,0.8] potent
                 [0.6,0.7]
                 [0.7,0.8] C=1
)";

  BOOST_CHECK_EQUAL(os.str(), expectedOutput);
}

BOOST_AUTO_TEST_CASE(JustCreatedNodeIsNotPotent) { BOOST_CHECK_EQUAL(node.isPotent(), false); }

BOOST_AUTO_TEST_CASE(JustCreatedNodeHasCardinalityEqualsZero) { BOOST_CHECK_EQUAL(node.cardinality(), 0); }

BOOST_AUTO_TEST_CASE(PromoteNode)
{
  testNode.promote();

  BOOST_CHECK_EQUAL(testNode.cardinality(), 1);
  BOOST_CHECK_EQUAL(testNode.left()->cardinality(), dummyCardinality - 1);
  BOOST_CHECK_EQUAL(testNode.right()->cardinality(), dummyCardinality - 1);
}

BOOST_AUTO_TEST_CASE(DemoteNode)
{
  testNode.promote();
  testNode.demote();
  BOOST_CHECK_EQUAL(testNode.cardinality(), 0);
  BOOST_CHECK_EQUAL(testNode.left()->cardinality(), dummyCardinality);
  BOOST_CHECK_EQUAL(testNode.right()->cardinality(), dummyCardinality);
  BOOST_CHECK_EQUAL(testNode.isPotent(), true);
}

BOOST_AUTO_TEST_CASE(MidPointOfANodeIsNotHalfPoint)
{
  std::vector<double> ypos{-2.0, -1.5, -1, 0};
  std::unique_ptr<Node<double>> root{createSegmentTree(ypos)};
  auto right = root->right();
  BOOST_CHECK_EQUAL(right->interval(), Interval<double>(-1.5, 0));
  BOOST_CHECK(right->midpoint() != 1.5 / 2);
  BOOST_CHECK_EQUAL(right->midpoint(), -1);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
