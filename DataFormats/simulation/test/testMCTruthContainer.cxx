// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCTruthContainer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include <algorithm>
#include <iostream>

namespace o2
{
BOOST_AUTO_TEST_CASE(MCTruth)
{
  using TruthElement = long;
  dataformats::MCTruthContainer<TruthElement> container;
  container.addElement(0, TruthElement(1));
  container.addElement(0, TruthElement(2));
  container.addElement(1, TruthElement(1));
  container.addElement(2, TruthElement(10));

  // this is not possible: (how to test for it)
  // container.addElement(0,TruthElement(0));

  // check header/index information
  BOOST_CHECK(container.getMCTruthHeader(0).index == 0);
  BOOST_CHECK(container.getMCTruthHeader(1).index == 2);
  BOOST_CHECK(container.getMCTruthHeader(2).index == 3);

  // check MC truth information
  BOOST_CHECK(container.getElement(0) == 1);
  BOOST_CHECK(container.getElement(1) == 2);
  BOOST_CHECK(container.getElement(2) == 1);
  BOOST_CHECK(container.getElement(3) == 10);

  // get iterable container view on labels for index 0
  auto view = container.getLabels(0);
  BOOST_CHECK(view.size() == 2);
  BOOST_CHECK(view[0] == 1);
  BOOST_CHECK(view[1] == 2);
  // try to sort the view
  std::sort(view.begin(), view.end(), [](TruthElement a, TruthElement b) { return a > b; });
  BOOST_CHECK(view[0] == 2);
  BOOST_CHECK(view[1] == 1);
  // verify sorting took effect inside container
  auto view2 = container.getLabels(0);
  BOOST_CHECK(view2[0] == 2);
  BOOST_CHECK(view2[1] == 1);

  // same for another data index
  view = container.getLabels(2);
  BOOST_CHECK(view.size() == 1);
  BOOST_CHECK(view[0] == 10);

  // try to get something invalid
  view = container.getLabels(10);
  BOOST_CHECK(view.size() == 0);

  // test assignment/copy
  auto copy = container;
  view = copy.getLabels(2);
  BOOST_CHECK(view.size() == 1);
  BOOST_CHECK(view[0] == 10);

  // add multiple labels
  std::vector<TruthElement> newlabels = {101, 102, 103};
  container.addElements(2, newlabels);
  view = container.getLabels(2);
  BOOST_CHECK(view.size() == 4);
  BOOST_CHECK(view[0] == 10);
  BOOST_CHECK(view[1] == 101);
  BOOST_CHECK(view[2] == 102);
  BOOST_CHECK(view[3] == 103);

  // test merging
  {
    dataformats::MCTruthContainer<TruthElement> container1;
    container1.addElement(0, TruthElement(1));
    container1.addElement(0, TruthElement(2));
    container1.addElement(1, TruthElement(1));
    container1.addElement(2, TruthElement(10));

    dataformats::MCTruthContainer<TruthElement> container2;
    container2.addElement(0, TruthElement(11));
    container2.addElement(0, TruthElement(12));
    container2.addElement(1, TruthElement(1));
    container2.addElement(2, TruthElement(10));

    container1.mergeAtBack(container2);
    auto lview = container1.getLabels(3); //
    BOOST_CHECK(lview.size() == 2);
    BOOST_CHECK(lview[0] == 11);
    BOOST_CHECK(lview[1] == 12);
    BOOST_CHECK(container1.getIndexedSize() == 6);
    BOOST_CHECK(container1.getNElements() == 8);
  }
}

BOOST_AUTO_TEST_CASE(MCTruth_RandomAccess)
{
  using TruthElement = long;
  dataformats::MCTruthContainer<TruthElement> container;
  container.addElementRandomAccess(0, TruthElement(1));
  container.addElementRandomAccess(0, TruthElement(2));
  container.addElementRandomAccess(1, TruthElement(1));
  container.addElementRandomAccess(2, TruthElement(10));
  container.addElementRandomAccess(1, TruthElement(5));
  container.addElementRandomAccess(0, TruthElement(5));
  // add element at end
  container.addElement(3, TruthElement(20));
  container.addElement(3, TruthElement(21));

  // check header/index information
  BOOST_CHECK(container.getMCTruthHeader(0).index == 0);
  BOOST_CHECK(container.getMCTruthHeader(1).index == 3);
  BOOST_CHECK(container.getMCTruthHeader(2).index == 5);
  BOOST_CHECK(container.getMCTruthHeader(3).index == 6);

  // get iterable container view on labels
  {
    auto view = container.getLabels(1);
    BOOST_CHECK(view.size() == 2);
    BOOST_CHECK(view[0] == 1);
    BOOST_CHECK(view[1] == 5);
  }

  {
    auto view = container.getLabels(3);
    BOOST_CHECK(view.size() == 2);
    BOOST_CHECK(view[0] == 20);
    BOOST_CHECK(view[1] == 21);
  }
}

BOOST_AUTO_TEST_CASE(LabelContainer_noncont)
{
  using TruthElement = long;
  // creates a container where labels might not be contiguous
  dataformats::LabelContainer<TruthElement, false> container;
  container.addLabel(0, TruthElement(1));
  container.addLabel(1, TruthElement(1));
  container.addLabel(0, TruthElement(10));
  container.addLabel(2, TruthElement(20));

  auto view = container.getLabels(0);
  BOOST_CHECK(view.size() == 2);
  for (auto& e : view) {
    std::cerr << e << "\n";
  }

  {
    auto view2 = container.getLabels(10);
    BOOST_CHECK(view2.size() == 0);
    for (auto& e : view2) {
      // should not come here
      BOOST_CHECK(false);
    }
  }

  {
    auto view2 = container.getLabels(2);
    BOOST_CHECK(view2.size() == 1);
    std::cout << "ELEMENTS OF LABEL 2\n";
    for (auto& e : view2) {
      // should not come here
      std::cout << e << "\n";
    }
    std::cout << "------\n";
  }

  std::vector<TruthElement> v;
  container.fillVectorOfLabels(0, v);
  BOOST_CHECK(v.size() == 2);
  std::sort(v.begin(), v.end(), [](TruthElement a, TruthElement b) { return a > b; });
  for (auto& e : v) {
    std::cerr << e << "\n";
  }

  const int R = 3;
  const int L = 5;
  // test with more elements
  dataformats::LabelContainer<TruthElement, false> container2;
  for (int run = 0; run < R; ++run) {
    for (int i = 0; i < L; ++i) {
      container2.addLabel(i, TruthElement(run));
    }
  }
  // testing stage
  for (int i = 0; i < L; ++i) {
    auto labelview = container2.getLabels(i);
    BOOST_CHECK(labelview.size() == R);
    // count elements when iterating over view
    int counter = 0;
    std::cout << "CHECK CONTENT FOR INDEX " << i << "\n";
    std::cout << "----- \n";
    for (auto& l : labelview) {
      counter++;
      std::cout << l << "\n";
    }
    std::cout << "#### " << i << "\n";
    std::cout << counter << "\n";
    BOOST_CHECK(labelview.size() == counter);
  }

  // in this case we have to add the elements contiguously per dataindex:
  dataformats::LabelContainer<TruthElement, true> cont2;
  cont2.addLabel(0, TruthElement(1));
  cont2.addLabel(0, TruthElement(10));
  cont2.addLabel(1, TruthElement(1));
  cont2.addLabel(2, TruthElement(20));
  {
    auto view2 = cont2.getLabels(0);
    BOOST_CHECK(view2.size() == 2);
    for (auto& e : view2) {
      std::cerr << e << "\n";
    }
  }
  {
    auto view2 = cont2.getLabels(1);
    BOOST_CHECK(view2.size() == 1);
    for (auto& e : view2) {
      std::cerr << e << "\n";
    }
  }
  {
    auto view2 = cont2.getLabels(2);
    BOOST_CHECK(view2.size() == 1);
    for (auto& e : view2) {
      std::cerr << e << "\n";
    }
  }

  // get labels for nonexisting dataelement
  BOOST_CHECK(cont2.getLabels(100).size() == 0);
}

BOOST_AUTO_TEST_CASE(MCTruthContainer_move)
{
  using TruthElement = long;
  using Container = dataformats::MCTruthContainer<TruthElement>;
  Container container;
  container.addElement(0, TruthElement(1));
  container.addElement(0, TruthElement(2));
  container.addElement(1, TruthElement(1));
  container.addElement(2, TruthElement(10));

  Container container2 = std::move(container);
  BOOST_CHECK(container.getIndexedSize() == 0);
  BOOST_CHECK(container.getNElements() == 0);

  std::swap(container, container2);
  BOOST_CHECK(container2.getIndexedSize() == 0);
  BOOST_CHECK(container2.getNElements() == 0);
  BOOST_CHECK(container.getIndexedSize() == 3);
  BOOST_CHECK(container.getNElements() == 4);
}

} // namespace o2
