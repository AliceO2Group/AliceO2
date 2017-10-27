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
  BOOST_CHECK(container.getMCTruthHeader(0).size == 2);
  BOOST_CHECK(container.getMCTruthHeader(0).index == 0);
  BOOST_CHECK(container.getMCTruthHeader(1).size == 1);
  BOOST_CHECK(container.getMCTruthHeader(1).index == 2);
  BOOST_CHECK(container.getMCTruthHeader(2).size == 1);
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
  std::sort(view.begin(), view.end(), [](TruthElement a, TruthElement b){return a>b;});
  BOOST_CHECK(view[0] == 2);
  BOOST_CHECK(view[1] == 1);

  // same for another data index
  view = container.getLabels(2);
  BOOST_CHECK(view.size() == 1);
  BOOST_CHECK(view[0] == 10);

  // try to get something invalid
  view = container.getLabels(10);
  BOOST_CHECK(view.size() == 0);
}


BOOST_AUTO_TEST_CASE(MCTruth_Iter)
{
  using TruthElement = long;
  dataformats::MCTruthContainerList<TruthElement> container;
  container.addElement(0, TruthElement(1));
  container.addElement(1, TruthElement(1));
  container.addElement(0, TruthElement(10));
  container.addElement(2, TruthElement(20));

  BOOST_CHECK(container.getNElements() == 4);

  auto iter = container.begin(2);
  for(;iter!=container.end();++iter){
    std::cerr << *iter << "\n";
  }

  iter = container.begin(2);
  for(;iter!=container.end();++iter){
    std::cerr << *iter << "\n";
  }

  auto view = container.getLabels(0);
  for(auto &e : view) {
    std::cerr << e << "\n";
  }
 
  // FIXME: sorting
  // std::sort(view.begin(), view.end(), [](TruthElement a, TruthElement b){return a>b;});
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
  for(auto &e : view) {
    std::cerr << e << "\n";
  }

  std::vector<TruthElement> v;
  container.fillVectorOfLabels(0,v);  
  BOOST_CHECK(v.size()==2);
  std::sort(v.begin(), v.end(), [](TruthElement a, TruthElement b){return a>b;});
  for(auto &e : v) {
    std::cerr << e << "\n";
  }
  
  // in this case we have to add the elements contiguously per dataindex:
  dataformats::LabelContainer<TruthElement, true> cont2;
  cont2.addLabel(0, TruthElement(1));
  cont2.addLabel(0, TruthElement(10));
  cont2.addLabel(1, TruthElement(1));
  cont2.addLabel(2, TruthElement(20));

  auto view2 = cont2.getLabels(0);
  BOOST_CHECK(view2.size() == 2);
  for(auto &e : view2) {
    std::cerr << e << "\n";
  }

  // get labels for nonexisting dataelement
  BOOST_CHECK(cont2.getLabels(100).size() == 0);
}

} // end namespace
