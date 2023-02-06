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

#define BOOST_TEST_MODULE Test MCTruthContainer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include <algorithm>
#include <iostream>
#include <TFile.h>
#include <TTree.h>

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
  container.addNoLabelIndex(3);
  container.addElement(4, TruthElement(4));

  // not supported, must throw
  BOOST_CHECK_THROW(container.addElement(0, TruthElement(0)), std::runtime_error);

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

  // add multiple labels (to last index which already had a label)
  std::vector<TruthElement> newlabels = {101, 102, 103};
  container.addElements(4, newlabels);
  view = container.getLabels(4);
  BOOST_CHECK(view.size() == 4);
  BOOST_CHECK(view[0] == 4);
  BOOST_CHECK(view[1] == 101);
  BOOST_CHECK(view[2] == 102);
  BOOST_CHECK(view[3] == 103);

  // check empty labels case
  view = container.getLabels(3);
  BOOST_CHECK(view.size() == 0);

  // add empty label vector
  std::vector<TruthElement> newlabels2 = {};
  container.addElements(5, newlabels2);
  view = container.getLabels(5);
  BOOST_CHECK(view.size() == 0);

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

    dataformats::MCTruthContainer<TruthElement> containerA;

    container1.mergeAtBack(container2);

    containerA.mergeAtBack(container1, 0, 2);
    containerA.mergeAtBack(container1, 2, 2);

    auto lview = container1.getLabels(3); //
    auto lviewA = containerA.getLabels(3);
    BOOST_CHECK(lview.size() == 2);
    BOOST_CHECK(lview[0] == 11);
    BOOST_CHECK(lview[1] == 12);
    BOOST_CHECK(container1.getIndexedSize() == 6);
    BOOST_CHECK(container1.getNElements() == 8);
    BOOST_CHECK(lview.size() == lviewA.size());
    BOOST_CHECK(lview[0] == lviewA[0] && lview[1] == lviewA[1]);
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

BOOST_AUTO_TEST_CASE(MCTruthContainer_flatten)
{
  using TruthElement = long;
  using TruthContainer = dataformats::MCTruthContainer<TruthElement>;
  TruthContainer container;
  container.addElement(0, TruthElement(1));
  container.addElement(0, TruthElement(2));
  container.addElement(1, TruthElement(1));
  container.addElement(2, TruthElement(10));

  std::vector<char> buffer;
  container.flatten_to(buffer);
  BOOST_REQUIRE(buffer.size() > sizeof(TruthContainer::FlatHeader));
  auto& header = *reinterpret_cast<TruthContainer::FlatHeader*>(buffer.data());
  BOOST_CHECK(header.nofHeaderElements == container.getIndexedSize());
  BOOST_CHECK(header.nofTruthElements == container.getNElements());

  TruthContainer restoredContainer;
  restoredContainer.restore_from(buffer.data(), buffer.size());

  // check header/index information
  BOOST_CHECK(restoredContainer.getMCTruthHeader(0).index == 0);
  BOOST_CHECK(restoredContainer.getMCTruthHeader(1).index == 2);
  BOOST_CHECK(restoredContainer.getMCTruthHeader(2).index == 3);

  // check MC truth information
  BOOST_CHECK(restoredContainer.getElement(0) == 1);
  BOOST_CHECK(restoredContainer.getElement(1) == 2);
  BOOST_CHECK(restoredContainer.getElement(2) == 1);
  BOOST_CHECK(restoredContainer.getElement(3) == 10);

  // check the special version ConstMCTruthContainer
  using ConstMCTruthContainer = dataformats::ConstMCTruthContainer<TruthElement>;
  ConstMCTruthContainer cc;
  container.flatten_to(cc);

  BOOST_CHECK(cc.getIndexedSize() == container.getIndexedSize());
  BOOST_CHECK(cc.getNElements() == container.getNElements());
  BOOST_CHECK(cc.getLabels(0).size() == container.getLabels(0).size());
  BOOST_CHECK(cc.getLabels(1).size() == container.getLabels(1).size());
  BOOST_CHECK(cc.getLabels(2).size() == container.getLabels(2).size());
  BOOST_CHECK(cc.getLabels(2)[0] == container.getLabels(2)[0]);
  BOOST_CHECK(cc.getLabels(2)[0] == 10);
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

BOOST_AUTO_TEST_CASE(MCTruthContainer_ROOTIO)
{
  using TruthElement = o2::MCCompLabel;
  using Container = dataformats::MCTruthContainer<TruthElement>;
  Container container;
  const size_t BIGSIZE{1000000};
  container.addNoLabelIndex(0); // the first index does not have a label
  for (int i = 1; i < BIGSIZE; ++i) {
    container.addElement(i, TruthElement(i, i, i));
    container.addElement(i, TruthElement(i + 1, i, i));
  }
  std::vector<char> buffer;
  container.flatten_to(buffer);

  // We use the special IO split container to stream to a file and back
  dataformats::IOMCTruthContainerView io(buffer);
  {
    TFile f("tmp2.root", "RECREATE");
    TTree tree("o2sim", "o2sim");
    auto br = tree.Branch("Labels", &io, 32000, 2);
    tree.Branch("LabelsOriginal", &container, 32000, 2);
    tree.Fill();
    tree.Write();
    f.Close();
  }

  // read back
  TFile f2("tmp2.root", "OPEN");
  auto tree2 = (TTree*)f2.Get("o2sim");
  dataformats::IOMCTruthContainerView* io2 = nullptr;
  auto br2 = tree2->GetBranch("Labels");
  BOOST_CHECK(br2 != nullptr);
  br2->SetAddress(&io2);
  br2->GetEntry(0);

  // make a const MC label container out of it
  using ConstMCTruthContainer = dataformats::ConstMCTruthContainer<TruthElement>;
  ConstMCTruthContainer cc;
  io2->copyandflatten(cc);

  BOOST_CHECK(cc.getNElements() == (BIGSIZE - 1) * 2);
  BOOST_CHECK(cc.getIndexedSize() == BIGSIZE);
  BOOST_CHECK(cc.getLabels(0).size() == 0);
  BOOST_CHECK(cc.getLabels(1).size() == 2);
  BOOST_CHECK(cc.getLabels(1)[0] == TruthElement(1, 1, 1));
  BOOST_CHECK(cc.getLabels(1)[1] == TruthElement(2, 1, 1));
  BOOST_CHECK(cc.getLabels(BIGSIZE - 1).size() == 2);
  BOOST_CHECK(cc.getLabels(BIGSIZE - 1)[0] == TruthElement(BIGSIZE - 1, BIGSIZE - 1, BIGSIZE - 1));
  BOOST_CHECK(cc.getLabels(BIGSIZE - 1)[1] == TruthElement(BIGSIZE, BIGSIZE - 1, BIGSIZE - 1));

  // testing convenience API to retrieve a constant label container from a ROOT file, entry 0
  auto cont = o2::dataformats::MCLabelIOHelper::loadFromTTree(tree2, "Labels", 0);
  auto cont2 = o2::dataformats::MCLabelIOHelper::loadFromTTree(tree2, "LabelsOriginal", 0);

  BOOST_CHECK(cont);
  BOOST_CHECK(cont2);
  BOOST_CHECK(cont->getNElements() == (BIGSIZE - 1) * 2);
  BOOST_CHECK(cont2->getNElements() == (BIGSIZE - 1) * 2);
  BOOST_CHECK(cont->getLabels(0).size() == 0);
  BOOST_CHECK(cont2->getLabels(0).size() == 0);
  BOOST_CHECK(cont->getLabels(BIGSIZE - 1)[0] == TruthElement(BIGSIZE - 1, BIGSIZE - 1, BIGSIZE - 1));
  BOOST_CHECK(cont->getLabels(BIGSIZE - 1)[1] == TruthElement(BIGSIZE, BIGSIZE - 1, BIGSIZE - 1));
  BOOST_CHECK(cont2->getLabels(BIGSIZE - 1)[0] == TruthElement(BIGSIZE - 1, BIGSIZE - 1, BIGSIZE - 1));
  BOOST_CHECK(cont2->getLabels(BIGSIZE - 1)[1] == TruthElement(BIGSIZE, BIGSIZE - 1, BIGSIZE - 1));
}

} // namespace o2
