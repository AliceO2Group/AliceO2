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
#define BOOST_TEST_MODULE Test EMCAL Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <map>
#include <vector>
#include <gsl/span>
#include <CommonDataFormat/InteractionRecord.h>
#include <DataFormatsEMCAL/Cell.h>
#include <EMCALReconstruction/RecoContainer.h>

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(RecoContainer_test)
{
  RecoContainer testcontainer;

  std::map<o2::InteractionRecord, int> truthNumberCells, truthNumberLEDMONs;

  // test 1: Check appending cells to existing container
  o2::InteractionRecord testIR(1023, 384128);
  auto& currentEvent = testcontainer.getEventContainer(testIR);

  std::vector<int> towers1 = {12, 382, 922, 1911};
  std::vector<double> energies1 = {0.2, 10., 1.1, 0.4};
  std::vector<double> times1 = {1, 29, 0, 2};

  for (int icell = 0; icell < 4; icell++) {
    currentEvent.setCell(towers1[icell], energies1[icell], times1[icell], ChannelType_t::HIGH_GAIN, 9238, 1, 1, true);
  }

  BOOST_CHECK_EQUAL(currentEvent.getCells().size(), 4);

  std::vector<int> towers2 = {57, 292, 4592, 11922};
  std::vector<double> energies2 = {0.2, 10., 1.1, 0.4};
  std::vector<double> times2 = {10, 3, 1, 5};
  auto& newcurrent = testcontainer.getEventContainer(testIR);
  for (int icell = 0; icell < 4; icell++) {
    newcurrent.setCell(towers2[icell], energies2[icell], times2[icell], ChannelType_t::HIGH_GAIN, 9238, 2, 2, true);
  }

  BOOST_CHECK_EQUAL(newcurrent.getCells().size(), 8);
  BOOST_CHECK_EQUAL(testcontainer.getNumberOfEvents(), 1);
  truthNumberCells[testIR] = 8;
  truthNumberLEDMONs[testIR] = 0;

  // test 2: Adding new event to container
  o2::InteractionRecord secondIR(2021, 384130);
  auto& secondevent = testcontainer.getEventContainer(secondIR);
  BOOST_CHECK_EQUAL(testcontainer.getNumberOfEvents(), 2);
  BOOST_CHECK_EQUAL(secondevent.getCells().size(), 0);

  // test 3: Merge HG and LG cells
  secondevent.setCell(1023, 1.41023, 0.2, ChannelType_t::HIGH_GAIN, 2902, 1, 1, true);
  secondevent.setCell(1023, 1.4013, 0.91, ChannelType_t::LOW_GAIN, 2902, 1, 1, true);
  secondevent.setCell(2821, 16.2, 6, ChannelType_t::HIGH_GAIN, 1293, 3, 3, true);
  secondevent.setCell(2821, 129., 10, ChannelType_t::LOW_GAIN, 1293, 3, 3, true);
  BOOST_CHECK_EQUAL(secondevent.getCells().size(), 2);
  int nCellHG = 0, nCellLG = 0;
  for (const auto& cell : secondevent.getCells()) {
    switch (cell.mCellData.getType()) {
      case ChannelType_t::HIGH_GAIN:
        nCellHG++;
        break;
      case ChannelType_t::LOW_GAIN:
        nCellLG++;
        break;

      default:
        break;
    }
  }
  BOOST_CHECK_EQUAL(nCellHG, 1);
  BOOST_CHECK_EQUAL(nCellLG, 1);

  // test 4: test LGnoHG and HGSaturated
  secondevent.setCell(12034, 0.3, 10, ChannelType_t::LOW_GAIN, 3942, 10, 10, true);
  secondevent.setCell(12392, 18., 94, ChannelType_t::HIGH_GAIN, 1209, 20, 20, true);
  int nLGnoHG = 0, nHGOutOfRange = 0;
  for (const auto& cell : secondevent.getCells()) {
    if (cell.mIsLGnoHG) {
      nLGnoHG++;
    }
    if (cell.mHGOutOfRange) {
      nHGOutOfRange++;
    }
  }
  BOOST_CHECK_EQUAL(nLGnoHG, 1);
  BOOST_CHECK_EQUAL(nHGOutOfRange, 1);

  // test 5: test LEDMon Cells
  std::vector<int> ledmonTowers = {293, 3842, 1820};
  std::vector<double> ledmonEnergies = {5.4, 5.2, 6.2};
  std::vector<double> ledmonTimes = {230, 303, 280};
  for (int iledmon = 0; iledmon < 3; iledmon++) {
    secondevent.setLEDMONCell(ledmonTowers[iledmon], ledmonEnergies[iledmon], ledmonTimes[iledmon], ChannelType_t::HIGH_GAIN, 3302, 139, 20, true);
  }
  BOOST_CHECK_EQUAL(secondevent.getCells().size(), 4);
  BOOST_CHECK_EQUAL(secondevent.getLEDMons().size(), 3);
  truthNumberCells[secondIR] = 4;
  truthNumberLEDMONs[secondIR] = 3;

  // test 6: test sorting of collisions
  std::vector<o2::InteractionRecord> collisions{testIR, secondIR};
  std::sort(collisions.begin(), collisions.end(), std::less<>());
  auto sortedCollisions = testcontainer.getOrderedInteractions();
  BOOST_CHECK_EQUAL_COLLECTIONS(collisions.begin(), collisions.end(), sortedCollisions.begin(), sortedCollisions.end());

  // test 7: read the container
  RecoContainerReader iterator(testcontainer);
  std::vector<o2::InteractionRecord> foundInteractions;
  std::map<o2::InteractionRecord, int> foundNCells, foundLEDMONs;
  while (iterator.hasNext()) {
    auto& event = iterator.nextEvent();
    foundInteractions.push_back(event.getInteractionRecord());
    foundNCells[event.getInteractionRecord()] = event.getNumberOfCells();
    foundLEDMONs[event.getInteractionRecord()] = event.getNumberOfLEDMONs();
    if (event.getNumberOfCells()) {
      // test sorting cells
      event.sortCells(false);
      int lastcell = -1;
      for (const auto& cell : event.getCells()) {
        if (lastcell > -1) {
          BOOST_CHECK_LT(lastcell, cell.mCellData.getTower());
        }
        lastcell = cell.mCellData.getTower();
      }
    }
    if (event.getNumberOfLEDMONs()) {
      // test sorting LEDMONS
      event.sortCells(true);
      int lastLEDMON = -1;
      for (const auto& ledmon : event.getLEDMons()) {
        if (lastLEDMON > -1) {
          BOOST_CHECK_LT(lastLEDMON, ledmon.mCellData.getTower());
        }
        lastLEDMON = ledmon.mCellData.getTower();
      }
    }
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(collisions.begin(), collisions.end(), foundInteractions.begin(), foundInteractions.end());
  for (auto truthCell = truthNumberCells.begin(), foundCell = foundNCells.begin(); truthCell != truthNumberCells.end(); truthCell++, foundCell++) {
    BOOST_CHECK_EQUAL(truthCell->first.bc, foundCell->first.bc);
    BOOST_CHECK_EQUAL(truthCell->first.orbit, foundCell->first.orbit);
    BOOST_CHECK_EQUAL(truthCell->second, foundCell->second);
  }
  for (auto truthLEDMON = truthNumberLEDMONs.begin(), foundLEDMON = foundLEDMONs.begin(); truthLEDMON != truthNumberLEDMONs.end(); truthLEDMON++, foundLEDMON++) {
    BOOST_CHECK_EQUAL(truthLEDMON->first.bc, foundLEDMON->first.bc);
    BOOST_CHECK_EQUAL(truthLEDMON->first.orbit, foundLEDMON->first.orbit);
    BOOST_CHECK_EQUAL(truthLEDMON->second, foundLEDMON->second);
  }

  // test 8: reset Container
  testcontainer.reset();
  BOOST_CHECK_EQUAL(testcontainer.getNumberOfEvents(), 0);
}

} // namespace emcal
} // namespace o2