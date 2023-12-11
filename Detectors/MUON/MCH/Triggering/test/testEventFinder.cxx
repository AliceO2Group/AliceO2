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

/// \file testEventFinder.cxx
/// \brief Test the grouping of MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

// test pattern:
// BC     :  0       4       8       12      16      20      24      28      32
// MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
// MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -

#define BOOST_TEST_MODULE Test MCHTriggering EventFinder
#define BOOST_TEST_DYN_LINK

#include <initializer_list>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>

#include "CommonUtils/ConfigurableParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MCHTriggering/EventFinder.h"

BOOST_AUTO_TEST_SUITE(mchtriggering)

using ObjLabel = std::pair<uint32_t, o2::MCCompLabel>;

o2::dataformats::MCLabelContainer createLabels(std::initializer_list<ObjLabel> objLabels)
{
  o2::dataformats::MCLabelContainer labels{};
  for (const auto& objLabel : objLabels) {
    labels.addElement(objLabel.first, objLabel.second);
  }
  return labels;
}

std::vector<o2::mch::ROFRecord> mchROFs = {
  {{0, 0}, 0, 2, 4},   // contains digit #1,2
  {{4, 0}, 2, 1, 4},   // contains digit #3
  {{8, 0}, 3, 3, 4},   // contains digit #4,5,6
  {{20, 0}, 6, 1, 4}}; // contains digit #7

std::vector<o2::mch::Digit> mchDigits = {
  {100, 0, 10, 0, 1, false},   // #1
  {100, 1, 10, 0, 1, false},   // #2
  {100, 2, 10, 4, 1, false},   // #3
  {100, 2, 10, 8, 1, true},    // #4
  {100, 3, 10, 8, 1, false},   // #5
  {100, 0, 10, 8, 1, false},   // #6
  {100, 2, 10, 20, 1, false}}; // #7

o2::dataformats::MCLabelContainer mchLabels = createLabels(
  {{0, {0, 0, 0}},   // digit #1
   {1, {1, 0, 0}},   // digit #2
   {1, {2, 0, 0}},   // digit #2
   {2, {0, 1, 0}},   // digit #3
   {3, {0, 2, 0}},   // digit #4
   {3, {true}},      // digit #4
   {4, {1, 2, 0}},   // digit #5
   {5, {2, 2, 0}},   // digit #6
   {6, {0, 3, 0}}}); // digit #7

std::vector<o2::mid::ROFRecord> midROFs = {
  {{3, 0}, o2::mid::EventType::Standard, 0, 1},
  {{13, 0}, o2::mid::EventType::Standard, 0, 1},
  {{20, 0}, o2::mid::EventType::Standard, 0, 1}};

template <typename T>
void compare(const std::vector<T>& output, const std::vector<T>& expected)
{
  BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(), output.end(),
                                expected.begin(), expected.end());
}

void compare(const o2::dataformats::MCLabelContainer& output, const o2::dataformats::MCLabelContainer& expected)
{
  BOOST_CHECK_EQUAL(output.getIndexedSize(), expected.getIndexedSize());

  for (int i = 0; i < output.getIndexedSize(); ++i) {
    const auto outputLabels = output.getLabels(i);
    const auto expectedLabels = expected.getLabels(i);
    BOOST_CHECK_EQUAL_COLLECTIONS(outputLabels.begin(), outputLabels.end(),
                                  expectedLabels.begin(), expectedLabels.end());
  }
}

BOOST_AUTO_TEST_CASE(TriggerWidth1BC)
{
  // BC     :  0       4       8       12      16      20      24      28      32
  // MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
  // MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // TRG rge:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // OUT ROF:  - - -[-]- - - - - - - - - - - - - - - -[-]- - - - - - - - - - - -
  // OUT ROF #1 contains digits from MCH ROF #1
  // OUT ROF #2 contains digits from MCH ROF #4

  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[0]", 0);
  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[1]", 1);

  o2::mch::EventFinder eventFinder{};
  eventFinder.run(mchROFs, mchDigits, &mchLabels, midROFs);

  std::vector<o2::mch::ROFRecord> expectedROFs = {
    {{3, 0}, 0, 2, 1},   // contains digit #1,2
    {{20, 0}, 2, 1, 1}}; // contains digit #7

  std::vector<o2::mch::Digit> expectedDigits = {
    {100, 0, 10, 0, 1, false},   // #1
    {100, 1, 10, 0, 1, false},   // #2
    {100, 2, 10, 20, 1, false}}; // #7

  o2::dataformats::MCLabelContainer expectedLabels = createLabels(
    {{0, {0, 0, 0}},   // digit #1
     {1, {1, 0, 0}},   // digit #2
     {1, {2, 0, 0}},   // digit #2
     {2, {0, 3, 0}}}); // digit #7

  compare(eventFinder.getOutputROFs(), expectedROFs);
  compare(eventFinder.getOutputDigits(), expectedDigits);
  compare(eventFinder.getOutputLabels(), expectedLabels);
}

BOOST_AUTO_TEST_CASE(TriggerWidth4BC)
{
  // BC     :  0       4       8       12      16      20      24      28      32
  // MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
  // MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // TRG rge:  -[- - - -]- - - - - -[- - - -]- - -[- - - -]- - - - - - - - - - -
  // OUT ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // OUT ROF #1 contains digits from MCH ROF #1,2
  // OUT ROF #2 contains digits from MCH ROF #3
  // OUT ROF #3 contains digits from MCH ROF #4

  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[0]", -2);
  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[1]", 2);

  o2::mch::EventFinder eventFinder{};
  eventFinder.run(mchROFs, mchDigits, &mchLabels, midROFs);

  std::vector<o2::mch::ROFRecord> expectedROFs = {
    {{3, 0}, 0, 3, 1},   // contains digit #1,2,3
    {{13, 0}, 3, 3, 1},  // contains digit #4,5,6
    {{20, 0}, 6, 1, 1}}; // contains digit #7

  std::vector<o2::mch::Digit> expectedDigits = {
    {100, 0, 10, 0, 1, false},   // #1
    {100, 1, 10, 0, 1, false},   // #2
    {100, 2, 10, 4, 1, false},   // #3
    {100, 2, 10, 8, 1, true},    // #4
    {100, 3, 10, 8, 1, false},   // #5
    {100, 0, 10, 8, 1, false},   // #6
    {100, 2, 10, 20, 1, false}}; // #7

  o2::dataformats::MCLabelContainer expectedLabels = createLabels(
    {{0, {0, 0, 0}},   // digit #1
     {1, {1, 0, 0}},   // digit #2
     {1, {2, 0, 0}},   // digit #2
     {2, {0, 1, 0}},   // digit #3
     {3, {0, 2, 0}},   // digit #4
     {3, {true}},      // digit #4
     {4, {1, 2, 0}},   // digit #5
     {5, {2, 2, 0}},   // digit #6
     {6, {0, 3, 0}}}); // digit #7

  compare(eventFinder.getOutputROFs(), expectedROFs);
  compare(eventFinder.getOutputDigits(), expectedDigits);
  compare(eventFinder.getOutputLabels(), expectedLabels);
}

BOOST_AUTO_TEST_CASE(TriggerWidth9BC)
{
  // BC     :  0       4       8       12      16      20      24      28      32
  // MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
  // MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // TRG rge: [- - - - - - - - -]-[- - - - - - -[- -]- - - - - - -]- - - - - - -
  // OUT ROF:  - - -[- - - - - - - - - - -]- - - - - -[-]- - - - - - - - - - - -
  // OUT ROF #1 contains digits from MCH ROF #1,2,3
  // OUT ROF #2 contains digits from MCH ROF #4

  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[0]", -3);
  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[1]", 6);

  o2::mch::EventFinder eventFinder{};
  eventFinder.run(mchROFs, mchDigits, &mchLabels, midROFs);

  std::vector<o2::mch::ROFRecord> expectedROFs = {
    {{3, 0}, 0, 4, 11},  // contains digit #1+6,2,3+4,5
    {{20, 0}, 4, 1, 1}}; // contains digit #7

  std::vector<o2::mch::Digit> expectedDigits = {
    {100, 0, 20, 0, 2, false},   // #1+6
    {100, 1, 10, 0, 1, false},   // #2
    {100, 2, 20, 4, 2, true},    // #3+4
    {100, 3, 10, 8, 1, false},   // #5
    {100, 2, 10, 20, 1, false}}; // #7

  o2::dataformats::MCLabelContainer expectedLabels = createLabels(
    {{0, {0, 0, 0}},   // digit #1+6
     {0, {2, 2, 0}},   // digit #1+6
     {1, {1, 0, 0}},   // digit #2
     {1, {2, 0, 0}},   // digit #2
     {2, {0, 1, 0}},   // digit #3+4
     {2, {0, 2, 0}},   // digit #3+4
     {2, {true}},      // digit #3+4
     {3, {1, 2, 0}},   // digit #5
     {4, {0, 3, 0}}}); // digit #7

  compare(eventFinder.getOutputROFs(), expectedROFs);
  compare(eventFinder.getOutputDigits(), expectedDigits);
  compare(eventFinder.getOutputLabels(), expectedLabels);
}

BOOST_AUTO_TEST_CASE(TriggerWidth15BC)
{
  // BC     :  0       4       8       12      16      20      24      28      32
  // MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
  // MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // TRG rge:[.- - - - - - - -[- - - - -]- -[- - - - - - - -]- - - - - - -]- - -
  // OUT ROF:  - - -[- - - - - - - - - - - - - - - - - -]- - - - - - - - - - - -
  // OUT ROF #1 contains digits from MCH ROF #1,2,3,4

  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[0]", -5);
  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[1]", 10);

  o2::mch::EventFinder eventFinder{};
  eventFinder.run(mchROFs, mchDigits, &mchLabels, midROFs);

  std::vector<o2::mch::ROFRecord> expectedROFs = {
    {{3, 0}, 0, 4, 18}}; // contains digit #1+6,2,3+4+7,5

  std::vector<o2::mch::Digit> expectedDigits = {
    {100, 0, 20, 0, 2, false},  // #1+6
    {100, 1, 10, 0, 1, false},  // #2
    {100, 2, 30, 4, 3, true},   // #3+4+7
    {100, 3, 10, 8, 1, false}}; // #5

  o2::dataformats::MCLabelContainer expectedLabels = createLabels(
    {{0, {0, 0, 0}},   // digit #1+6
     {0, {2, 2, 0}},   // digit #1+6
     {1, {1, 0, 0}},   // digit #2
     {1, {2, 0, 0}},   // digit #2
     {2, {0, 1, 0}},   // digit #3+4+7
     {2, {0, 2, 0}},   // digit #3+4+7
     {2, {true}},      // digit #3+4+7
     {2, {0, 3, 0}},   // digit #3+4+7
     {3, {1, 2, 0}}}); // digit #5

  compare(eventFinder.getOutputROFs(), expectedROFs);
  compare(eventFinder.getOutputDigits(), expectedDigits);
  compare(eventFinder.getOutputLabels(), expectedLabels);
}

BOOST_AUTO_TEST_CASE(TriggerWidth4BCShift10BC)
{
  // BC     :  0       4       8       12      16      20      24      28      32
  // MCH ROF: [- - - -|- - - -|- - - -]- - - - - - - -[- - - -]- - - - - - - - -
  // MID ROF:  - - -[-]- - - - - - - - -[-]- - - - - -[-]- - - - - - - - - - - -
  // TRG rge:  - - - - - - - - - - -[- - - -]- - - - - -[- - - -]- - -[- - - -]-
  // OUT ROF:  - - -[-]- - - - - - - - -[-]- - - - - - - - - - - - - - - - - - -
  // OUT ROF #1 contains digits from MCH ROF #3
  // OUT ROF #2 contains digits from MCH ROF #4

  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[0]", 8);
  o2::conf::ConfigurableParam::setValue("MCHTriggering", "triggerRange[1]", 12);

  o2::mch::EventFinder eventFinder{};
  eventFinder.run(mchROFs, mchDigits, &mchLabels, midROFs);

  std::vector<o2::mch::ROFRecord> expectedROFs = {
    {{3, 0}, 0, 3, 1},   // contains digit #4,5,6
    {{13, 0}, 3, 1, 1}}; // contains digit #7

  std::vector<o2::mch::Digit> expectedDigits = {
    {100, 2, 10, 8, 1, true},    // #4
    {100, 3, 10, 8, 1, false},   // #5
    {100, 0, 10, 8, 1, false},   // #6
    {100, 2, 10, 20, 1, false}}; // #7

  o2::dataformats::MCLabelContainer expectedLabels = createLabels(
    {{0, {0, 2, 0}},   // digit #4
     {0, {true}},      // digit #4
     {1, {1, 2, 0}},   // digit #5
     {2, {2, 2, 0}},   // digit #6
     {3, {0, 3, 0}}}); // digit #7

  compare(eventFinder.getOutputROFs(), expectedROFs);
  compare(eventFinder.getOutputDigits(), expectedDigits);
  compare(eventFinder.getOutputLabels(), expectedLabels);
}

BOOST_AUTO_TEST_SUITE_END()
