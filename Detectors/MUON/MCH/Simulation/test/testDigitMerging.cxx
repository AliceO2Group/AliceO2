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

#define BOOST_TEST_MODULE Test MCHSimulation DigitMerging
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "DataFormatsMCH/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DigitMerging.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

using o2::mch::Digit;

std::vector<Digit> createNonOverlappingDigits()
{
  return std::vector<Digit>{
    {100, 2, 5, 0},
    {100, 3, 6, 0},
    {100, 1, 2, 0},
    {100, 0, 1, 0}};
}

std::vector<o2::MCCompLabel> createLabelsNonOverlappingDigits()
{
  return std::vector<o2::MCCompLabel>{
    {0, 0, 10, false},
    {0, 0, 10, false},
    {10, 0, 10, false},
    {11, 0, 10, false}};
}

std::vector<Digit> createOverlappingDigits()
{
  return std::vector<Digit>{
    {100, 2, 5, 0},
    {100, 3, 6, 0},
    {100, 1, 2, 0},
    {100, 0, 0, 0},
    {100, 0, 1, 0},
    {100, 1, 3, 0},
    {100, 3, 7, 0},
    {100, 1, 4, 0}};
}

std::vector<o2::MCCompLabel> createLabelsOverlappingDigits()
{
  return std::vector<o2::MCCompLabel>{
    {0, 0, 10, false},
    {0, 0, 10, false},
    {10, 0, 10, false},
    {11, 0, 10, false},
    {10, 0, 10, false},
    {2, 0, 10, false},
    {4, 0, 10, false},
    {5, 0, 10, false},
    {6, 0, 10, false}};
}

std::vector<Digit> expected()
{
  return std::vector<Digit>{
    {100, 0, 1, 0},
    {100, 1, 9, 0},
    {100, 2, 5, 0},
    {100, 3, 13, 0}};
}

std::vector<o2::MCCompLabel> labelexpected()
{
  return std::vector<o2::MCCompLabel>{
    {0, 0, 10, false},
    {0, 0, 10, false},
    {10, 0, 10, false},
    {11, 0, 10, false}};
}

BOOST_DATA_TEST_CASE(DigitMergingIdentity, boost::unit_test::data::make(mergingFunctions()), mergingFunction)
{
  auto m = mergingFunction(createNonOverlappingDigits(), createLabelsNonOverlappingDigits());
  auto e = m;
  BOOST_CHECK(std::is_permutation(m.begin(), m.end(), e.begin()));
}

BOOST_DATA_TEST_CASE(DigitMerging, boost::unit_test::data::make(mergingFunctions()), mergingFunction)
{
  auto m = mergingFunction(createOverlappingDigits(), createLabelsOverlappingDigits());
  BOOST_CHECK(std::is_permutation(m.begin(), m.end(), expected().begin()));
}
