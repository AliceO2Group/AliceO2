// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   test_DataGenerator.cxx
//  @author Matthias Richter
//  @since  2016-12-06
//  @brief  Test program for simple data generator

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataGenerator.h"
#include <iostream>
#include <iomanip>
#include <vector>

template <typename DistributionType, typename... Args>
bool testWithDistribution(Args&&... args)
{
  using value_type = typename DistributionType::result_type;
  o2::test::DataGenerator<value_type, DistributionType> dg(std::forward<Args>(args)...);

  std::vector<int> throws(dg.nbins);
  const unsigned nRolls = 1000000;

  for (unsigned n = 0; n < nRolls; n++) {
    value_type v = dg();
    unsigned bin = v / dg.step - dg.min;
    BOOST_REQUIRE(bin < dg.nbins);
    throws[bin]++;
  }

  int mostAbundantValueBin = 0;
  int mostAbundantValueCount = 0;
  auto highestProbability = dg.getProbability(dg.min);
  highestProbability = 0;
  for (auto i : dg) {
    int bin = i / dg.step - dg.min;
    BOOST_REQUIRE(bin >= 0);
    if (mostAbundantValueCount < throws[bin]) {
      mostAbundantValueBin = bin;
      mostAbundantValueCount = throws[bin];
    }
    if (highestProbability < dg.getProbability(i)) {
      highestProbability = dg.getProbability(i);
    }
    std::cout << std::setw(4) << std::right << i << ": "            //
              << std::setw(11) << std::left << dg.getProbability(i) //
              << " -- "                                             //
              << throws[bin]                                        //
              << std::endl;                                         //
  }
  std::vector<int> mostProbableValueBins;
  for (auto i : dg) {
    int bin = i / dg.step - dg.min;
    if (dg.getProbability(i) >= highestProbability) {
      mostProbableValueBins.push_back(bin);
    }
  }
  auto& list = mostProbableValueBins;
  BOOST_CHECK(std::find(list.begin(), list.end(), mostAbundantValueBin) != list.end());

  return true;
}

BOOST_AUTO_TEST_CASE(test_DataGenerator)
{
  std::cout << "Testing normal distribution" << std::endl;
  using normal_distribution = o2::test::normal_distribution<double>;
  testWithDistribution<normal_distribution>(-7.5, 7.5, 1., 0., 1.);

  std::cout << "Testing poisson distribution" << std::endl;
  using poisson_distribution = o2::test::poisson_distribution<int>;
  testWithDistribution<poisson_distribution>(0, 15, 1, 3);

  std::cout << "Testing geometric distribution" << std::endl;
  using geometric_distribution = o2::test::geometric_distribution<int>;
  testWithDistribution<geometric_distribution>(0, 31, 1, 0.3);
}
