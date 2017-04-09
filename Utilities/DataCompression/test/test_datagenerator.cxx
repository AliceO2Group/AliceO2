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

//  @file   test_datagenerator.cxx
//  @author Matthias Richter
//  @since  2015-12-06
//  @brief  Test program for simple data generator

//
/*
   g++ --std=c++11 -g -ggdb -o test_datagenerator test_datagenerator.cxx
*/

#include "DataGenerator.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main()
{

  //typedef AliceO2::Test::normal_distribution<double> TestDistribution_t;
  //AliceO2::Test::DataGenerator<TestDistribution_t::result_type, TestDistribution_t> dg(-7.5, 7.5, 1., 0., 1.);
  //typedef AliceO2::Test::poisson_distribution<int> TestDistribution_t;
  //AliceO2::Test::DataGenerator<TestDistribution_t::result_type, TestDistribution_t> dg(0, 15, 1, 3);
  typedef o2::Test::geometric_distribution<int> TestDistribution_t;
  o2::Test::DataGenerator<TestDistribution_t::result_type, TestDistribution_t> dg(0, 31, 1, 0.3);

  typedef TestDistribution_t::result_type value_type;

  std::vector<value_type> throws(dg.nbins);
  const int nRolls = 1000000;

  for (int n = 0; n < nRolls; n++) {
    value_type v = dg();
    int bin = v/dg.step - dg.min;
    if (bin >= dg.nbins) {
      std::cout << v << " " << bin << std::endl;
    } else {
      throws[bin]++;
    }
  }

  for (auto i : dg) {
    int bin = i/dg.step - dg.min;
    std::cout << std::setw(4)  << std::right << i << ": "
              << std::setw(11) << std::left << dg.getProbability(i)
              << " -- "
              << throws[bin]
              << std::endl;
  }
}
