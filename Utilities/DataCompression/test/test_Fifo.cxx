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

//  @file   test_Fifo.cxx
//  @author Matthias Richter
//  @since  2016-12-06
//  @brief  Test program for thread safe FIFO

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "Fifo.h"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <thread>

const int nEvents = 6;
const int sleepTime = 1;

template<typename T>
bool processValue(T value) {
  std::cout << "processing " << value << std::endl;
  sleep(sleepTime);
  return (value + 1 < nEvents);
}

template<class FifoT, typename T>
void pushFifo(FifoT& fifo, T value = FifoT::value_type) {
  std::cout << "pushing " << value << std::endl;
  fifo.push(value);
}

BOOST_AUTO_TEST_CASE(test_Fifo)
{
  using value_type = unsigned int;
  o2::test::Fifo<value_type> fifo;

  // start a consumer thread which pulls from the FIFO to the function
  // processValue with a simulated prcessing of one second.
  std::thread consumer([&fifo]()
                       {
                         do {}
                         while (fifo.pull([](value_type v)
                                          {
                                            return processValue(v);
                                          }
                                          )
                                );
                       }
                       );

  // fill some values into the FIFO which the consumer can process
  // immediately
  unsigned int value = 0;
  pushFifo(fifo, value++);
  pushFifo(fifo, value++);
  pushFifo(fifo, value++);

  // now continue filling with a period longer than the consumer
  // processing, consumer and producer are in sync once consumer
  // has processed events which have been added to the FIFO before
  while (value<nEvents) {
    sleep(2*sleepTime);
    pushFifo(fifo, value++);
  }

  consumer.join();
}
