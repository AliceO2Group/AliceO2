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

//  @file   test_fifo.cxx
//  @author Matthias Richter
//  @since  2015-12-06
//  @brief  Test program for thread safe FIFO

//
/*
   g++ --std=c++11 -g -ggdb -pthread -o test_fifo test_fifo.cxx
*/

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

template<class Fifo, typename T>
void pushFifo(Fifo& fifo, T value) {
  std::cout << "pushing " << value << std::endl;
  fifo.push(value);
}

int main()
{
  o2::Test::Fifo<unsigned int> fifo;

  // start a consumer thread which pulls from the FIFO to the function
  // processValue with a simulated prcessing of one second.
  std::thread consumer([&fifo]()
                       {
                         do {}
                         while (fifo.pull([](unsigned int v)
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
