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

#define BOOST_TEST_MODULE Test TRD_TrapConfigEvent
/// \file testTRDTrapConfigEvent
/// \brief This task tests the trap config

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <algorithm>
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "DataFormatsTRD/HelperMethods.h"

namespace o2::trd
{

// structs to make defining the tests easier.
struct mcmIndexing {
  uint mSector;
  uint mStack;
  uint mLayer;
  uint mRob;
  uint mMcm;
  mcmIndexing(int sector, int stack, int layer, int rob, int mcm) : mSector(sector), mStack(stack), mLayer(layer), mRob(rob), mMcm(mcm){};
};

struct addresstest {
  uint32_t mAddress;
  uint32_t mValue;
  addresstest(uint32_t address, uint32_t value) : mAddress(address), mValue(value){};
};

// vectors of what and where we want to test.
std::vector<int> mcmids = {0, 1, 127, 128, 12917, 69119, 69120, 69130};
std::vector<mcmIndexing> mcmfullindex = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 0, 7, 15}, {0, 0, 1, 0, 0}, {3, 1, 4, 7, 5}, {17, 4, 5, 7, 15}, {17, 4, 5, 8, 15}, {18, 0, 0, 1, 2}}; // decomposition of the line above, last 2 of course being fictitious

std::vector<uint32_t> registersOfInterest; // store of the registers we want to look at.
std::vector<addresstest> registervalues;   // index by address,value

void trapregCheck(std::unique_ptr<TrapConfigEvent>& trapconfig, uint32_t mcmidx, int mcmidxcount)
{
  // loop over the mcm
  // loop over the register
  for (const auto& [key, value] : registervalues) {
    uint32_t address = key;
    uint32_t index = trapconfig->getRegIndexByAddr(address);
    std::string name = trapconfig->getRegNameByAddr(address);
    if (trapconfig->setRegisterValueByIdx(value, index, mcmidx)) {
      BOOST_CHECK_EQUAL(trapconfig->getRegisterValueByIdx(index, mcmidx), value);
      BOOST_CHECK_EQUAL(trapconfig->getRegisterValueByAddr(address, mcmidx), value);
      BOOST_CHECK_EQUAL(trapconfig->getRegisterValueByIdx(index, HelperMethods::getMCMId(mcmfullindex[mcmidxcount].mSector, mcmfullindex[mcmidxcount].mStack, mcmfullindex[mcmidxcount].mLayer, mcmfullindex[mcmidxcount].mRob, mcmfullindex[mcmidxcount].mMcm)), value);
      BOOST_CHECK_EQUAL(trapconfig->getRegisterValueByAddr(address, HelperMethods::getMCMId(mcmfullindex[mcmidxcount].mSector, mcmfullindex[mcmidxcount].mStack, mcmfullindex[mcmidxcount].mLayer, mcmfullindex[mcmidxcount].mRob, mcmfullindex[mcmidxcount].mMcm)), value);
      BOOST_CHECK_EQUAL(trapconfig->getRegisterValueByName(name, mcmidx), value);
    }
  }
}

/// \brief Test the trap register generation functions
//
BOOST_AUTO_TEST_CASE(TRDTrapConfigEventInternals)
{
  // test trap register initialisation
  TrapRegInfo trapreg;
  trapreg.init("testreg", 0x3000, 6, 0, 1, false, 6);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 20);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x3f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 6, 0, 5, false, 6);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 26);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x3f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 1);
  trapreg.init("testreg", 0x3000, 5, 0, 1, false, 5);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 22);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x1f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 5, 0, 4, false, 5);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 7);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x1f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 5, 0, 5, false, 5);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 2);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x1f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 5, 0, 6, false, 5);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 27);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x1f);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 1);
  trapreg.init("testreg", 0x3000, 31, 0, 1, false, 31);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 1);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x7fffffff);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 1);
  trapreg.init("testreg", 0x3000, 31, 0, 0, false, 31);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 1);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0x7fffffff);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 32, 0, 0, false, 31);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 0);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0xffffffff);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 0);
  trapreg.init("testreg", 0x3000, 32, 0, 1, false, 31);
  BOOST_CHECK_EQUAL(trapreg.getShift(), 0);
  BOOST_CHECK_EQUAL(trapreg.getMask(), 0xffffffff);
  BOOST_CHECK_EQUAL(trapreg.getDataWordNumber(), 1);
}

BOOST_AUTO_TEST_CASE(TRDTrapConfigEventGetSet)
{

  std::unique_ptr<TrapConfigEvent> trapconfig(new TrapConfigEvent());

  std::vector<uint32_t> registerAddressToLookAt = {
    0x3180, // first register TPL00
    0x3185, // TPL05 last reg of first 32 bit word
    0x3186, // TPL06 first reg of second 32 bit word
    0x3187, // TPL07 second reg of second 32 bit word
    0x31ff, // TPL7F last TPL register
    0x30A0, // FGA0 first reg after TPL7f
    0x30A4, // last 6bit reg in a 32 bit word
    0x30A5, // the next 6bit reg, first in the subsequent 32 bit word
    0x30A6, // the next 6bit reg, second in the subsequent 32 bit word
    0x30A7, // the next 6bit reg, third in the subsequent 32 bit word
    0x30B4, // last 6 bit register
    0x3080, // first 10 bit register
    0x308C, // next 2 of 10 bit  register spanning a 32 bit data word.
    0x308D, // 2nd part of above
    0x313F, //
    0x3000, //
    0x3002, //
    0x3003, //
    0x300F, //
    0x3020, //
    0x3022, //
    0x3028, //
    0x302A, // 15 bits last of the set by itself in a lone 32 bit reg
    0x3030, //
    0x3050, //
    0x3051, //
    0x3052, // 5 bit in the middle of a register and the last of the set.
    0x3053, //
    0x0B6C, // last 10 bit word
    0x0B6D, // then a 32 bit register again
    0x0B6E, // first 15 bit reg
    0x0B6F, // last 15 bit reg in a 32bit word
    0x0B80, // first
    0x0B81, // second
    0xD000, // lone 32 bit reg
    0xD001, // lone 16 bit reg
    0xD002, // first 10 after a 16
    0xD003, // middle 10 in a 32 bit word (30 used)
    0x0D40, // last 10 bit in a 32 bit word
    0x0D41, // 11 bit in the subsequent 32 bit word*/
    0x315C  // last register
  };

  // setup the resgisters we will look at chosen for various reasons, size, on the edges, changes of register bit size
  for (auto& reg : registerAddressToLookAt) {
    registersOfInterest.push_back(trapconfig->getRegIndexByAddr(reg));
    // add a zero value, value of 1, mid point and its max.
    auto max = trapconfig->getRegisterMax(reg);
    registervalues.emplace_back(addresstest(reg, 0));
    registervalues.emplace_back(addresstest(reg, 1));
    registervalues.emplace_back(addresstest(reg, max / 2));
    registervalues.emplace_back(addresstest(reg, max));
  }

  // walk through map and test
  int count = 0;
  for (auto& mcm : mcmids) {
    trapregCheck(trapconfig, mcm, count);
    ++count;
  }
  // all those values have been written so we can now try read those back in bulk
  std::unique_ptr<std::array<uint32_t, constants::MAXMCMCOUNT * TrapConfigEvent::kLastReg>> allregisters(new std::array<uint32_t, constants::MAXMCMCOUNT * TrapConfigEvent::kLastReg>()); // all data
  std::unique_ptr<std::array<uint32_t, constants::MAXMCMCOUNT>> allregisterdata(new std::array<uint32_t, constants::MAXMCMCOUNT>());                                                      // data for a single register across all mcm
  std::unique_ptr<std::array<uint32_t, TrapConfigEvent::kLastReg>> allmcmregisters(new std::array<uint32_t, TrapConfigEvent::kLastReg>());                                                // data for a single mcm, all registers.
  std::memset(&allregisters->at(0), 0, sizeof(uint32_t));
  std::memset(&allregisterdata->at(0), 0, sizeof(uint32_t));
  std::memset(&allmcmregisters->at(0), 0, sizeof(uint32_t));
  // loop over all registers written in trapregcheck and check the incoming array that those values are set.
  int registerindex = 0;
  uint32_t registerreadvalue = 0;
  for (int valuecount = 0; valuecount < 4; ++valuecount) {
    // loop over each of the value sets 0,1,mid, max
    std::vector<uint32_t> addressesseen;
    std::map<uint32_t, uint32_t> setValues;
    // loop through all registers for pulling out the valuecount instance of each, put data in and then check against the regall;
    // its sorted so we know a key will come 4 times sequentially, and use valuecount to denote which value we are working with.
    for (int regtest = valuecount; regtest < registervalues.size(); regtest += 4) { // loop over the 4th elements to pull out like values, 0,1,max/2,max
      registerreadvalue = registervalues[regtest].mValue;
      for (auto& mcmidx : mcmids) {
        uint32_t address = registervalues[regtest].mAddress;

        uint32_t index = trapconfig->getRegIndexByAddr(address);
        std::string name = trapconfig->getRegNameByAddr(address);
        uint32_t retval = trapconfig->setRegisterValueByIdx(registerreadvalue, index, mcmidx);
      }
    }
    // all registers now written to ccdbconfig for this value set
    int elemcount = 0;
    // now check the values that were written
    trapconfig->getAll(*allregisters);
    // check done in next double for loop
    uint32_t value = 0;
    int registersofinterestindex = 0;
    for (auto& reg : registersOfInterest) {
      // pull out the values on for a register
      trapconfig->getAllMCMByAddress(reg, *allregisterdata);
      // check the values are set correctly
      for (auto& mcmidx : mcmids) {
        // loop over all mcm for the given register
        if (mcmidx < constants::MAXMCMCOUNT) {
          auto regidx = reg;
          value = registervalues[registersofinterestindex * 4 + valuecount].mValue;
          BOOST_CHECK_EQUAL(allregisterdata->at(mcmidx), value);
          BOOST_CHECK_EQUAL(allregisters->at(mcmidx * TrapConfigEvent::kLastReg + regidx), value); // do the big one at the same time as the little one, pointless repeating the loop seperately
        }
      }
      ++registersofinterestindex;
    }

    for (auto& mcmidx : mcmids) {
      if (mcmidx < constants::MAXMCMCOUNT) {
        // pull out the values on for a specific mcm
        trapconfig->getAllRegisters(mcmidx, *allmcmregisters);
        // check the values are set correctly
        int regcount = 0;
        for (auto& reg : registersOfInterest) {
          // loop over all registers for the given mcmq
          uint32_t regindex = reg;
          value = registervalues[regcount * 4 + valuecount].mValue;
          BOOST_CHECK_EQUAL(allmcmregisters->at(regindex), value);
          ++regcount;
        }
      }
    }
  }
}
// still to test
// get
} // namespace o2::trd
