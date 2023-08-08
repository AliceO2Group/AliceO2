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
#include "DataFormatsTRD/TrapRegisters.h"
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

struct regtotest {
  uint32_t mRegIdx;
  std::array<uint32_t, 4> mValue;
  regtotest(uint32_t regidx, std::array<uint32_t, 4> value) : mRegIdx(regidx), mValue(value){};
};

// vectors of what and where we want to test.
std::vector<int> mcmids = {0, 1, 127, 128, 12917, 69119, 69120, 69130};
std::vector<mcmIndexing> mcmfullindex = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 0, 7, 15}, {0, 0, 1, 0, 0}, {3, 1, 4, 7, 5}, {17, 4, 5, 7, 15}, {17, 4, 5, 8, 15}, {18, 0, 0, 1, 2}}; // decomposition of the line above, last 2 of course being fictitious

std::vector<uint32_t> registersOfInterest; // store of the registers we want to look at.
std::vector<regtotest> registervalues;     // index by regidx,value[4]

void trapregCheck(std::unique_ptr<TrapConfigEvent>& trapconfig, uint32_t mcmidx, int mcmidxcount)
{
  // loop over the mcm
  // loop over the register
  for (const auto& [key, values] : registervalues) {
    uint32_t regidx = key;
    uint32_t index = regidx;
    std::string name = trapconfig->getRegisterName(index);
    for (auto& value : values) {
      if (trapconfig->setRegisterValue(value, index, mcmidx)) {
        BOOST_CHECK_EQUAL(trapconfig->getRegisterValue(index, mcmidx), value);
        BOOST_CHECK_EQUAL(trapconfig->getRegisterValue(regidx, mcmidx), value);
        BOOST_CHECK_EQUAL(trapconfig->getRegisterValue(index, HelperMethods::getMCMId(mcmfullindex[mcmidxcount].mSector, mcmfullindex[mcmidxcount].mStack, mcmfullindex[mcmidxcount].mLayer, mcmfullindex[mcmidxcount].mRob, mcmfullindex[mcmidxcount].mMcm)), value);
        BOOST_CHECK_EQUAL(trapconfig->getRegisterValue(regidx, HelperMethods::getMCMId(mcmfullindex[mcmidxcount].mSector, mcmfullindex[mcmidxcount].mStack, mcmfullindex[mcmidxcount].mLayer, mcmfullindex[mcmidxcount].mRob, mcmfullindex[mcmidxcount].mMcm)), value);
        BOOST_CHECK_EQUAL(trapconfig->getRegisterValue(trapconfig->getRegIndexByName(name), mcmidx), value);
      }
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

  std::unique_ptr<TrapConfigEvent> trapconfig1(new TrapConfigEvent());

  std::vector<uint32_t> registerToLookAt = {
    TrapRegisters::kTPL00,   // first register TPL00
    TrapRegisters::kTPL05,   // TPL05 last reg of first 32 bit word
    TrapRegisters::kTPL06,   // TPL06 first reg of second 32 bit word
    TrapRegisters::kTPL07,   // TPL07 second reg of second 32 bit word
    TrapRegisters::kTPL7F,   // TPL7F last TPL register
    TrapRegisters::kFGA0,    // FGA0 first reg after TPL7f
    TrapRegisters::kFGA4,    // last 6bit reg in a 32 bit word
    TrapRegisters::kFGA5,    // the next 6bit reg, first in the subsequent 32 bit word
    TrapRegisters::kFGA6,    // the next 6bit reg, second in the subsequent 32 bit word
    TrapRegisters::kFGA7,    // the next 6bit reg, third in the subsequent 32 bit word
    TrapRegisters::kFGA20,   // last 6 bit register
    TrapRegisters::kFGF0,    // first 10 bit register
    TrapRegisters::kFGF12,   // next 2 of 10 bit  register spanning a 32 bit data word.
    TrapRegisters::kFGF13,   // 2nd part of above
    TrapRegisters::kFLL3F,   //
    TrapRegisters::kTPPT0,   //
    TrapRegisters::kTPFE,    //
    TrapRegisters::kTPPGR,   //
    TrapRegisters::kEBPC,    //
    TrapRegisters::kFPTC,    //
    TrapRegisters::kFPCL,    //
    TrapRegisters::kFGTA,    //
    TrapRegisters::kFGCL,    // 15 bits last of the set by itself in a lone 32 bit reg
    TrapRegisters::kFTAL,    //
    TrapRegisters::kADCMSK,  //
    TrapRegisters::kADCINB,  //
    TrapRegisters::kADCDAC,  // 5 bit in the middle of a register and the last of the set.
    TrapRegisters::kADCPAR,  //
    TrapRegisters::kIA3IRQC, // last 10 bit word
    TrapRegisters::kIRQHW3,  // then a 32 bit register again
    TrapRegisters::kIRQHL3,  // last 15 bit reg in a 32bit word
    TrapRegisters::kCTGDINI, // first
    TrapRegisters::kCTGCTRL, // second
    TrapRegisters::kMEMRW,   // lone 32 bit reg
    TrapRegisters::kMEMCOR,  // lone 16 bit reg
    TrapRegisters::kDMDELA,  // first 10 after a 16
    TrapRegisters::kDMDELS,  // middle 10 in a 32 bit word (30 used)
    TrapRegisters::kNMOD,    // last 10 bit in a 32 bit word
    TrapRegisters::kNDLY,    // 11 bit in the subsequent 32 bit word*/
    TrapRegisters::kPASACHM  // last register
  };
  TrapRegisters trapRegisters;
  // setup the resgisters we will look at chosen for various reasons, size, on the edges, changes of register bit size
  for (auto& reg : registerToLookAt) {
    registersOfInterest.push_back(reg);
    // add a zero value, value of 1, mid point and its max.
    auto max = trapconfig1->getRegisterMax(reg);
    std::array<uint32_t, 4> regvalues;
    regvalues[0] = 0;
    regvalues[1] = 1;
    regvalues[2] = max / 2;
    regvalues[3] = max;
    registervalues.emplace_back(regtotest(reg, regvalues));
  }

  // walk through map and test
  int count = 0;
  for (auto& mcm : mcmids) {
    trapregCheck(trapconfig1, mcm, count);
    ++count;
  }
  // all those values have been written so we can now try read those back in bulk
  std::array<uint32_t, constants::MAXMCMCOUNT> allregisterdata{0};  // data for a single register across all mcm
  std::array<uint32_t, TrapRegisters::kLastReg> allmcmregisters{0}; // data for a single mcm, all registers. std::memset(&allregisters->at(0), 0, sizeof(uint32_t));
  std::unique_ptr<TrapConfigEvent> trapconfig(new TrapConfigEvent());
  // loop over all mcm's we want to test and add them to the trapconfig via setregister, this allows us to also view them before anything happens for debugging purposees.
  for (auto& mcmidx : mcmids) {
    uint32_t retval = trapconfig->setRegisterValue(0, 0, mcmidx);
  }

  int registerindex = 0;
  uint32_t registerreadvalue = 0;
  for (int valuecount = 0; valuecount < 4; ++valuecount) {
    // loop over each of the value sets 0,1,mid, max
    std::vector<uint32_t> addressesseen;
    std::map<uint32_t, uint32_t> setValues;
    // loop through all registers for pulling out the valuecount instance of each, put data in and then check against the regall;
    // its sorted so we know a key will come 4 times sequentially, and use valuecount to denote which value we are working with.
    uint32_t fgcal_value = registervalues[TrapRegisters::kFGCL + valuecount].mValue[valuecount];
    for (int regtest = 0; regtest < registervalues.size(); ++regtest) { // loop over the 4th elements to pull out like values, 0,1,max/2,max
      registerreadvalue = registervalues[regtest].mValue[valuecount];
      uint32_t index = registervalues[regtest].mRegIdx;
      for (auto& mcmidx : mcmids) {
        std::string name = trapconfig->getRegisterName(index);
        uint32_t retval = trapconfig->setRegisterValue(registerreadvalue, index, mcmidx);
        if (!retval) {
          // index is invalid or mcmidx is invalid
          continue;
        }
      }
    }
    // all registers now written to ccdbconfig for this value set
    int elemcount = 0;
    // now check the values that were written
    // trapconfig->getAll(*allregisters);
    // check done in next double for loop
    uint32_t value = 0;
    int registersofinterestindex = 0;

    for (auto& mcmidx : mcmids) {
      if (mcmidx < 0 || mcmidx > constants::MAXCHAMBER)
        continue;
      // pull out the values on for a specific mcm
      trapconfig->getAllRegisters(mcmidx, allmcmregisters);
      // check the values are set correctly
      int regcount = 0;
      for (auto& reg : registersOfInterest) {
        // loop over all registers for the given mcmq
        auto value = registervalues[regcount].mValue[valuecount];
        // value = registervalues[regcount + valuecount].mValue[1];
        if (reg == TrapRegisters::kFGA5) {
          int count = 0;
          for (auto& mcmreg : allmcmregisters) {
          }
        }
        BOOST_CHECK_EQUAL(allmcmregisters[reg], value);
        ++regcount;
      }
    }
  }
}

} // namespace o2::trd
