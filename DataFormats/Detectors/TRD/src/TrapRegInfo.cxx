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

#include "DataFormatsTRD/TrapRegisters.h"
#include "DataFormatsTRD/TrapRegInfo.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "DataFormatsTRD/RawData.h"

#include <fairlogger/Logger.h>

#include <array>
#include <map>

using namespace o2::trd;

TrapRegInfo::TrapRegInfo(const std::string& name, int addr, int nBits, int base, int wordoffset, bool ignorechange, uint32_t max)
{
  init(name, addr, nBits, base, wordoffset, ignorechange, max);
}

TrapRegInfo::~TrapRegInfo() = default;

void TrapRegInfo::init(const std::string& name, int addr, int nbits, int base, int wordnumber, bool ignorechange, uint32_t max)
{
  // initialise a TRAP register information
  // uint32_t mNbits,mBase,mWordNumber,mShift,mMask,mMax;
  // see headerfile for exact definition, WordNumber: the sequence of register in a block of data, like registers.
  int bitoffset;
  int gapbits = 0;
  int bitwordoffset32; // the beginning of the 32 bit word containing this reg
  int packedwordsize = 30;
  /*if(name == "ADCMSK"){
     LOGP(info, " TrapRegInfo : before {} with nbits={} addr {:08x} mask {:04x} word number {} and baseword {} max {} ", name, nbits, addr, 0, wordnumber, base, max);
  }*/
  if (addr != 0) {
    mAddr = addr;
    mName = name;
    mNbits = nbits;
    mBase = base;
    mWordNumber = wordnumber;
    mMax = pow(2, max) - 1; // this can be different from the mask, not sure why.
    packedwordsize = 30;
    if (mNbits > 30 || mNbits == 16 || mNbits == 4) { // these are 32 bit aligned the rest are 30 bit aligned.
      packedwordsize = 32;
      if (mNbits == 31) {
        gapbits = 1;
      }
    }
    mDataWordNumber = mWordNumber * (nbits + gapbits) / packedwordsize; // the 32 bit word offset to the word containing the register in question
    bitoffset = mWordNumber * (nbits)-mDataWordNumber * packedwordsize; // the offset bit with in that word for the start of this register
    mMask = (1 << mNbits) - 1;                                          // e.g. 5 = 0x1f 10=0x3ff
    mShift = 32 - bitoffset - mNbits;                                   // 32-remainder= the right hand side of bits that need to be dropped.
    if (mNbits == 32) {
      mShift = 0;
      mMask = 0xffffffff;
    }
    if (mNbits == 31) {
      mShift = 1;
    }
  } else {
    LOGP(warn, "Initialising an TRAP register with address of {:08x} ", addr);
    mAddr = 0;
    mName = "";
    mNbits = 0;
    mBase = 0;
    mWordNumber = 0;
    mMax = 0;
    mShift = 0;
  }
  /*if(name == "ADCMSK"){
    LOGP(info, " TrapRegInfo end : {} with nbits={} addr {:08x} mask {:04x} word number {} and baseword {} max {} shift {} ", getName(), getNbits(), getAddr(), getMask(), getWordNumber(), getBase(), getMax(), getShift());
  }
  if(mShift==42) {
    LOGP(info, "Shift is 42 TrapRegInfo end : {} with nbits={} addr {:08x} mask {:04x} word number {} and baseword {} max {} shift {} ", getName(), getNbits(), getAddr(), getMask(), getWordNumber(), getBase(), getMax(), getShift());
  }
  if(mMask==0x1f) {
    LOGP(info, "Shift is 0x1f TrapRegInfo end : {} with nbits={} addr {:08x} mask {:04x} word number {} and baseword {} max {} shift {} ", getName(), getNbits(), getAddr(), getMask(), getWordNumber(), getBase(), getMax(), getShift());
  }*/
}

void TrapRegInfo::logTrapRegInfo()
{
  LOGP(info, " TrapRegInfo : {} with nbits={} addr {:08x} mask {:04x} word number {} and baseword {} max {} ", getName(), getNbits(), getAddr(), getMask(), getWordNumber(), getBase(), getMax());
}
