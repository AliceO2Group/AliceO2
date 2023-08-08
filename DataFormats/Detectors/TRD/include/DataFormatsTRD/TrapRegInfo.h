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

#ifndef O2_TRDTRAPREGINFO_H
#define O2_TRDTRAPREGINFO_H

#include "CommonDataFormat/InteractionRecord.h"
#include <fairlogger/Logger.h>

#include "DataFormatsTRD/Constants.h"

namespace o2::trd
{

class TrapRegInfo
{
  // class to store the parameters associated with a register
  // some info related to the hardware, some to the packing/unpacking we do here.
 public:
  TrapRegInfo() = default;
  TrapRegInfo(const std::string& name, int addr, int nBits, int base, int wordoffset, bool ignorechange, uint32_t max);

  ~TrapRegInfo();

  void init(const std::string& name, int addr, int nBits, int base, int wordnumber, bool ignorechange, uint32_t max);

  // getters and setters, this is just a storage class.
  const std::string getName() const { return mName; }
  const unsigned short getAddr() const { return mAddr; }
  const unsigned short getNbits() const { return mNbits; }
  const unsigned int getBase() const { return mBase; }
  const unsigned int getWordNumber() const { return mWordNumber; }
  const unsigned int getDataWordNumber() const { return mDataWordNumber; }
  const unsigned int getShift() const { return mShift; }
  const uint32_t getMask() const { return mMask; }
  const uint32_t getMax() const { return mMax; }
  bool getIgnoreChange() { return mIgnoreChange; }

  void setName(const std::string name) { mName = name; }
  void setAddr(const uint32_t addr) { mAddr = addr; }
  void setNbits(const uint32_t bits) { mNbits = bits; }
  void setBase(const uint32_t base) { mBase = base; }
  void setWordNumber(const uint32_t wordnum) { mWordNumber = wordnum; }
  void setDataWordNumber(const uint32_t datawordnum) { mDataWordNumber = datawordnum; }
  void setShift(const uint32_t shift) { mShift = shift; }
  void setMask(const uint32_t mask) { mMask = mask; }
  void setMax(uint32_t max) { mMax = pow(2, max) - 1; }
  void setIgnoreChange(const uint32_t ignorechange) { mIgnoreChange = ignorechange; }

  void logTrapRegInfo(); // output the contents to log info

 private:
  // TrapRegInfo(const TrapRegInfo& rhs);
  // TrapRegInfo& operator=(const TrapRegInfo& rhs);

  // fixed properties of the register
  // which do not need to be stored on a per mcm basis
  // nominal example :
  //  3         2         1         0
  // 10987654321098765432109876543210
  // ----------------------xxxxx-----
  // TPL01 Nbits=5; mBase=0;WordNumber=0;DataWordNumber=0;Mask=0x170;Shift=5;
  //  3         2         1         0
  // 10987654321098765432109876543210
  // --xxxxxxxxxx--------------------
  // FGF10 Nbits=10; mBase=27;WordNumber=10;DataWordNumber=10;Mask=0x3ff00000;Shift=20;
  // mTrapRegisters[kFGF10].init("FGF10", 0x308A, 10, 27, 10, false, 10);
  std::string mName;        //!< Name of the register
  uint16_t mAddr;           //!< Address in GIO of TRAP
  uint16_t mNbits;          //!< Number of bits, from 1 to 32
  uint32_t mBase;           //!< base of this registers block, i.e. TFL?? will have 0 and is in the range [0,kTrapRegistersSize]
  uint32_t mWordNumber;     //!< word number offset, of size Nbits, in the block of registers
  uint32_t mDataWordNumber; //!< offset, into "compressed" 32 bit words for the 32 bit word containing this register, offset from the base;
  uint32_t mMask;           //!< mask to extract register from the word identified by WordNumber
  uint32_t mMax;            //!< max is not the same as the mask, some values come in as 15 bit mask but are only 12 or 13 bits for max, IRQ values for example
  uint32_t mShift;          //!< shift to extract the register
  bool mIgnoreChange;       //!< we are not concerned with this value changing for purposes of the differential comparison.
  ClassDefNV(TrapRegInfo, 1);
};

} // namespace o2::trd

#endif
