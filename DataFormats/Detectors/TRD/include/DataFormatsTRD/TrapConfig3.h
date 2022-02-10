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

#ifndef O2_TRDTRAPCONFIG3_H
#define O2_TRDTRAPCONFIG3_H

#include <fairlogger/Logger.h>
#include "DataFormatsTRD/Constants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <vector>
#include <memory>
#include <map>
#include <bitset>

#include "Rtypes.h"
#include "TH2F.h"

namespace o2::trd
{

static const int kTrapRegistersSize = 151; // size of our compressed trap registers in units of 32 bits

// enum of all TRAP registers, to be used for access to them
// registers in the order they come in.
enum TrapReg {

  kTPL00,
  kTPL01,
  kTPL02,
  kTPL03,
  kTPL04,
  kTPL05,
  kTPL06,
  kTPL07,
  kTPL08,
  kTPL09,
  kTPL0A,
  kTPL0B,
  kTPL0C,
  kTPL0D,
  kTPL0E,
  kTPL0F,
  kTPL10,
  kTPL11,
  kTPL12,
  kTPL13,
  kTPL14,
  kTPL15,
  kTPL16,
  kTPL17,
  kTPL18,
  kTPL19,
  kTPL1A,
  kTPL1B,
  kTPL1C,
  kTPL1D,
  kTPL1E,
  kTPL1F,
  kTPL20,
  kTPL21,
  kTPL22,
  kTPL23,
  kTPL24,
  kTPL25,
  kTPL26,
  kTPL27,
  kTPL28,
  kTPL29,
  kTPL2A,
  kTPL2B,
  kTPL2C,
  kTPL2D,
  kTPL2E,
  kTPL2F,
  kTPL30,
  kTPL31,
  kTPL32,
  kTPL33,
  kTPL34,
  kTPL35,
  kTPL36,
  kTPL37,
  kTPL38,
  kTPL39,
  kTPL3A,
  kTPL3B,
  kTPL3C,
  kTPL3D,
  kTPL3E,
  kTPL3F,
  kTPL40,
  kTPL41,
  kTPL42,
  kTPL43,
  kTPL44,
  kTPL45,
  kTPL46,
  kTPL47,
  kTPL48,
  kTPL49,
  kTPL4A,
  kTPL4B,
  kTPL4C,
  kTPL4D,
  kTPL4E,
  kTPL4F,
  kTPL50,
  kTPL51,
  kTPL52,
  kTPL53,
  kTPL54,
  kTPL55,
  kTPL56,
  kTPL57,
  kTPL58,
  kTPL59,
  kTPL5A,
  kTPL5B,
  kTPL5C,
  kTPL5D,
  kTPL5E,
  kTPL5F,
  kTPL60,
  kTPL61,
  kTPL62,
  kTPL63,
  kTPL64,
  kTPL65,
  kTPL66,
  kTPL67,
  kTPL68,
  kTPL69,
  kTPL6A,
  kTPL6B,
  kTPL6C,
  kTPL6D,
  kTPL6E,
  kTPL6F,
  kTPL70,
  kTPL71,
  kTPL72,
  kTPL73,
  kTPL74,
  kTPL75,
  kTPL76,
  kTPL77,
  kTPL78,
  kTPL79,
  kTPL7A,
  kTPL7B,
  kTPL7C,
  kTPL7D,
  kTPL7E,
  kTPL7F,
  kFGA0,
  kFGA1,
  kFGA2,
  kFGA3,
  kFGA4,
  kFGA5,
  kFGA6,
  kFGA7,
  kFGA8,
  kFGA9,
  kFGA10,
  kFGA11,
  kFGA12,
  kFGA13,
  kFGA14,
  kFGA15,
  kFGA16,
  kFGA17,
  kFGA18,
  kFGA19,
  kFGA20,
  kFGF0,
  kFGF1,
  kFGF2,
  kFGF3,
  kFGF4,
  kFGF5,
  kFGF6,
  kFGF7,
  kFGF8,
  kFGF9,
  kFGF10,
  kFGF11,
  kFGF12,
  kFGF13,
  kFGF14,
  kFGF15,
  kFGF16,
  kFGF17,
  kFGF18,
  kFGF19,
  kFGF20,
  kCPU0CLK,
  kCPU1CLK,
  kCPU2CLK,
  kCPU3CLK,
  kNICLK,
  kFILCLK,
  kPRECLK,
  kADCEN,
  kNIODE,
  kNIOCE,
  kNIIDE,
  kNIICE,
  kEBIS,
  kEBIT,
  kEBIL,
  kTPVT,
  kTPVBY,
  kTPCT,
  kTPCL,
  kTPCBY,
  kTPD,
  kTPCI0,
  kTPCI1,
  kTPCI2,
  kTPCI3,
  kEBIN,
  kFLBY,
  kFPBY,
  kFGBY,
  kFTBY,
  kFCBY,
  kFLL00,
  kFLL01,
  kFLL02,
  kFLL03,
  kFLL04,
  kFLL05,
  kFLL06,
  kFLL07,
  kFLL08,
  kFLL09,
  kFLL0A,
  kFLL0B,
  kFLL0C,
  kFLL0D,
  kFLL0E,
  kFLL0F,
  kFLL10,
  kFLL11,
  kFLL12,
  kFLL13,
  kFLL14,
  kFLL15,
  kFLL16,
  kFLL17,
  kFLL18,
  kFLL19,
  kFLL1A,
  kFLL1B,
  kFLL1C,
  kFLL1D,
  kFLL1E,
  kFLL1F,
  kFLL20,
  kFLL21,
  kFLL22,
  kFLL23,
  kFLL24,
  kFLL25,
  kFLL26,
  kFLL27,
  kFLL28,
  kFLL29,
  kFLL2A,
  kFLL2B,
  kFLL2C,
  kFLL2D,
  kFLL2E,
  kFLL2F,
  kFLL30,
  kFLL31,
  kFLL32,
  kFLL33,
  kFLL34,
  kFLL35,
  kFLL36,
  kFLL37,
  kFLL38,
  kFLL39,
  kFLL3A,
  kFLL3B,
  kFLL3C,
  kFLL3D,
  kFLL3E,
  kFLL3F,
  kTPPT0,
  kTPFS,
  kTPFE,
  kTPPGR,
  kTPPAE,
  kTPQS0,
  kTPQE0,
  kTPQS1,
  kTPQE1,
  kEBD,
  kEBAQA,
  kEBSIA,
  kEBSF,
  kEBSIM,
  kEBPP,
  kEBPC,
  kFPTC,
  kFPNP,
  kFPCL,
  kFGTA,
  kFGTB,
  kFGCL,
  kFTAL,
  kFTLL,
  kFTLS,
  kFCW1,
  kFCW2,
  kFCW3,
  kFCW4,
  kFCW5,
  kTPFP,
  kTPHT,
  kADCMSK,
  kADCINB,
  kADCDAC,
  kADCPAR,
  kADCTST,
  kSADCAZ,
  kPASADEL,
  kPASAPHA,
  kPASAPRA,
  kPASADAC,
  kPASASTL,
  kPASAPR1,
  kPASAPR0,
  kSADCTRG,
  kSADCRUN,
  kSADCPWR,
  kL0TSIM,
  kSADCEC,
  kSADCMC,
  kSADCOC,
  kSADCGTB,
  kSEBDEN,
  kSEBDOU,
  kSML0,
  kSML1,
  kSML2,
  kSMMODE,
  kNITM0,
  kNITM1,
  kNITM2,
  kNIP4D,
  kARBTIM,
  kIA0IRQ0,
  kIA0IRQ1,
  kIA0IRQ2,
  kIA0IRQ3,
  kIA0IRQ4,
  kIA0IRQ5,
  kIA0IRQ6,
  kIA0IRQ7,
  kIA0IRQ8,
  kIA0IRQ9,
  kIA0IRQA,
  kIA0IRQB,
  kIA0IRQC,
  kIRQSW0,
  kIRQHW0,
  kIRQHL0,
  kIA1IRQ0,
  kIA1IRQ1,
  kIA1IRQ2,
  kIA1IRQ3,
  kIA1IRQ4,
  kIA1IRQ5,
  kIA1IRQ6,
  kIA1IRQ7,
  kIA1IRQ8,
  kIA1IRQ9,
  kIA1IRQA,
  kIA1IRQB,
  kIA1IRQC,
  kIRQSW1,
  kIRQHW1,
  kIRQHL1,
  kIA2IRQ0,
  kIA2IRQ1,
  kIA2IRQ2,
  kIA2IRQ3,
  kIA2IRQ4,
  kIA2IRQ5,
  kIA2IRQ6,
  kIA2IRQ7,
  kIA2IRQ8,
  kIA2IRQ9,
  kIA2IRQA,
  kIA2IRQB,
  kIA2IRQC,
  kIRQSW2,
  kIRQHW2,
  kIRQHL2,
  kIA3IRQ0,
  kIA3IRQ1,
  kIA3IRQ2,
  kIA3IRQ3,
  kIA3IRQ4,
  kIA3IRQ5,
  kIA3IRQ6,
  kIA3IRQ7,
  kIA3IRQ8,
  kIA3IRQ9,
  kIA3IRQA,
  kIA3IRQB,
  kIA3IRQC,
  kIRQSW3,
  kIRQHW3,
  kIRQHL3,
  kCTGDINI,
  kCTGCTRL,
  kMEMRW,
  kMEMCOR,
  kDMDELA,
  kDMDELS,
  kNMOD,
  kNDLY,
  kNED,
  kNTRO,
  kNRRO,
  kNBND,
  kNP0,
  kNP1,
  kNP2,
  kNP3,
  kC08CPU0,
  kC09CPU0, // will be q2 pid window settings.
  kC10CPU0,
  kC11CPU0,
  kC12CPUA,
  kC13CPUA,
  kC14CPUA,
  kC15CPUA,
  kC08CPU1,
  kC09CPU1, // version of trap code.
  kC10CPU1,
  kC11CPU1,
  kC08CPU2,
  kC09CPU2,
  kC10CPU2,
  kC11CPU2,
  kC08CPU3,
  kC09CPU3,
  kC10CPU3,
  kC11CPU3,
  kNES,
  kNTP,
  kNCUT,
  kPASACHM,
  kLastReg
};

class TrapRegInfo
{
  // class to store the parameters associated with a register
  // some info related to the hardware, some to the packing/unpacking we do here.
 public:
  TrapRegInfo();
  TrapRegInfo(const std::string& name, int addr, int nBits, int base, int wordoffset, bool ignorechange, uint32_t max);

  ~TrapRegInfo();

  void init(const std::string& name, int addr, int nBits, int base, int wordnumber, bool ignorechange, uint32_t max);

  // getters and setters, this is just a storage class.
  std::string getName() { return mName; }
  unsigned short getAddr() { return mAddr; }
  unsigned short getNbits() { return mNbits; }
  unsigned int getBase() { return mBase; }
  unsigned int getWordNumber() { return mWordNumber; }
  unsigned int getDataWordNumber() { return mDataWordNumber; }
  unsigned int getShift() { return mShift; }
  uint32_t getMask() { return mMask; }
  uint32_t getMax() { return mMax; }
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
  TrapRegInfo(const TrapRegInfo& rhs);
  TrapRegInfo& operator=(const TrapRegInfo& rhs);

  // fixed properties of the register
  // which do not need to be stored on a per mcm basis
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

class TrapConfig3
{
  // class that is actually stored in ccdb.
  // it holds a compressed version of what comes in the config events
 public:
  TrapConfig3();
  ~TrapConfig3() = default;

  // get a config register value by index, addr, and name, via mcm index
  uint32_t getRegisterValue(uint32_t regidx, int mcmidx);
  bool setRegisterValue(uint32_t data, uint32_t regidx, int mcmidx);
  uint32_t getRegisterValueByIdx(uint32_t regidx, int mcmidx);
  bool setRegisterValueByIdx(uint32_t data, uint32_t regidx, int mcmidx);
  uint32_t getRegisterValueByAddr(uint32_t addr, int mcmidx);
  bool setRegisterValueByAddr(uint32_t data, uint32_t addr, int mcmidx);
  uint32_t getRegisterValueByName(const std::string& name, int mcmidx);
  bool setRegisterValueByName(uint32_t data, const std::string& regname, int mcmidx);

  // get a config register value by index, addr, and name, via sector/stack/layer/rob/mcm
  // no setters for sector/stack/layer/rob/mcm as writing, we only write from retrieved config events.
  uint32_t getRegisterValueByIdx(uint32_t regix, int sector, int stack, int layer, int rob, int mcm);
  uint32_t getRegisterValueByAddr(uint32_t regaddr, int sector, int stack, int layer, int rob, int mcm);
  uint32_t getRegisterValueByIdx(uint32_t regix, int detector, int rob, int mcm);
  uint32_t getRegisterValueByAddr(uint32_t regaddr, int detector, int rob, int mcm);

  // get a registers name by addres and index;
  std::string getRegNameByAddr(uint16_t addr);
  std::string getRegNameByIdx(unsigned int regidx);

  // get a registers index (enum value) by address or name
  int32_t getRegIndexByAddr(unsigned int addr);
  int32_t getRegIndexByName(const std::string& name);

  // get a registers address by index or name
  int32_t getRegAddrByIdx(unsigned int regidx);
  int32_t getRegAddrByName(const std::string& name);

  // retrieve the trap register info for a particular register, no need to index into det/rob/mcm as these are constant over a config
  o2::trd::TrapRegInfo* getTrapRegInfoByAddr(uint32_t addr);
  o2::trd::TrapRegInfo* getTrapRegInfoByIdx(uint32_t idx);

  // get the indenfiying characteristics of a register given its address.
  void getRegisterByAddr(uint32_t registeraddr, std::string& regname, int32_t& newregidx, int32_t& numberbits);

  // return all the registers for a particular mcm
  void getAllRegisters(int mcmidx, std::array<uint32_t, kLastReg>& registers);

  // return all the mcm values for a particular register index
  void getAllMCMByIndex(int regindex, std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT>& registers);

  // return all the mcm values for a particular register name
  void getAllMCMByName(std::string registername, std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT>& mcms);

  // return all the mcm for a particular register address
  void getAllMCMByAddress(int registeraddress, std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT>& mcms);

  // return all the config data in an unpacked array.
  void getAll(std::array<uint32_t, kLastReg * o2::trd::constants::MAXMCMCOUNT>& configdata);

  // return the full unpacked config event.
  bool printRegister(int addr, int det, int rob, int mcm);
  bool printRegister(TrapRegInfo* reg, int det, int rob, int mcm);

  const std::array<o2::trd::TrapRegInfo, kLastReg>& getTrapRegisters() { return mTrapRegisters; }

  // population pending
  void setConfigVersion(uint32_t version) { mTrapConfigVersion = version; }           // these must some how be gotten from git or wingdb.
  void setConfigNumber(uint32_t number) { mTrapConfigNumber = number; }               // these must be gotten from git or wingdb.
  void setConfigSavedVersion(uint16_t version) { mTrapConfigSavedVersion = version; } // the version that is saved, for the ability to later save the config differently.
  uint32_t getConfigVersion() { return mTrapConfigVersion; }                          // these must some how be gotten from git or wingdb.
  uint32_t getConfigName() { return mTrapConfigNumber; }                              // these must be gotten from git or wingdb.
  uint16_t getConfigSavedVersion() { return mTrapConfigSavedVersion; }                // the version that is saved, for the ability to later save the config differently.

  inline bool operator==(const TrapConfig3& other);

  // for compliance with the same interface to o2::trd::TrapConfig and its run1/2 inherited interface:
  // only these 2 are used in the simulations.
  uint32_t getDmemUnsigned(uint32_t address, int detector, int rob, int mcm);
  uint32_t getTrapReg(uint32_t index, int detector, int rob, int mcm);

 private:
  std::array<o2::trd::TrapRegInfo, kLastReg> mTrapRegisters;                              //!< store of layout of each block of mTrapRegisterSize, populated via initialiseRegisters
  std::bitset<o2::trd::constants::MAXMCMCOUNT> mMCMDataHasChanged;                        //!< does the mcm actually contain data.
  std::array<uint32_t, kTrapRegistersSize * o2::trd::constants::MAXMCMCOUNT> mConfigData; //!< one block of data per mcm.
  std::map<uint16_t, uint16_t> mTrapRegistersAddressIndexMap;                             //!< map of address into mTrapRegisters, populated at the end of initialiseRegisters
  void initialiseRegisters();

  uint32_t mTrapConfigNumber;       //!< the version number coming from the config that is written to the traps, from config register C09CPU01
  uint32_t mTrapConfigVersion;      //!< the version of the coming from the config that is written to the traps, from DigitHCHeader (svn version number)
  uint16_t mTrapConfigSavedVersion; //!< the version that is saved, for the ability to later save the config differently.
  ClassDefNV(TrapConfig3, 1);
};

}; // namespace o2::trd
#endif
