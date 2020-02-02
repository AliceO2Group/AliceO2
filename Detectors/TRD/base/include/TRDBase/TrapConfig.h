// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRAPCONFIG_H
#define O2_TRAPCONFIG_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>

// Configuration of the TRD Tracklet Processor
// (TRD Front-End Electronics)
// There is a manual to describe all the internals of the electronics.
// A very detailed manual, will try put it in the source code.
// TRAP registers
// TRAP data memory (DMEM)
//
using namespace std;

namespace o2
{
namespace trd
{

class TrapConfig
{
 public:
  TrapConfig();
  ~TrapConfig();

  // allocation
  enum Alloc_t {
    kAllocNone,
    kAllocGlobal,
    kAllocByDetector,
    kAllocByHC,
    kAllocByMCM,
    kAllocByMergerType,
    kAllocByLayer,
    kAllocByMCMinSM,
    kAllocLast
  }; // possible granularities for allocation
     // common to registers and DMEM

  // registers
  enum TrapReg_t { kSML0,
                   kSML1,
                   kSML2,
                   kSMMODE,
                   kSMCMD,
                   kNITM0,
                   kNITM1,
                   kNITM2,
                   kNIP4D,
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
                   kC08CPU0,
                   kC09CPU0,
                   kC10CPU0,
                   kC11CPU0,
                   kC12CPUA,
                   kC13CPUA,
                   kC14CPUA,
                   kC15CPUA,
                   kC08CPU1,
                   kC09CPU1,
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
                   kNMOD,
                   kNDLY,
                   kNED,
                   kNTRO,
                   kNRRO,
                   kNES,
                   kNTP,
                   kNBND,
                   kNP0,
                   kNP1,
                   kNP2,
                   kNP3,
                   kNCUT,
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
                   kEBIS,
                   kEBIT,
                   kEBIL,
                   kEBIN,
                   kFLBY,
                   kFPBY,
                   kFGBY,
                   kFTBY,
                   kFCBY,
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
                   kADCMSK,
                   kADCINB,
                   kADCDAC,
                   kADCPAR,
                   kADCTST,
                   kSADCAZ,
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
                   kPASADEL,
                   kPASAPHA,
                   kPASAPRA,
                   kPASADAC,
                   kPASACHM,
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
                   kMEMRW,
                   kMEMCOR,
                   kDMDELA,
                   kDMDELS,
                   kLastReg }; // enum of all TRAP registers, to be used for access to them
  static const int mlastAlloc = kAllocLast;
  bool setTrapRegAlloc(TrapReg_t reg, Alloc_t mode) { return mRegisterValue[reg].allocate(mode); }
  bool setTrapReg(TrapReg_t reg, int value, int det);
  bool setTrapReg(TrapReg_t reg, int value, int det, int rob, int mcm);

  int getTrapReg(TrapReg_t reg, int det = -1, int rob = -1, int mcm = -1);

  void resetRegs();
  std::string getConfigVersion() { return mTrapConfigVersion; }
  std::string getConfigName() { return mTrapConfigName; }
  void setConfigVersion(std::string version) { mTrapConfigVersion = version; }
  void setConfigName(std::string name) { mTrapConfigName = name; }
  // data memory (DMEM)
  bool setDmemAlloc(int addr, Alloc_t mode);
  bool setDmem(int addr, unsigned int value, int det);
  bool setDmem(int addr, unsigned int value, int det, int rob, int mcm);
  bool setDmem(int addr, int value) { return setDmem(addr, (unsigned int)value); }
  bool setDmem(int addr, int value, int det, int rob, int mcm) { return setDmem(addr, (unsigned int)value, det, rob, mcm); }

  unsigned int getDmemUnsigned(int addr, int det, int rob, int mcm);

  void resetDmem();

  // access by 16-bit address
  unsigned int peek(int addr, int det, int rob, int mcm);
  bool poke(int addr, unsigned int value, int det, int rob, int mcm);

  // helper methods
  std::string getRegName(TrapReg_t reg) { return ((reg >= 0) && (reg < kLastReg)) ? mRegisterValue[reg].getName() : ""; }
  unsigned short getRegAddress(TrapReg_t reg) { return ((reg >= 0) && (reg < kLastReg)) ? mRegisterValue[reg].getAddr() : 0; }
  unsigned short getRegNBits(TrapReg_t reg) { return ((reg >= 0) && (reg < kLastReg)) ? mRegisterValue[reg].getNbits() : 0; }
  unsigned int getRegResetValue(TrapReg_t reg) { return ((reg >= 0) && (reg < kLastReg)) ? mRegisterValue[reg].getResetValue() : 0; }

  TrapReg_t getRegByAddress(int address);

  bool printTrapReg(TrapReg_t reg, int det = -1, int rob = -1, int mcm = -1);
  bool printTrapAddr(int addr, int det = -1, int rob = -1, int mcm = -1);

  void printMemDatx(ostream& os, int addr);
  void printMemDatx(ostream& os, int addr, int det, int rob, int mcm);
  void printMemDatx(ostream& os, TrapReg_t reg);
  void printMemDatx(ostream& os, TrapReg_t reg, int det, int rob, int mcm);
  void printDatx(ostream& os, unsigned int addr, unsigned int data, int rob, int mcm);

  void printVerify(ostream& os, int det, int rob, int mcm);

  static const int mgkDmemStartAddress = 0xc000; // start address in TRAP GIO
  static const int mgkDmemWords = 0x400;         // number of words in DMEM

  static const int mgkImemStartAddress = 0xe000; // start address in TRAP GIO
  static const int mgkImemWords = 0x1000;        // number of words in IMEM

  static const int mgkDbankStartAddress = 0xf000; // start address in TRAP GIO
  static const int mgkDbankWords = 0x0100;        // number of words in DBANK

  class TrapValue
  {
   public:
    TrapValue();
    ~TrapValue() = default;

    bool allocate(Alloc_t mode);
    bool allocatei(int mode);
    std::string getName() { return mName; }
    static const std::array<int, TrapConfig::mlastAlloc> mgkSize; //= {0, 1, 540, 1080, 8 * 18 * 540, 4, 6, 8 * 18 * 30};
    //static const std::array<int,TrapConfig::mlastAlloc> mgkSize; // required array dimension for different allocation modes
    //this is used purely for copying data from run2 ocdb to run3 ccdb.
    void setDataFromRun2(int value, int valid, int index)
    {
      if (index < mData.size()) {
        mData[index] = value;
        mValid[index] = valid;
      } else
        cout << "attempt to write data outside array with size : " << mData.size() << "and index of :" << index;
    }
    int getDataSize() { return mData.size(); }

   protected:
    bool setData(unsigned int value);
    bool setData(unsigned int value, int det);
    bool setData(unsigned int value, int det, int rob, int mcm);

    unsigned int getData(int det, int rob, int mcm);

    int getIdx(int det, int rob, int mcm);

   private:
    TrapValue(const TrapValue& rhs);            // not implemented
    TrapValue& operator=(const TrapValue& rhs); // not implemented

    Alloc_t mAllocMode;              // allocation mode
    std::vector<unsigned int> mData; //[mSize] data array
    std::vector<bool> mValid;        //[mSize] valid flag
    std::string mName;
  };

  class TrapRegister : public TrapValue
  {
   public:
    TrapRegister();
    ~TrapRegister();

    void init(const char* name, int addr, int nBits, int resetValue);
    void initfromrun2(const char* name, int addr, int nBits, int resetValue);
    void reset() { setData(mResetValue); }

    bool setValue(int value, int det) { return setData(value, det); }
    bool setValue(int value, int det, int rob, int mcm) { return setData(value, det, rob, mcm); }

    int getValue(int det, int rob, int mcm) { return getData(det, rob, mcm); }
    std::string getName() { return mName; }
    unsigned short getAddr() { return mAddr; }
    unsigned short getNbits() { return mNbits; }
    unsigned int getResetValue() { return mResetValue; }

   protected:
    TrapRegister(const TrapRegister& rhs);
    TrapRegister& operator=(const TrapRegister& rhs);

    // fixed properties of the register
    // which do not need to be stored
    std::string mName;        //! Name of the register
    unsigned short mAddr;     //! Address in GIO of TRAP
    unsigned short mNbits;    //! Number of bits, from 1 to 32
    unsigned int mResetValue; //! reset value
  };

  class TrapDmemWord : public TrapValue
  {
   public:
    TrapDmemWord() : TrapValue(), mName(""), mAddr(0) {}
    ~TrapDmemWord() = default;

    void reset() { setData(0); }

    bool setValue(unsigned int value, int det) { return setData(value, det); }
    bool setValue(unsigned int value, int det, int rob, int mcm) { return setData(value, det, rob, mcm); }

    unsigned int getValue(int det, int rob, int mcm) { return getData(det, rob, mcm); }

    void setAddress(unsigned short addr)
    {
      mAddr = addr;
      std::stringstream mNamestream;
      mNamestream << "DMEM 0x" << hex << mAddr;
      mName = mNamestream.str();
    }
    std::string getName() { return mName; }

   protected:
    TrapDmemWord(const TrapDmemWord& rhs);            // not implemented
    TrapDmemWord& operator=(const TrapDmemWord& rhs); // not implemented

    std::string mName;
    unsigned short mAddr; //! address
  };

  // protected:
  void initRegs();

  // configuration registers
  std::array<TrapRegister, kLastReg> mRegisterValue{}; // array of TRAP register values in use

  // DMEM
  std::array<TrapDmemWord, mgkDmemWords> mDmem{}; // TRAP data memory

  static const int mgkMcmlistSize = 256; // list of MCMs to which a value has to be written

  static bool mgRegAddressMapInitialized;
  std::array<TrapReg_t, 0x400 + 0x200 + 0x4> mgRegAddressMap{};

  const std::array<int, 3> mgkRegisterAddressBlockStart = {0x0a00, 0x3000, 0xd000};
  const std::array<int, 3> mgkRegisterAddressBlockSize = {0x0400, 0x0200, 0x0004};
  std::string mTrapConfigName;
  std::string mTrapConfigVersion;

 private:
  TrapConfig& operator=(const TrapConfig& rhs); // not implemented
  TrapConfig(const TrapConfig& cfg);            // not implemented
};
} //namespace trd
} //namespace o2
#endif
