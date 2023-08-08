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

#ifndef O2_TRDTRAPREGISTERS_H
#define O2_TRDTRAPREGISTERS_H

#include "CommonDataFormat/InteractionRecord.h"
#include <fairlogger/Logger.h>
#include <cwchar>

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapRegInfo.h"
#include "TrapRegInfo.h"

namespace o2
{
namespace trd
{

static const int kTrapRegistersSize = 151; //!< size of our compressed trap registers in units of 32 bits
                                           // enum of all TRAP registers, to be used for access to them

class TrapRegisters
{
 public:
  // registers in the order they come in.
  // clang-format off
  enum TrapReg {
    kTPL00, kTPL01, kTPL02, kTPL03, kTPL04, kTPL05, kTPL06, kTPL07, kTPL08, kTPL09, kTPL0A,
    kTPL0B, kTPL0C, kTPL0D, kTPL0E, kTPL0F, kTPL10, kTPL11, kTPL12, kTPL13, kTPL14, kTPL15,
    kTPL16, kTPL17, kTPL18, kTPL19, kTPL1A, kTPL1B, kTPL1C, kTPL1D, kTPL1E, kTPL1F, kTPL20,
    kTPL21, kTPL22, kTPL23, kTPL24, kTPL25, kTPL26, kTPL27, kTPL28, kTPL29, kTPL2A, kTPL2B,
    kTPL2C, kTPL2D, kTPL2E, kTPL2F, kTPL30, kTPL31, kTPL32, kTPL33, kTPL34, kTPL35, kTPL36,
    kTPL37, kTPL38, kTPL39, kTPL3A, kTPL3B, kTPL3C, kTPL3D, kTPL3E, kTPL3F, kTPL40, kTPL41,
    kTPL42, kTPL43, kTPL44, kTPL45, kTPL46, kTPL47, kTPL48, kTPL49, kTPL4A, kTPL4B, kTPL4C,
    kTPL4D, kTPL4E, kTPL4F, kTPL50, kTPL51, kTPL52, kTPL53, kTPL54, kTPL55, kTPL56, kTPL57,
    kTPL58, kTPL59, kTPL5A, kTPL5B, kTPL5C, kTPL5D, kTPL5E, kTPL5F, kTPL60, kTPL61, kTPL62,
    kTPL63, kTPL64, kTPL65, kTPL66, kTPL67, kTPL68, kTPL69, kTPL6A, kTPL6B, kTPL6C, kTPL6D,
    kTPL6E, kTPL6F, kTPL70, kTPL71, kTPL72, kTPL73, kTPL74, kTPL75, kTPL76, kTPL77, kTPL78,
    kTPL79, kTPL7A, kTPL7B, kTPL7C, kTPL7D, kTPL7E, kTPL7F, kFGA0, kFGA1, kFGA2, kFGA3, kFGA4,
    kFGA5, kFGA6, kFGA7, kFGA8, kFGA9, kFGA10, kFGA11, kFGA12, kFGA13, kFGA14, kFGA15, kFGA16,
    kFGA17, kFGA18, kFGA19, kFGA20, kFGF0, kFGF1, kFGF2, kFGF3, kFGF4, kFGF5, kFGF6, kFGF7, kFGF8,
    kFGF9, kFGF10, kFGF11, kFGF12, kFGF13, kFGF14, kFGF15, kFGF16, kFGF17, kFGF18, kFGF19, kFGF20,
    kCPU0CLK, kCPU1CLK, kCPU2CLK, kCPU3CLK, kNICLK, kFILCLK, kPRECLK, kADCEN, kNIODE, kNIOCE,
    kNIIDE, kNIICE, kEBIS, kEBIT, kEBIL, kTPVT, kTPVBY, kTPCT, kTPCL, kTPCBY, kTPD, kTPCI0, kTPCI1,
    kTPCI2, kTPCI3, kEBIN, kFLBY, kFPBY, kFGBY, kFTBY, kFCBY, kFLL00, kFLL01, kFLL02, kFLL03, kFLL04,
    kFLL05, kFLL06, kFLL07, kFLL08, kFLL09, kFLL0A, kFLL0B, kFLL0C, kFLL0D, kFLL0E, kFLL0F, kFLL10,
    kFLL11, kFLL12, kFLL13, kFLL14, kFLL15, kFLL16, kFLL17, kFLL18, kFLL19, kFLL1A, kFLL1B, kFLL1C,
    kFLL1D, kFLL1E, kFLL1F, kFLL20, kFLL21, kFLL22, kFLL23, kFLL24, kFLL25, kFLL26, kFLL27, kFLL28,
    kFLL29, kFLL2A, kFLL2B, kFLL2C, kFLL2D, kFLL2E, kFLL2F, kFLL30, kFLL31, kFLL32, kFLL33, kFLL34,
    kFLL35, kFLL36, kFLL37, kFLL38, kFLL39, kFLL3A, kFLL3B, kFLL3C, kFLL3D, kFLL3E, kFLL3F, kTPPT0,
    kTPFS, kTPFE, kTPPGR, kTPPAE, kTPQS0, kTPQE0, kTPQS1, kTPQE1, kEBD, kEBAQA, kEBSIA, kEBSF,
    kEBSIM, kEBPP, kEBPC, kFPTC, kFPNP, kFPCL, kFGTA, kFGTB, kFGCL, kFTAL, kFTLL, kFTLS,
    kFCW1, kFCW2, kFCW3, kFCW4, kFCW5, kTPFP, kTPHT, kADCMSK, kADCINB, kADCDAC, kADCPAR, kADCTST,
    kSADCAZ, kPASADEL, kPASAPHA, kPASAPRA, kPASADAC, kPASASTL, kPASAPR1, kPASAPR0, kSADCTRG, kSADCRUN,
    kSADCPWR, kL0TSIM, kSADCEC, kSADCMC, kSADCOC, kSADCGTB, kSEBDEN, kSEBDOU, kSML0, kSML1, kSML2,
    kSMMODE, kNITM0, kNITM1, kNITM2, kNIP4D, kARBTIM, kIA0IRQ0, kIA0IRQ1, kIA0IRQ2, kIA0IRQ3, kIA0IRQ4,
    kIA0IRQ5, kIA0IRQ6, kIA0IRQ7, kIA0IRQ8, kIA0IRQ9, kIA0IRQA, kIA0IRQB, kIA0IRQC, kIRQSW0, kIRQHW0,
    kIRQHL0, kIA1IRQ0, kIA1IRQ1, kIA1IRQ2, kIA1IRQ3, kIA1IRQ4, kIA1IRQ5, kIA1IRQ6, kIA1IRQ7, kIA1IRQ8,
    kIA1IRQ9, kIA1IRQA, kIA1IRQB, kIA1IRQC, kIRQSW1, kIRQHW1, kIRQHL1, kIA2IRQ0, kIA2IRQ1, kIA2IRQ2,
    kIA2IRQ3, kIA2IRQ4, kIA2IRQ5, kIA2IRQ6, kIA2IRQ7, kIA2IRQ8, kIA2IRQ9, kIA2IRQA, kIA2IRQB, kIA2IRQC,
    kIRQSW2, kIRQHW2, kIRQHL2, kIA3IRQ0, kIA3IRQ1, kIA3IRQ2, kIA3IRQ3, kIA3IRQ4, kIA3IRQ5, kIA3IRQ6,
    kIA3IRQ7, kIA3IRQ8, kIA3IRQ9, kIA3IRQA, kIA3IRQB, kIA3IRQC, kIRQSW3, kIRQHW3, kIRQHL3, kCTGDINI,
    kCTGCTRL, kMEMRW, kMEMCOR, kDMDELA, kDMDELS, kNMOD, kNDLY, kNED, kNTRO, kNRRO, kNBND, kNP0, kNP1,
    kNP2, kNP3, kC08CPU0, kQ2VINFO /*C09CPU0  will be q2 pid window settings, and partial version info.*/,
    kC10CPU0, kC11CPU0, kC12CPUA, kC13CPUA, kC14CPUA, kC15CPUA, kC08CPU1, kVINFO/*kC09CPU1  version of trap code.*/,
    kC10CPU1, kC11CPU1, kC08CPU2, kNDRIFT /*kC09CPU2*/, kC10CPU2, kC11CPU2, kC08CPU3, kYCORR /*kC09CPU3*/,
    kC10CPU3, kC11CPU3, kNES, kNTP, kNCUT, kPASACHM, kLastReg
  };
  // clang-format on

 public:
  TrapRegisters();
  ~TrapRegisters() = default;
  /*static TrapRegisters* Instance(){
      static TrapRegisters trapreg;
      return &trapreg;
  }*/
  const uint32_t getRegBase(int idx) { return mTrapRegisters[idx].getBase(); }
  const uint32_t getRegWordNumber(int idx) { return mTrapRegisters[idx].getWordNumber(); }
  const uint32_t getRegMask(int idx) { return mTrapRegisters[idx].getMask(); }
  const uint32_t getRegShift(int idx) { return mTrapRegisters[idx].getShift(); }
  TrapRegInfo& operator[](uint32_t); // simplify the access via an overloaded operator to the registers.

  int32_t getRegIndexByName(const std::string& name);
  int32_t getRegAddrByName(const std::string& name);
  int32_t getRegAddr(const uint16_t regidx) { return mTrapRegisters[regidx].getAddr(); }

 private:
  // TrapRegisters();  // to make singleton
  std::array<TrapRegInfo, kLastReg> mTrapRegisters;         // store of layout of each block of mTrapRegisterSize, populated via initialiseRegisters
  std::array<uint32_t, kLastReg> mTrapRegisterAverageValue; // store the average values to be used for those ones where the mcm is not in the data stream
  void initialiseRegisters();

  ClassDefNV(TrapRegisters, 1);
};

} // namespace trd
} // namespace o2

#endif
