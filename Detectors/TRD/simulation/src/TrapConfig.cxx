// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRAP config                                                           //
//                                                                        //
//  Author: J. Klein (Jochen.Klein@cern.ch) (run2 version                 //
//          S. Murray (murrays@cern.ch)                   //
////////////////////////////////////////////////////////////////////////////

#include "TRDBase/Geometry.h"
#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapConfig.h"
#include "DataFormatsTRD/Constants.h"
#include <fairlogger/Logger.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>

using namespace std;
using namespace o2::trd;
using namespace o2::trd::constants;

bool TrapConfig::mgRegAddressMapInitialized = false;

const std::array<int, TrapConfig::mlastAlloc> o2::trd::TrapConfig::TrapValue::mgkSize = {0, 1, 540, 1080, 8 * 18 * 540, 4, 6, 8 * 18 * 30};

TrapConfig::TrapConfig()
{
  // default constructor

  // initialize and reset the TRAP registers
  initRegs();
  resetRegs();

  for (int iWord = 0; iWord < mgkDmemWords; ++iWord) {
    mDmem[iWord].setAddress(iWord + mgkDmemStartAddress);
  }

  // initialize the map from address to register
  if (!mgRegAddressMapInitialized) {
    for (int iReg = 0; iReg < kLastReg; iReg++) {
      int addr = mRegisterValue[iReg].getAddr();
      if (addr < mgkRegisterAddressBlockStart[0]) {
        LOG(error) << "Register address 0x" << hex << std::setw(4) << addr << " not handled in register map";
      } else if (addr < mgkRegisterAddressBlockStart[0] + mgkRegisterAddressBlockSize[0]) {
        mgRegAddressMap[addr - mgkRegisterAddressBlockStart[0]] = (TrapReg_t)iReg;
      } else if (addr < mgkRegisterAddressBlockStart[1]) {
        LOG(error) << "Register address 0x" << hex << setw(4) << addr << " not handled in register map";
      } else if (addr < mgkRegisterAddressBlockStart[1] + mgkRegisterAddressBlockSize[1]) {
        mgRegAddressMap[addr - mgkRegisterAddressBlockStart[1] + mgkRegisterAddressBlockSize[0]] = (TrapReg_t)iReg;
      } else if (addr < mgkRegisterAddressBlockStart[2]) {
        LOG(error) << "Register address 0x" << hex << setw(4) << addr << " not handled in register map";
      } else if (addr < mgkRegisterAddressBlockStart[2] + mgkRegisterAddressBlockSize[2]) {
        int ind = addr - mgkRegisterAddressBlockStart[2] + mgkRegisterAddressBlockSize[1] + mgkRegisterAddressBlockSize[0];
        mgRegAddressMap[ind] = (TrapReg_t)iReg;
      } else {
        LOG(error) << "Register address 0x" << hex << setw(4) << addr << " not handled in register map";
      }
    }
    mgRegAddressMapInitialized = true;
  }
}

TrapConfig::~TrapConfig() = default;

void TrapConfig::initRegs()
{
  // initialize all TRAP registers

  //                              Name          Address  Nbits   Reset Value
  mRegisterValue[kSML0].init("SML0", 0x0A00, 15, 0x4050); // Global state machine
  mRegisterValue[kSML1].init("SML1", 0x0A01, 15, 0x4200);
  mRegisterValue[kSML2].init("SML2", 0x0A02, 15, 0x4384);
  mRegisterValue[kSMMODE].init("SMMODE", 0x0A03, 16, 0xF0E2);
  mRegisterValue[kSMCMD].init("SMCMD", 0x0A04, 16, 0x0000);
  mRegisterValue[kNITM0].init("NITM0", 0x0A08, 14, 0x3FFF);
  mRegisterValue[kNITM1].init("NITM1", 0x0A09, 14, 0x3FFF);
  mRegisterValue[kNITM2].init("NITM2", 0x0A0A, 14, 0x3FFF);
  mRegisterValue[kNIP4D].init("NIP4D", 0x0A0B, 7, 0x7F);
  mRegisterValue[kCPU0CLK].init("CPU0CLK", 0x0A20, 5, 0x07);
  mRegisterValue[kCPU1CLK].init("CPU1CLK", 0x0A22, 5, 0x07);
  mRegisterValue[kCPU2CLK].init("CPU2CLK", 0x0A24, 5, 0x07);
  mRegisterValue[kCPU3CLK].init("CPU3CLK", 0x0A26, 5, 0x07);
  mRegisterValue[kNICLK].init("NICLK", 0x0A28, 5, 0x07);
  mRegisterValue[kFILCLK].init("FILCLK", 0x0A2A, 5, 0x07);
  mRegisterValue[kPRECLK].init("PRECLK", 0x0A2C, 5, 0x07);
  mRegisterValue[kADCEN].init("ADCEN", 0x0A2E, 5, 0x07);
  mRegisterValue[kNIODE].init("NIODE", 0x0A30, 5, 0x07);
  mRegisterValue[kNIOCE].init("NIOCE", 0x0A32, 6, 0x21); // bit 5 is status bit (read-only)!
  mRegisterValue[kNIIDE].init("NIIDE", 0x0A34, 5, 0x07);
  mRegisterValue[kNIICE].init("NIICE", 0x0A36, 5, 0x07);
  mRegisterValue[kARBTIM].init("ARBTIM", 0x0A3F, 4, 0x0);      // Arbiter
  mRegisterValue[kIA0IRQ0].init("IA0IRQ0", 0x0B00, 12, 0x000); // IVT om CPU0
  mRegisterValue[kIA0IRQ1].init("IA0IRQ1", 0x0B01, 12, 0x000);
  mRegisterValue[kIA0IRQ2].init("IA0IRQ2", 0x0B02, 12, 0x000);
  mRegisterValue[kIA0IRQ3].init("IA0IRQ3", 0x0B03, 12, 0x000);
  mRegisterValue[kIA0IRQ4].init("IA0IRQ4", 0x0B04, 12, 0x000);
  mRegisterValue[kIA0IRQ5].init("IA0IRQ5", 0x0B05, 12, 0x000);
  mRegisterValue[kIA0IRQ6].init("IA0IRQ6", 0x0B06, 12, 0x000);
  mRegisterValue[kIA0IRQ7].init("IA0IRQ7", 0x0B07, 12, 0x000);
  mRegisterValue[kIA0IRQ8].init("IA0IRQ8", 0x0B08, 12, 0x000);
  mRegisterValue[kIA0IRQ9].init("IA0IRQ9", 0x0B09, 12, 0x000);
  mRegisterValue[kIA0IRQA].init("IA0IRQA", 0x0B0A, 12, 0x000);
  mRegisterValue[kIA0IRQB].init("IA0IRQB", 0x0B0B, 12, 0x000);
  mRegisterValue[kIA0IRQC].init("IA0IRQC", 0x0B0C, 12, 0x000);
  mRegisterValue[kIRQSW0].init("IRQSW0", 0x0B0D, 13, 0x1FFF);
  mRegisterValue[kIRQHW0].init("IRQHW0", 0x0B0E, 13, 0x0000);
  mRegisterValue[kIRQHL0].init("IRQHL0", 0x0B0F, 13, 0x0000);
  mRegisterValue[kIA1IRQ0].init("IA1IRQ0", 0x0B20, 12, 0x000); // IVT om CPU1
  mRegisterValue[kIA1IRQ1].init("IA1IRQ1", 0x0B21, 12, 0x000);
  mRegisterValue[kIA1IRQ2].init("IA1IRQ2", 0x0B22, 12, 0x000);
  mRegisterValue[kIA1IRQ3].init("IA1IRQ3", 0x0B23, 12, 0x000);
  mRegisterValue[kIA1IRQ4].init("IA1IRQ4", 0x0B24, 12, 0x000);
  mRegisterValue[kIA1IRQ5].init("IA1IRQ5", 0x0B25, 12, 0x000);
  mRegisterValue[kIA1IRQ6].init("IA1IRQ6", 0x0B26, 12, 0x000);
  mRegisterValue[kIA1IRQ7].init("IA1IRQ7", 0x0B27, 12, 0x000);
  mRegisterValue[kIA1IRQ8].init("IA1IRQ8", 0x0B28, 12, 0x000);
  mRegisterValue[kIA1IRQ9].init("IA1IRQ9", 0x0B29, 12, 0x000);
  mRegisterValue[kIA1IRQA].init("IA1IRQA", 0x0B2A, 12, 0x000);
  mRegisterValue[kIA1IRQB].init("IA1IRQB", 0x0B2B, 12, 0x000);
  mRegisterValue[kIA1IRQC].init("IA1IRQC", 0x0B2C, 12, 0x000);
  mRegisterValue[kIRQSW1].init("IRQSW1", 0x0B2D, 13, 0x1FFF);
  mRegisterValue[kIRQHW1].init("IRQHW1", 0x0B2E, 13, 0x0000);
  mRegisterValue[kIRQHL1].init("IRQHL1", 0x0B2F, 13, 0x0000);
  mRegisterValue[kIA2IRQ0].init("IA2IRQ0", 0x0B40, 12, 0x000); // IVT om CPU2
  mRegisterValue[kIA2IRQ1].init("IA2IRQ1", 0x0B41, 12, 0x000);
  mRegisterValue[kIA2IRQ2].init("IA2IRQ2", 0x0B42, 12, 0x000);
  mRegisterValue[kIA2IRQ3].init("IA2IRQ3", 0x0B43, 12, 0x000);
  mRegisterValue[kIA2IRQ4].init("IA2IRQ4", 0x0B44, 12, 0x000);
  mRegisterValue[kIA2IRQ5].init("IA2IRQ5", 0x0B45, 12, 0x000);
  mRegisterValue[kIA2IRQ6].init("IA2IRQ6", 0x0B46, 12, 0x000);
  mRegisterValue[kIA2IRQ7].init("IA2IRQ7", 0x0B47, 12, 0x000);
  mRegisterValue[kIA2IRQ8].init("IA2IRQ8", 0x0B48, 12, 0x000);
  mRegisterValue[kIA2IRQ9].init("IA2IRQ9", 0x0B49, 12, 0x000);
  mRegisterValue[kIA2IRQA].init("IA2IRQA", 0x0B4A, 12, 0x000);
  mRegisterValue[kIA2IRQB].init("IA2IRQB", 0x0B4B, 12, 0x000);
  mRegisterValue[kIA2IRQC].init("IA2IRQC", 0x0B4C, 12, 0x000);
  mRegisterValue[kIRQSW2].init("IRQSW2", 0x0B4D, 13, 0x1FFF);
  mRegisterValue[kIRQHW2].init("IRQHW2", 0x0B4E, 13, 0x0000);
  mRegisterValue[kIRQHL2].init("IRQHL2", 0x0B4F, 13, 0x0000);
  mRegisterValue[kIA3IRQ0].init("IA3IRQ0", 0x0B60, 12, 0x000); // IVT om CPU3
  mRegisterValue[kIA3IRQ1].init("IA3IRQ1", 0x0B61, 12, 0x000);
  mRegisterValue[kIA3IRQ2].init("IA3IRQ2", 0x0B62, 12, 0x000);
  mRegisterValue[kIA3IRQ3].init("IA3IRQ3", 0x0B63, 12, 0x000);
  mRegisterValue[kIA3IRQ4].init("IA3IRQ4", 0x0B64, 12, 0x000);
  mRegisterValue[kIA3IRQ5].init("IA3IRQ5", 0x0B65, 12, 0x000);
  mRegisterValue[kIA3IRQ6].init("IA3IRQ6", 0x0B66, 12, 0x000);
  mRegisterValue[kIA3IRQ7].init("IA3IRQ7", 0x0B67, 12, 0x000);
  mRegisterValue[kIA3IRQ8].init("IA3IRQ8", 0x0B68, 12, 0x000);
  mRegisterValue[kIA3IRQ9].init("IA3IRQ9", 0x0B69, 12, 0x000);
  mRegisterValue[kIA3IRQA].init("IA3IRQA", 0x0B6A, 12, 0x000);
  mRegisterValue[kIA3IRQB].init("IA3IRQB", 0x0B6B, 12, 0x000);
  mRegisterValue[kIA3IRQC].init("IA3IRQC", 0x0B6C, 12, 0x000);
  mRegisterValue[kIRQSW3].init("IRQSW3", 0x0B6D, 13, 0x1FFF);
  mRegisterValue[kIRQHW3].init("IRQHW3", 0x0B6E, 13, 0x0000);
  mRegisterValue[kIRQHL3].init("IRQHL3", 0x0B6F, 13, 0x0000);
  mRegisterValue[kCTGDINI].init("CTGDINI", 0x0B80, 32, 0x00000000); // Global Counter/Timer
  mRegisterValue[kCTGCTRL].init("CTGCTRL", 0x0B81, 12, 0xE3F);
  mRegisterValue[kC08CPU0].init("C08CPU0", 0x0C00, 32, 0x00000000); // CPU constants
  mRegisterValue[kC09CPU0].init("C09CPU0", 0x0C01, 32, 0x00000000);
  mRegisterValue[kC10CPU0].init("C10CPU0", 0x0C02, 32, 0x00000000);
  mRegisterValue[kC11CPU0].init("C11CPU0", 0x0C03, 32, 0x00000000);
  mRegisterValue[kC12CPUA].init("C12CPUA", 0x0C04, 32, 0x00000000);
  mRegisterValue[kC13CPUA].init("C13CPUA", 0x0C05, 32, 0x00000000);
  mRegisterValue[kC14CPUA].init("C14CPUA", 0x0C06, 32, 0x00000000);
  mRegisterValue[kC15CPUA].init("C15CPUA", 0x0C07, 32, 0x00000000);
  mRegisterValue[kC08CPU1].init("C08CPU1", 0x0C08, 32, 0x00000000);
  mRegisterValue[kC09CPU1].init("C09CPU1", 0x0C09, 32, 0x00000000);
  mRegisterValue[kC10CPU1].init("C10CPU1", 0x0C0A, 32, 0x00000000);
  mRegisterValue[kC11CPU1].init("C11CPU1", 0x0C0B, 32, 0x00000000);
  mRegisterValue[kC08CPU2].init("C08CPU2", 0x0C10, 32, 0x00000000);
  mRegisterValue[kC09CPU2].init("C09CPU2", 0x0C11, 32, 0x00000000);
  mRegisterValue[kC10CPU2].init("C10CPU2", 0x0C12, 32, 0x00000000);
  mRegisterValue[kC11CPU2].init("C11CPU2", 0x0C13, 32, 0x00000000);
  mRegisterValue[kC08CPU3].init("C08CPU3", 0x0C18, 32, 0x00000000);
  mRegisterValue[kC09CPU3].init("C09CPU3", 0x0C19, 32, 0x00000000);
  mRegisterValue[kC10CPU3].init("C10CPU3", 0x0C1A, 32, 0x00000000);
  mRegisterValue[kC11CPU3].init("C11CPU3", 0x0C1B, 32, 0x00000000);
  mRegisterValue[kNMOD].init("NMOD", 0x0D40, 6, 0x08); // NI intermace
  mRegisterValue[kNDLY].init("NDLY", 0x0D41, 30, 0x24924924);
  mRegisterValue[kNED].init("NED", 0x0D42, 16, 0xA240);
  mRegisterValue[kNTRO].init("NTRO", 0x0D43, 18, 0x3FFFC);
  mRegisterValue[kNRRO].init("NRRO", 0x0D44, 18, 0x3FFFC);
  mRegisterValue[kNES].init("NES", 0x0D45, 32, 0x00000000);
  mRegisterValue[kNTP].init("NTP", 0x0D46, 32, 0x0000FFFF);
  mRegisterValue[kNBND].init("NBND", 0x0D47, 16, 0x6020);
  mRegisterValue[kNP0].init("NP0", 0x0D48, 11, 0x44C);
  mRegisterValue[kNP1].init("NP1", 0x0D49, 11, 0x44C);
  mRegisterValue[kNP2].init("NP2", 0x0D4A, 11, 0x44C);
  mRegisterValue[kNP3].init("NP3", 0x0D4B, 11, 0x44C);
  mRegisterValue[kNCUT].init("NCUT", 0x0D4C, 32, 0xFFFFFFFF);
  mRegisterValue[kTPPT0].init("TPPT0", 0x3000, 7, 0x01); // Filter and Preprocessor
  mRegisterValue[kTPFS].init("TPFS", 0x3001, 7, 0x05);
  mRegisterValue[kTPFE].init("TPFE", 0x3002, 7, 0x14);
  mRegisterValue[kTPPGR].init("TPPGR", 0x3003, 7, 0x15);
  mRegisterValue[kTPPAE].init("TPPAE", 0x3004, 7, 0x1E);
  mRegisterValue[kTPQS0].init("TPQS0", 0x3005, 7, 0x00);
  mRegisterValue[kTPQE0].init("TPQE0", 0x3006, 7, 0x0A);
  mRegisterValue[kTPQS1].init("TPQS1", 0x3007, 7, 0x0B);
  mRegisterValue[kTPQE1].init("TPQE1", 0x3008, 7, 0x14);
  mRegisterValue[kEBD].init("EBD", 0x3009, 3, 0x0);
  mRegisterValue[kEBAQA].init("EBAQA", 0x300A, 7, 0x00);
  mRegisterValue[kEBSIA].init("EBSIA", 0x300B, 7, 0x20);
  mRegisterValue[kEBSF].init("EBSF", 0x300C, 1, 0x1);
  mRegisterValue[kEBSIM].init("EBSIM", 0x300D, 1, 0x1);
  mRegisterValue[kEBPP].init("EBPP", 0x300E, 1, 0x1);
  mRegisterValue[kEBPC].init("EBPC", 0x300F, 1, 0x1);
  mRegisterValue[kEBIS].init("EBIS", 0x3014, 10, 0x005);
  mRegisterValue[kEBIT].init("EBIT", 0x3015, 12, 0x028);
  mRegisterValue[kEBIL].init("EBIL", 0x3016, 8, 0xF0);
  mRegisterValue[kEBIN].init("EBIN", 0x3017, 1, 0x1);
  mRegisterValue[kFLBY].init("FLBY", 0x3018, 1, 0x0);
  mRegisterValue[kFPBY].init("FPBY", 0x3019, 1, 0x0);
  mRegisterValue[kFGBY].init("FGBY", 0x301A, 1, 0x0);
  mRegisterValue[kFTBY].init("FTBY", 0x301B, 1, 0x0);
  mRegisterValue[kFCBY].init("FCBY", 0x301C, 1, 0x0);
  mRegisterValue[kFPTC].init("FPTC", 0x3020, 2, 0x3);
  mRegisterValue[kFPNP].init("FPNP", 0x3021, 9, 0x078);
  mRegisterValue[kFPCL].init("FPCL", 0x3022, 1, 0x1);
  mRegisterValue[kFGTA].init("FGTA", 0x3028, 12, 0x014);
  mRegisterValue[kFGTB].init("FGTB", 0x3029, 12, 0x80C);
  mRegisterValue[kFGCL].init("FGCL", 0x302A, 1, 0x1);
  mRegisterValue[kFTAL].init("FTAL", 0x3030, 10, 0x0F6);
  mRegisterValue[kFTLL].init("FTLL", 0x3031, 9, 0x11D);
  mRegisterValue[kFTLS].init("FTLS", 0x3032, 9, 0x0D3);
  mRegisterValue[kFCW1].init("FCW1", 0x3038, 8, 0x1E);
  mRegisterValue[kFCW2].init("FCW2", 0x3039, 8, 0xD4);
  mRegisterValue[kFCW3].init("FCW3", 0x303A, 8, 0xE6);
  mRegisterValue[kFCW4].init("FCW4", 0x303B, 8, 0x4A);
  mRegisterValue[kFCW5].init("FCW5", 0x303C, 8, 0xEF);
  mRegisterValue[kTPFP].init("TPFP", 0x3040, 9, 0x037);
  mRegisterValue[kTPHT].init("TPHT", 0x3041, 14, 0x00A0);
  mRegisterValue[kTPVT].init("TPVT", 0x3042, 6, 0x00);
  mRegisterValue[kTPVBY].init("TPVBY", 0x3043, 1, 0x0);
  mRegisterValue[kTPCT].init("TPCT", 0x3044, 5, 0x08);
  mRegisterValue[kTPCL].init("TPCL", 0x3045, 5, 0x01);
  mRegisterValue[kTPCBY].init("TPCBY", 0x3046, 1, 0x1);
  mRegisterValue[kTPD].init("TPD", 0x3047, 4, 0xF);
  mRegisterValue[kTPCI0].init("TPCI0", 0x3048, 5, 0x00);
  mRegisterValue[kTPCI1].init("TPCI1", 0x3049, 5, 0x00);
  mRegisterValue[kTPCI2].init("TPCI2", 0x304A, 5, 0x00);
  mRegisterValue[kTPCI3].init("TPCI3", 0x304B, 5, 0x00);
  mRegisterValue[kADCMSK].init("ADCMSK", 0x3050, 21, 0x1FFFFF);
  mRegisterValue[kADCINB].init("ADCINB", 0x3051, 2, 0x2);
  mRegisterValue[kADCDAC].init("ADCDAC", 0x3052, 5, 0x10);
  mRegisterValue[kADCPAR].init("ADCPAR", 0x3053, 18, 0x195EF);
  mRegisterValue[kADCTST].init("ADCTST", 0x3054, 2, 0x0);
  mRegisterValue[kSADCAZ].init("SADCAZ", 0x3055, 1, 0x1);
  mRegisterValue[kFGF0].init("FGF0", 0x3080, 9, 0x000);
  mRegisterValue[kFGF1].init("FGF1", 0x3081, 9, 0x000);
  mRegisterValue[kFGF2].init("FGF2", 0x3082, 9, 0x000);
  mRegisterValue[kFGF3].init("FGF3", 0x3083, 9, 0x000);
  mRegisterValue[kFGF4].init("FGF4", 0x3084, 9, 0x000);
  mRegisterValue[kFGF5].init("FGF5", 0x3085, 9, 0x000);
  mRegisterValue[kFGF6].init("FGF6", 0x3086, 9, 0x000);
  mRegisterValue[kFGF7].init("FGF7", 0x3087, 9, 0x000);
  mRegisterValue[kFGF8].init("FGF8", 0x3088, 9, 0x000);
  mRegisterValue[kFGF9].init("FGF9", 0x3089, 9, 0x000);
  mRegisterValue[kFGF10].init("FGF10", 0x308A, 9, 0x000);
  mRegisterValue[kFGF11].init("FGF11", 0x308B, 9, 0x000);
  mRegisterValue[kFGF12].init("FGF12", 0x308C, 9, 0x000);
  mRegisterValue[kFGF13].init("FGF13", 0x308D, 9, 0x000);
  mRegisterValue[kFGF14].init("FGF14", 0x308E, 9, 0x000);
  mRegisterValue[kFGF15].init("FGF15", 0x308F, 9, 0x000);
  mRegisterValue[kFGF16].init("FGF16", 0x3090, 9, 0x000);
  mRegisterValue[kFGF17].init("FGF17", 0x3091, 9, 0x000);
  mRegisterValue[kFGF18].init("FGF18", 0x3092, 9, 0x000);
  mRegisterValue[kFGF19].init("FGF19", 0x3093, 9, 0x000);
  mRegisterValue[kFGF20].init("FGF20", 0x3094, 9, 0x000);
  mRegisterValue[kFGA0].init("FGA0", 0x30A0, 6, 0x00);
  mRegisterValue[kFGA1].init("FGA1", 0x30A1, 6, 0x00);
  mRegisterValue[kFGA2].init("FGA2", 0x30A2, 6, 0x00);
  mRegisterValue[kFGA3].init("FGA3", 0x30A3, 6, 0x00);
  mRegisterValue[kFGA4].init("FGA4", 0x30A4, 6, 0x00);
  mRegisterValue[kFGA5].init("FGA5", 0x30A5, 6, 0x00);
  mRegisterValue[kFGA6].init("FGA6", 0x30A6, 6, 0x00);
  mRegisterValue[kFGA7].init("FGA7", 0x30A7, 6, 0x00);
  mRegisterValue[kFGA8].init("FGA8", 0x30A8, 6, 0x00);
  mRegisterValue[kFGA9].init("FGA9", 0x30A9, 6, 0x00);
  mRegisterValue[kFGA10].init("FGA10", 0x30AA, 6, 0x00);
  mRegisterValue[kFGA11].init("FGA11", 0x30AB, 6, 0x00);
  mRegisterValue[kFGA12].init("FGA12", 0x30AC, 6, 0x00);
  mRegisterValue[kFGA13].init("FGA13", 0x30AD, 6, 0x00);
  mRegisterValue[kFGA14].init("FGA14", 0x30AE, 6, 0x00);
  mRegisterValue[kFGA15].init("FGA15", 0x30AF, 6, 0x00);
  mRegisterValue[kFGA16].init("FGA16", 0x30B0, 6, 0x00);
  mRegisterValue[kFGA17].init("FGA17", 0x30B1, 6, 0x00);
  mRegisterValue[kFGA18].init("FGA18", 0x30B2, 6, 0x00);
  mRegisterValue[kFGA19].init("FGA19", 0x30B3, 6, 0x00);
  mRegisterValue[kFGA20].init("FGA20", 0x30B4, 6, 0x00);
  mRegisterValue[kFLL00].init("FLL00", 0x3100, 6, 0x00); // non-linearity table, 64 x 6 bits
  mRegisterValue[kFLL01].init("FLL01", 0x3101, 6, 0x00);
  mRegisterValue[kFLL02].init("FLL02", 0x3102, 6, 0x00);
  mRegisterValue[kFLL03].init("FLL03", 0x3103, 6, 0x00);
  mRegisterValue[kFLL04].init("FLL04", 0x3104, 6, 0x00);
  mRegisterValue[kFLL05].init("FLL05", 0x3105, 6, 0x00);
  mRegisterValue[kFLL06].init("FLL06", 0x3106, 6, 0x00);
  mRegisterValue[kFLL07].init("FLL07", 0x3107, 6, 0x00);
  mRegisterValue[kFLL08].init("FLL08", 0x3108, 6, 0x00);
  mRegisterValue[kFLL09].init("FLL09", 0x3109, 6, 0x00);
  mRegisterValue[kFLL0A].init("FLL0A", 0x310A, 6, 0x00);
  mRegisterValue[kFLL0B].init("FLL0B", 0x310B, 6, 0x00);
  mRegisterValue[kFLL0C].init("FLL0C", 0x310C, 6, 0x00);
  mRegisterValue[kFLL0D].init("FLL0D", 0x310D, 6, 0x00);
  mRegisterValue[kFLL0E].init("FLL0E", 0x310E, 6, 0x00);
  mRegisterValue[kFLL0F].init("FLL0F", 0x310F, 6, 0x00);
  mRegisterValue[kFLL10].init("FLL10", 0x3110, 6, 0x00);
  mRegisterValue[kFLL11].init("FLL11", 0x3111, 6, 0x00);
  mRegisterValue[kFLL12].init("FLL12", 0x3112, 6, 0x00);
  mRegisterValue[kFLL13].init("FLL13", 0x3113, 6, 0x00);
  mRegisterValue[kFLL14].init("FLL14", 0x3114, 6, 0x00);
  mRegisterValue[kFLL15].init("FLL15", 0x3115, 6, 0x00);
  mRegisterValue[kFLL16].init("FLL16", 0x3116, 6, 0x00);
  mRegisterValue[kFLL17].init("FLL17", 0x3117, 6, 0x00);
  mRegisterValue[kFLL18].init("FLL18", 0x3118, 6, 0x00);
  mRegisterValue[kFLL19].init("FLL19", 0x3119, 6, 0x00);
  mRegisterValue[kFLL1A].init("FLL1A", 0x311A, 6, 0x00);
  mRegisterValue[kFLL1B].init("FLL1B", 0x311B, 6, 0x00);
  mRegisterValue[kFLL1C].init("FLL1C", 0x311C, 6, 0x00);
  mRegisterValue[kFLL1D].init("FLL1D", 0x311D, 6, 0x00);
  mRegisterValue[kFLL1E].init("FLL1E", 0x311E, 6, 0x00);
  mRegisterValue[kFLL1F].init("FLL1F", 0x311F, 6, 0x00);
  mRegisterValue[kFLL20].init("FLL20", 0x3120, 6, 0x00);
  mRegisterValue[kFLL21].init("FLL21", 0x3121, 6, 0x00);
  mRegisterValue[kFLL22].init("FLL22", 0x3122, 6, 0x00);
  mRegisterValue[kFLL23].init("FLL23", 0x3123, 6, 0x00);
  mRegisterValue[kFLL24].init("FLL24", 0x3124, 6, 0x00);
  mRegisterValue[kFLL25].init("FLL25", 0x3125, 6, 0x00);
  mRegisterValue[kFLL26].init("FLL26", 0x3126, 6, 0x00);
  mRegisterValue[kFLL27].init("FLL27", 0x3127, 6, 0x00);
  mRegisterValue[kFLL28].init("FLL28", 0x3128, 6, 0x00);
  mRegisterValue[kFLL29].init("FLL29", 0x3129, 6, 0x00);
  mRegisterValue[kFLL2A].init("FLL2A", 0x312A, 6, 0x00);
  mRegisterValue[kFLL2B].init("FLL2B", 0x312B, 6, 0x00);
  mRegisterValue[kFLL2C].init("FLL2C", 0x312C, 6, 0x00);
  mRegisterValue[kFLL2D].init("FLL2D", 0x312D, 6, 0x00);
  mRegisterValue[kFLL2E].init("FLL2E", 0x312E, 6, 0x00);
  mRegisterValue[kFLL2F].init("FLL2F", 0x312F, 6, 0x00);
  mRegisterValue[kFLL30].init("FLL30", 0x3130, 6, 0x00);
  mRegisterValue[kFLL31].init("FLL31", 0x3131, 6, 0x00);
  mRegisterValue[kFLL32].init("FLL32", 0x3132, 6, 0x00);
  mRegisterValue[kFLL33].init("FLL33", 0x3133, 6, 0x00);
  mRegisterValue[kFLL34].init("FLL34", 0x3134, 6, 0x00);
  mRegisterValue[kFLL35].init("FLL35", 0x3135, 6, 0x00);
  mRegisterValue[kFLL36].init("FLL36", 0x3136, 6, 0x00);
  mRegisterValue[kFLL37].init("FLL37", 0x3137, 6, 0x00);
  mRegisterValue[kFLL38].init("FLL38", 0x3138, 6, 0x00);
  mRegisterValue[kFLL39].init("FLL39", 0x3139, 6, 0x00);
  mRegisterValue[kFLL3A].init("FLL3A", 0x313A, 6, 0x00);
  mRegisterValue[kFLL3B].init("FLL3B", 0x313B, 6, 0x00);
  mRegisterValue[kFLL3C].init("FLL3C", 0x313C, 6, 0x00);
  mRegisterValue[kFLL3D].init("FLL3D", 0x313D, 6, 0x00);
  mRegisterValue[kFLL3E].init("FLL3E", 0x313E, 6, 0x00);
  mRegisterValue[kFLL3F].init("FLL3F", 0x313F, 6, 0x00);
  mRegisterValue[kPASADEL].init("PASADEL", 0x3158, 8, 0xFF); // end om non-lin table
  mRegisterValue[kPASAPHA].init("PASAPHA", 0x3159, 6, 0x3F);
  mRegisterValue[kPASAPRA].init("PASAPRA", 0x315A, 6, 0x0F);
  mRegisterValue[kPASADAC].init("PASADAC", 0x315B, 8, 0x80);
  mRegisterValue[kPASACHM].init("PASACHM", 0x315C, 19, 0x7FFFF);
  mRegisterValue[kPASASTL].init("PASASTL", 0x315D, 8, 0xFF);
  mRegisterValue[kPASAPR1].init("PASAPR1", 0x315E, 1, 0x0);
  mRegisterValue[kPASAPR0].init("PASAPR0", 0x315F, 1, 0x0);
  mRegisterValue[kSADCTRG].init("SADCTRG", 0x3161, 1, 0x0);
  mRegisterValue[kSADCRUN].init("SADCRUN", 0x3162, 1, 0x0);
  mRegisterValue[kSADCPWR].init("SADCPWR", 0x3163, 3, 0x7);
  mRegisterValue[kL0TSIM].init("L0TSIM", 0x3165, 14, 0x0050);
  mRegisterValue[kSADCEC].init("SADCEC", 0x3166, 7, 0x00);
  mRegisterValue[kSADCMC].init("SADCMC", 0x3170, 8, 0xC0);
  mRegisterValue[kSADCOC].init("SADCOC", 0x3171, 8, 0x19);
  mRegisterValue[kSADCGTB].init("SADCGTB", 0x3172, 32, 0x37737700);
  mRegisterValue[kSEBDEN].init("SEBDEN", 0x3178, 3, 0x0);
  mRegisterValue[kSEBDOU].init("SEBDOU", 0x3179, 3, 0x0);
  mRegisterValue[kTPL00].init("TPL00", 0x3180, 5, 0x00); // pos table, 128 x 5 bits
  mRegisterValue[kTPL01].init("TPL01", 0x3181, 5, 0x00);
  mRegisterValue[kTPL02].init("TPL02", 0x3182, 5, 0x00);
  mRegisterValue[kTPL03].init("TPL03", 0x3183, 5, 0x00);
  mRegisterValue[kTPL04].init("TPL04", 0x3184, 5, 0x00);
  mRegisterValue[kTPL05].init("TPL05", 0x3185, 5, 0x00);
  mRegisterValue[kTPL06].init("TPL06", 0x3186, 5, 0x00);
  mRegisterValue[kTPL07].init("TPL07", 0x3187, 5, 0x00);
  mRegisterValue[kTPL08].init("TPL08", 0x3188, 5, 0x00);
  mRegisterValue[kTPL09].init("TPL09", 0x3189, 5, 0x00);
  mRegisterValue[kTPL0A].init("TPL0A", 0x318A, 5, 0x00);
  mRegisterValue[kTPL0B].init("TPL0B", 0x318B, 5, 0x00);
  mRegisterValue[kTPL0C].init("TPL0C", 0x318C, 5, 0x00);
  mRegisterValue[kTPL0D].init("TPL0D", 0x318D, 5, 0x00);
  mRegisterValue[kTPL0E].init("TPL0E", 0x318E, 5, 0x00);
  mRegisterValue[kTPL0F].init("TPL0F", 0x318F, 5, 0x00);
  mRegisterValue[kTPL10].init("TPL10", 0x3190, 5, 0x00);
  mRegisterValue[kTPL11].init("TPL11", 0x3191, 5, 0x00);
  mRegisterValue[kTPL12].init("TPL12", 0x3192, 5, 0x00);
  mRegisterValue[kTPL13].init("TPL13", 0x3193, 5, 0x00);
  mRegisterValue[kTPL14].init("TPL14", 0x3194, 5, 0x00);
  mRegisterValue[kTPL15].init("TPL15", 0x3195, 5, 0x00);
  mRegisterValue[kTPL16].init("TPL16", 0x3196, 5, 0x00);
  mRegisterValue[kTPL17].init("TPL17", 0x3197, 5, 0x00);
  mRegisterValue[kTPL18].init("TPL18", 0x3198, 5, 0x00);
  mRegisterValue[kTPL19].init("TPL19", 0x3199, 5, 0x00);
  mRegisterValue[kTPL1A].init("TPL1A", 0x319A, 5, 0x00);
  mRegisterValue[kTPL1B].init("TPL1B", 0x319B, 5, 0x00);
  mRegisterValue[kTPL1C].init("TPL1C", 0x319C, 5, 0x00);
  mRegisterValue[kTPL1D].init("TPL1D", 0x319D, 5, 0x00);
  mRegisterValue[kTPL1E].init("TPL1E", 0x319E, 5, 0x00);
  mRegisterValue[kTPL1F].init("TPL1F", 0x319F, 5, 0x00);
  mRegisterValue[kTPL20].init("TPL20", 0x31A0, 5, 0x00);
  mRegisterValue[kTPL21].init("TPL21", 0x31A1, 5, 0x00);
  mRegisterValue[kTPL22].init("TPL22", 0x31A2, 5, 0x00);
  mRegisterValue[kTPL23].init("TPL23", 0x31A3, 5, 0x00);
  mRegisterValue[kTPL24].init("TPL24", 0x31A4, 5, 0x00);
  mRegisterValue[kTPL25].init("TPL25", 0x31A5, 5, 0x00);
  mRegisterValue[kTPL26].init("TPL26", 0x31A6, 5, 0x00);
  mRegisterValue[kTPL27].init("TPL27", 0x31A7, 5, 0x00);
  mRegisterValue[kTPL28].init("TPL28", 0x31A8, 5, 0x00);
  mRegisterValue[kTPL29].init("TPL29", 0x31A9, 5, 0x00);
  mRegisterValue[kTPL2A].init("TPL2A", 0x31AA, 5, 0x00);
  mRegisterValue[kTPL2B].init("TPL2B", 0x31AB, 5, 0x00);
  mRegisterValue[kTPL2C].init("TPL2C", 0x31AC, 5, 0x00);
  mRegisterValue[kTPL2D].init("TPL2D", 0x31AD, 5, 0x00);
  mRegisterValue[kTPL2E].init("TPL2E", 0x31AE, 5, 0x00);
  mRegisterValue[kTPL2F].init("TPL2F", 0x31AF, 5, 0x00);
  mRegisterValue[kTPL30].init("TPL30", 0x31B0, 5, 0x00);
  mRegisterValue[kTPL31].init("TPL31", 0x31B1, 5, 0x00);
  mRegisterValue[kTPL32].init("TPL32", 0x31B2, 5, 0x00);
  mRegisterValue[kTPL33].init("TPL33", 0x31B3, 5, 0x00);
  mRegisterValue[kTPL34].init("TPL34", 0x31B4, 5, 0x00);
  mRegisterValue[kTPL35].init("TPL35", 0x31B5, 5, 0x00);
  mRegisterValue[kTPL36].init("TPL36", 0x31B6, 5, 0x00);
  mRegisterValue[kTPL37].init("TPL37", 0x31B7, 5, 0x00);
  mRegisterValue[kTPL38].init("TPL38", 0x31B8, 5, 0x00);
  mRegisterValue[kTPL39].init("TPL39", 0x31B9, 5, 0x00);
  mRegisterValue[kTPL3A].init("TPL3A", 0x31BA, 5, 0x00);
  mRegisterValue[kTPL3B].init("TPL3B", 0x31BB, 5, 0x00);
  mRegisterValue[kTPL3C].init("TPL3C", 0x31BC, 5, 0x00);
  mRegisterValue[kTPL3D].init("TPL3D", 0x31BD, 5, 0x00);
  mRegisterValue[kTPL3E].init("TPL3E", 0x31BE, 5, 0x00);
  mRegisterValue[kTPL3F].init("TPL3F", 0x31BF, 5, 0x00);
  mRegisterValue[kTPL40].init("TPL40", 0x31C0, 5, 0x00);
  mRegisterValue[kTPL41].init("TPL41", 0x31C1, 5, 0x00);
  mRegisterValue[kTPL42].init("TPL42", 0x31C2, 5, 0x00);
  mRegisterValue[kTPL43].init("TPL43", 0x31C3, 5, 0x00);
  mRegisterValue[kTPL44].init("TPL44", 0x31C4, 5, 0x00);
  mRegisterValue[kTPL45].init("TPL45", 0x31C5, 5, 0x00);
  mRegisterValue[kTPL46].init("TPL46", 0x31C6, 5, 0x00);
  mRegisterValue[kTPL47].init("TPL47", 0x31C7, 5, 0x00);
  mRegisterValue[kTPL48].init("TPL48", 0x31C8, 5, 0x00);
  mRegisterValue[kTPL49].init("TPL49", 0x31C9, 5, 0x00);
  mRegisterValue[kTPL4A].init("TPL4A", 0x31CA, 5, 0x00);
  mRegisterValue[kTPL4B].init("TPL4B", 0x31CB, 5, 0x00);
  mRegisterValue[kTPL4C].init("TPL4C", 0x31CC, 5, 0x00);
  mRegisterValue[kTPL4D].init("TPL4D", 0x31CD, 5, 0x00);
  mRegisterValue[kTPL4E].init("TPL4E", 0x31CE, 5, 0x00);
  mRegisterValue[kTPL4F].init("TPL4F", 0x31CF, 5, 0x00);
  mRegisterValue[kTPL50].init("TPL50", 0x31D0, 5, 0x00);
  mRegisterValue[kTPL51].init("TPL51", 0x31D1, 5, 0x00);
  mRegisterValue[kTPL52].init("TPL52", 0x31D2, 5, 0x00);
  mRegisterValue[kTPL53].init("TPL53", 0x31D3, 5, 0x00);
  mRegisterValue[kTPL54].init("TPL54", 0x31D4, 5, 0x00);
  mRegisterValue[kTPL55].init("TPL55", 0x31D5, 5, 0x00);
  mRegisterValue[kTPL56].init("TPL56", 0x31D6, 5, 0x00);
  mRegisterValue[kTPL57].init("TPL57", 0x31D7, 5, 0x00);
  mRegisterValue[kTPL58].init("TPL58", 0x31D8, 5, 0x00);
  mRegisterValue[kTPL59].init("TPL59", 0x31D9, 5, 0x00);
  mRegisterValue[kTPL5A].init("TPL5A", 0x31DA, 5, 0x00);
  mRegisterValue[kTPL5B].init("TPL5B", 0x31DB, 5, 0x00);
  mRegisterValue[kTPL5C].init("TPL5C", 0x31DC, 5, 0x00);
  mRegisterValue[kTPL5D].init("TPL5D", 0x31DD, 5, 0x00);
  mRegisterValue[kTPL5E].init("TPL5E", 0x31DE, 5, 0x00);
  mRegisterValue[kTPL5F].init("TPL5F", 0x31DF, 5, 0x00);
  mRegisterValue[kTPL60].init("TPL60", 0x31E0, 5, 0x00);
  mRegisterValue[kTPL61].init("TPL61", 0x31E1, 5, 0x00);
  mRegisterValue[kTPL62].init("TPL62", 0x31E2, 5, 0x00);
  mRegisterValue[kTPL63].init("TPL63", 0x31E3, 5, 0x00);
  mRegisterValue[kTPL64].init("TPL64", 0x31E4, 5, 0x00);
  mRegisterValue[kTPL65].init("TPL65", 0x31E5, 5, 0x00);
  mRegisterValue[kTPL66].init("TPL66", 0x31E6, 5, 0x00);
  mRegisterValue[kTPL67].init("TPL67", 0x31E7, 5, 0x00);
  mRegisterValue[kTPL68].init("TPL68", 0x31E8, 5, 0x00);
  mRegisterValue[kTPL69].init("TPL69", 0x31E9, 5, 0x00);
  mRegisterValue[kTPL6A].init("TPL6A", 0x31EA, 5, 0x00);
  mRegisterValue[kTPL6B].init("TPL6B", 0x31EB, 5, 0x00);
  mRegisterValue[kTPL6C].init("TPL6C", 0x31EC, 5, 0x00);
  mRegisterValue[kTPL6D].init("TPL6D", 0x31ED, 5, 0x00);
  mRegisterValue[kTPL6E].init("TPL6E", 0x31EE, 5, 0x00);
  mRegisterValue[kTPL6F].init("TPL6F", 0x31EF, 5, 0x00);
  mRegisterValue[kTPL70].init("TPL70", 0x31F0, 5, 0x00);
  mRegisterValue[kTPL71].init("TPL71", 0x31F1, 5, 0x00);
  mRegisterValue[kTPL72].init("TPL72", 0x31F2, 5, 0x00);
  mRegisterValue[kTPL73].init("TPL73", 0x31F3, 5, 0x00);
  mRegisterValue[kTPL74].init("TPL74", 0x31F4, 5, 0x00);
  mRegisterValue[kTPL75].init("TPL75", 0x31F5, 5, 0x00);
  mRegisterValue[kTPL76].init("TPL76", 0x31F6, 5, 0x00);
  mRegisterValue[kTPL77].init("TPL77", 0x31F7, 5, 0x00);
  mRegisterValue[kTPL78].init("TPL78", 0x31F8, 5, 0x00);
  mRegisterValue[kTPL79].init("TPL79", 0x31F9, 5, 0x00);
  mRegisterValue[kTPL7A].init("TPL7A", 0x31FA, 5, 0x00);
  mRegisterValue[kTPL7B].init("TPL7B", 0x31FB, 5, 0x00);
  mRegisterValue[kTPL7C].init("TPL7C", 0x31FC, 5, 0x00);
  mRegisterValue[kTPL7D].init("TPL7D", 0x31FD, 5, 0x00);
  mRegisterValue[kTPL7E].init("TPL7E", 0x31FE, 5, 0x00);
  mRegisterValue[kTPL7F].init("TPL7F", 0x31FF, 5, 0x00);
  mRegisterValue[kMEMRW].init("MEMRW", 0xD000, 7, 0x79); // end om pos table
  mRegisterValue[kMEMCOR].init("MEMCOR", 0xD001, 9, 0x000);
  mRegisterValue[kDMDELA].init("DMDELA", 0xD002, 4, 0x8);
  mRegisterValue[kDMDELS].init("DMDELS", 0xD003, 4, 0x8);
}

void TrapConfig::resetRegs()
{
  // Reset the content om all TRAP registers to the reset values (see TRAP User Manual)

  for (int iReg = 0; iReg < kLastReg; iReg++) {
    mRegisterValue[iReg].reset();
  }
}

void TrapConfig::resetDmem()
{
  // reset the data memory

  for (int iAddr = 0; iAddr < mgkDmemWords; iAddr++) {
    mDmem[iAddr].reset();
  }
}

int TrapConfig::getTrapReg(TrapReg_t reg, int det, int rob, int mcm)
{
  // get the value of an individual TRAP register
  // if it is individual for TRAPs a valid TRAP has to be specified

  if ((reg < 0) || (reg >= kLastReg)) {
    LOG(error) << "Non-existing register requested";
    return 0;
  } else {
    return mRegisterValue[reg].getValue(det, rob, mcm);
  }
}

bool TrapConfig::setTrapReg(TrapReg_t reg, int value, int det)
{
  // set a value for the given TRAP register on all chambers,

  return mRegisterValue[reg].setValue(value, det);
}

bool TrapConfig::setTrapReg(TrapReg_t reg, int value, int det, int rob, int mcm)
{
  // set the value for the given TRAP register of an individual MCM

  return mRegisterValue[reg].setValue(value, det, rob, mcm);
}

unsigned int TrapConfig::peek(int addr, int det, int rob, int mcm)
{
  // reading from given address

  if ((addr >= mgkDmemStartAddress) &&
      (addr < (mgkDmemStartAddress + mgkDmemWords))) {
    return getDmemUnsigned(addr, det, rob, mcm);
  } else {
    TrapReg_t mcmReg = getRegByAddress(addr);
    if (mcmReg >= 0 && mcmReg < kLastReg) {
      return (unsigned int)getTrapReg(mcmReg, det, rob, mcm);
    }
  }

  LOG(error) << "peek for invalid addr: 0x" << hex << setw(4) << addr;
  return 0;
}

bool TrapConfig::poke(int addr, unsigned int value, int det, int rob, int mcm)
{
  // writing to given address

  if ((addr >= mgkDmemStartAddress) &&
      (addr < (mgkDmemStartAddress + mgkDmemWords))) {
    LOG(debug1) << "DMEM 0x" << hex << setw(8) << addr << ": " << dec << value;
    return setDmem(addr, value, det, rob, mcm);
  } else {
    TrapReg_t mcmReg = getRegByAddress(addr);
    if (mcmReg >= 0 && mcmReg < kLastReg) {
      LOG(debug1) << "Register: " << getRegName(mcmReg) << " : " << value;
      return setTrapReg(mcmReg, (unsigned int)value, det, rob, mcm);
    }
  }

  LOG(error) << "poke for invalid address: 0x" << hex << setw(4) << addr;
  return false;
}

bool TrapConfig::setDmemAlloc(int addr, Alloc_t mode)
{
  addr = addr - mgkDmemStartAddress;

  if (addr < 0 || addr >= mgkDmemWords) {
    LOG(error) << "Invalid DMEM address: 0x%04x" << hex << std::setw(4) << addr + mgkDmemStartAddress;
    return false;
  } else {
    mDmem[addr].allocate(mode);
    return true;
  }
}

bool TrapConfig::setDmem(int addr, unsigned int value, int det)
{
  // set the content of the given DMEM address

  addr = addr - mgkDmemStartAddress;

  if (addr < 0 || addr >= mgkDmemWords) {
    LOG(error) << "No DMEM address: 0x" << hex << std::setw(8) << addr + mgkDmemStartAddress;
    return false;
  }

  if (!mDmem[addr].setValue(value, det)) {
    LOG(error) << "Problem writing to DMEM address 0x" << hex << std::setw(4) << addr;
    return false;
  } else {
    return true;
  }
}

bool TrapConfig::setDmem(int addr, unsigned int value, int det, int rob, int mcm)
{
  // set the content of the given DMEM address
  addr = addr - mgkDmemStartAddress;

  if (addr < 0 || addr >= mgkDmemWords) {
    LOG(error) << "No DMEM address: 0x" << hex << std::setw(8) << addr + mgkDmemStartAddress;
    return false;
  }

  if (!mDmem[addr].setValue(value, det, rob, mcm)) {
    LOG(error) << "Problem writing to DMEM address 0x" << hex << std::setw(4) << addr;
    return false;
  } else {
    return true;
  }
}

unsigned int TrapConfig::getDmemUnsigned(int addr, int det, int rob, int mcm)
{
  // get the content of the data memory at the given address
  // (only if the value is the same for all MCMs)

  addr = addr - mgkDmemStartAddress;

  if (addr < 0 || addr >= mgkDmemWords) {
    LOG(error) << "No DMEM address: 0x" << hex << std::setw(8) << addr + mgkDmemStartAddress;
    return 0;
  }

  return mDmem[addr].getValue(det, rob, mcm);
}

bool TrapConfig::printTrapReg(TrapReg_t reg, int det, int rob, int mcm)
{
  // print the value stored in the given register
  // if it is individual a valid MCM has to be specified

  if ((det >= 0 && det < MAXCHAMBER) &&
      (rob >= 0 && rob < NROBC1) &&
      (mcm >= 0 && mcm < NMCMROB + 2)) {
    LOG(info) << getRegName((TrapReg_t)reg) << "(" << std::setw(2) << getRegNBits((TrapReg_t)reg)
              << " bits) at 0x" << hex << std::setw(4) << getRegAddress((TrapReg_t)reg)
              << " is 0x" << hex << std::setw(8) << mRegisterValue[reg].getValue(det, rob, mcm)
              << " and resets to: 0x" << hex << std::setw(8) << getRegResetValue((TrapReg_t)reg)
              << " (currently individual mode)";
  } else {
    LOG(error) << "Register value is MCM-specific: Invalid detector, ROB or MCM requested";
    return false;
  }

  return true;
}

bool TrapConfig::printTrapAddr(int addr, int det, int rob, int mcm)
{
  // print the value stored at the given address in the MCM chip
  TrapReg_t reg = getRegByAddress(addr);
  if (reg >= 0 && reg < kLastReg) {
    return printTrapReg(reg, det, rob, mcm);
  } else {
    LOG(error) << "There is no register at address 0x" << hex << setw(8) << addr << " in the simulator";
    return false;
  }
}

TrapConfig::TrapReg_t TrapConfig::getRegByAddress(int address)
{
  // get register by its address
  // used for reading of configuration data as sent to real FEE

  if (address < mgkRegisterAddressBlockStart[0]) {
    return kLastReg;
  } else if (address < mgkRegisterAddressBlockStart[0] + mgkRegisterAddressBlockSize[0]) {
    return mgRegAddressMap[address - mgkRegisterAddressBlockStart[0]];
  } else if (address < mgkRegisterAddressBlockStart[1]) {
    return kLastReg;
  } else if (address < mgkRegisterAddressBlockStart[1] + mgkRegisterAddressBlockSize[1]) {
    return mgRegAddressMap[address - mgkRegisterAddressBlockStart[1] + mgkRegisterAddressBlockSize[0]];
  } else if (address < mgkRegisterAddressBlockStart[2]) {
    return kLastReg;
  } else if (address < mgkRegisterAddressBlockStart[2] + mgkRegisterAddressBlockSize[2]) {
    return mgRegAddressMap[address - mgkRegisterAddressBlockStart[2] + mgkRegisterAddressBlockSize[1] + mgkRegisterAddressBlockSize[0]];
  } else {
    return kLastReg;
  }
}

void TrapConfig::printMemDatx(ostream& os, int addr)
{
  // print the content of the data memory as datx

  printMemDatx(os, addr, 0, 0, 127);
}

void TrapConfig::printMemDatx(ostream& os, int addr, int det, int rob, int mcm)
{
  // print the content of the data memory as datx

  if (addr < mgkDmemStartAddress || addr >= mgkDmemStartAddress + mgkDmemWords) {
    LOG(error) << "Invalid DMEM address 0x" << hex << setw(8) << addr;
    return;
  }
  printDatx(os, addr, getDmemUnsigned(addr, det, rob, mcm), rob, mcm);
}

void TrapConfig::printMemDatx(ostream& os, TrapReg_t reg)
{
  // print the content of the data memory as datx

  printMemDatx(os, reg, 0, 0, 127);
}

void TrapConfig::printMemDatx(ostream& os, TrapReg_t reg, int det, int rob, int mcm)
{
  // print the content of the data memory as datx

  if (reg >= kLastReg) {
    LOG(error) << "Invalid register " << dec << reg;
    return;
  }
  printDatx(os, getRegAddress(reg), getTrapReg(reg, det, rob, mcm), rob, mcm);
}

void TrapConfig::printDatx(ostream& os, unsigned int addr, unsigned int data, int rob, int mcm)
{
  // print the value at the given address as datx

  os << std::setw(5) << 10
     << std::setw(8) << addr
     << std::setw(12) << data;
  if (mcm == 127) {
    os << std::setw(8) << 127;
  } else {
    os << std::setw(8) << FeeParam::aliToExtAli(rob, mcm);
  }

  os << std::endl;
}

void TrapConfig::printVerify(ostream& os, int det, int rob, int mcm)
{
  // print verification file in datx format

  for (int iReg = 0; iReg < kLastReg; ++iReg) {
    os << std::setw(5) << 9
       << std::setw(8) << getRegAddress((TrapReg_t)iReg)
       << std::setw(12) << getTrapReg((TrapReg_t)iReg, det, rob, mcm)
       << std::setw(8) << FeeParam::aliToExtAli(rob, mcm)
       << std::endl;
  }

  for (int iWord = 0; iWord < mgkDmemWords; ++iWord) {
    if (getDmemUnsigned(mgkDmemStartAddress + iWord, det, rob, mcm) == 0) {
      continue;
    }
    os << std::setw(5) << 9
       << std::setw(8) << mgkDmemStartAddress + iWord
       << std::setw(12) << getDmemUnsigned(mgkDmemStartAddress + iWord, det, rob, mcm)
       << std::setw(8) << FeeParam::aliToExtAli(rob, mcm)
       << std::endl;
  }
}

TrapConfig::TrapValue::TrapValue() : mAllocMode(kAllocGlobal)
{
  mData.resize(1);
  mValid.resize(1);
  mData[0] = 0;
  mValid[0] = true;
}

bool TrapConfig::TrapValue::allocate(Alloc_t alloc)
{
  // allocate memory for the specified granularity
  mAllocMode = alloc;
  int mSize = mgkSize[mAllocMode];

  if (mSize > 0) {
    mData.resize(mSize);
    mValid.resize(mSize);
    for (int i = 0; i < mSize; ++i) {
      mData[i] = 0;
      mValid[i] = false;
    }
  }

  return true;
}

// this exists purely to read in from ocdb to ccdb, and get around a root dictionary error of Alloc_t
bool TrapConfig::TrapValue::allocatei(int alloc)
{
  // allocate memory for the specified granularity
  mAllocMode = (Alloc_t)alloc;
  int mSize = mgkSize[mAllocMode];
  //cout << "in allocatei : with alloc = " << alloc << " and mSize is now :" << mSize << endl;
  if (mSize > 0) {
    mData.resize(mSize);
    mValid.resize(mSize);
    for (int i = 0; i < mSize; ++i) {
      mData[i] = 0;
      mValid[i] = false;
    }
  }

  return true;
}

int TrapConfig::TrapValue::getIdx(int det, int rob, int mcm)
{
  // return Idx to access the data for the given position

  int idx = -1;

  switch (mAllocMode) {
    case kAllocNone:
      idx = -1;
      break;
    case kAllocGlobal:
      idx = 0;
      break;
    case kAllocByDetector:
      idx = det;
      break;
    case kAllocByHC:
      idx = det + (rob % 2);
      break;
    case kAllocByMCM:
      idx = 18 * 8 * det + 18 * rob + mcm;
      break;
    case kAllocByLayer:
      idx = det % 6;
      break;
    case kAllocByMCMinSM:
      idx = 18 * 8 * (det % 30) + 18 * rob + mcm;
      break;
    default:
      idx = -1;
      LOG(error) << "Invalid allocation mode";
  }
  if (idx < mData.size()) {
    // LOG(info) << "Index ok " << dec << idx << " (size " << mData.size() << ") for " << this->getName() << " getIdx : " << det <<"::"<< rob<< "::" << mcm << "::" << mAllocMode;
    return idx;
  } else {
    LOG(warn) << "Index too large " << dec << idx << " (size " << mData.size() << ") for "
              << " getIdx : " << det << "::" << rob << "::" << mcm << "::" << mAllocMode;
    return -1;
  }
}

bool TrapConfig::TrapValue::setData(unsigned int value)
{
  // set the given value everywhere

  for (int i = 0; i < mData.size(); ++i) {
    mData[i] = value;
    mValid[i] = false;
  }

  return true;
}

bool TrapConfig::TrapValue::setData(unsigned int value, int det)
{
  // set the data for a given detector

  int idx = getIdx(det, 0, 0);

  if (idx >= 0) {
    // short cut for detector-wise allocation
    if (mAllocMode == kAllocByDetector) {
      if (mValid[idx] && (mData[idx] != value)) {
        LOG(debug) << "Overwriting previous value " << dec << mData[idx] << " of "
                   << " with " << value << " for " << det;
      }
      mData[idx] = value;
      mValid[idx] = true;
      return true;
    } else {
      for (int rob = 0; rob < 8; ++rob) {
        for (int mcm = 0; mcm < 18; ++mcm) {
          idx = getIdx(det, rob, mcm);
          if (mValid[idx] && (mData[idx] != value)) {
            LOG(debug) << "Overwriting previous value " << mData[idx] << " of "
                       << " with " << value << " for " << det << " " << rob << ":" << setw(2) << mcm;
          }
          mData[idx] = value;
          mValid[idx] = true;
        }
      }
      return true;
    }
  }

  if (mAllocMode == kAllocNone) {
    // assume nobody cares
    return true;
  }
  return false;
}

bool TrapConfig::TrapValue::setData(unsigned int value, int det, int rob, int mcm)
{
  // set data for an individual MCM

  int idx = getIdx(det, rob, mcm);

  if (idx >= 0) {
    if (mValid[idx] && (mData[idx] != value)) {
      LOG(debug) << "Overwriting previous value " << mData[idx] << " of "
                 << " with " << value << " " << det << ":" << rob << std::setw(2) << mcm << " (idx: " << idx << ")";
    }
    mData[idx] = value;
    mValid[idx] = true;
    return true;
  } else if (mAllocMode == kAllocNone) {
    return true;
  } else {
    LOG(error) << Form("setting failed");
    return false;
  }
}

unsigned int TrapConfig::TrapValue::getData(int det, int rob, int mcm)
{
  // read data for the given MCM

  int idx = getIdx(det, rob, mcm);
  if (idx >= 0) {
    if (!mValid[idx]) {
      LOG(debug1) << "reading from unwritten address: "
                  << " at idx " << idx << ":" << mValid[idx];
    }
    return mData[idx];
  } else {
    LOG(error) << "read from invalid address";
    return 0;
  }
}

TrapConfig::TrapRegister::TrapRegister() : TrapValue(), mName("invalid"), mAddr(0), mNbits(0), mResetValue(0)
{
  // default constructor
}

TrapConfig::TrapRegister::~TrapRegister() = default;

void TrapConfig::TrapRegister::init(const char* name, int addr, int nBits, int resetValue)
{
  // init the TRAP register

  if (mAddr == 0) {
    mName = name;
    mAddr = addr;
    mNbits = nBits;
    mResetValue = resetValue;
  } else {
    LOG(fatal) << "Re-initialising an existing TRAP register ";
  }
}

void TrapConfig::TrapRegister::initfromrun2(const char* name, int addr, int nBits, int resetValue)
{
  // init the TRAP register

  mName = name;
  mAddr = addr;
  mNbits = nBits;
  mResetValue = resetValue;
  //LOG(fatal) << "Re-initialising an existing TRAP register " << name << ":" << mName << " : " << addr << ":" << mAddr << " : " << nBits << ":" << mNbits <<  " : " << resetValue << ":" << mResetValue;
  //LOG(fatal) << "Re-initialising an existing TRAP register";
}

void TrapConfig::PrintDmemValue3(TrapConfig::TrapDmemWord* trapval, std::ofstream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << std::endl;
  output << "\t Array size : " << trapval->getDataSize() << std::endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << std::endl;
  }
}

void TrapConfig::PrintRegisterValue3(TrapConfig::TrapRegister* trapval, std::ofstream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << std::endl;
  output << "\t Array size : " << trapval->getDataSize() << std::endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << std::endl;
  }
}

void TrapConfig::DumpTrapConfig2File(std::string filename)
{
  std::ofstream outfile(filename);
  outfile << "Trap Registers : " << std::endl;
  for (int regvalue = 0; regvalue < TrapConfig::kLastReg; regvalue++) {
    outfile << " Trap : " << mRegisterValue[regvalue].getName()
            << " at : 0x " << std::hex << mRegisterValue[regvalue].getAddr() << std::dec
            << " with nbits : " << mRegisterValue[regvalue].getNbits()
            << " and reset value of : " << mRegisterValue[regvalue].getResetValue() << std::endl;
    // now for the inherited AliTRDtrapValue members;
    PrintRegisterValue3(&mRegisterValue[regvalue], outfile);
  }

  //  outfile << "done with regiser values now for dmemwords" << std::endl;
  outfile << "DMEM Words : " << std::endl;
  for (int dmemwords = 0; dmemwords < TrapConfig::mgkDmemWords; dmemwords++) {
    // copy fName, fAddr
    // inherited from trapvalue : fAllocMode, fSize fData and fValid
    //        trapconfig->mDmem[dmemwords].mName= run2config->fDmem[dmemwords].fName; // this gets set on setting the address
    outfile << "Name : " << mDmem[dmemwords].getName() << " :address : " << mDmem[dmemwords].getAddress() << std::endl;
    PrintDmemValue3(&mDmem[dmemwords], outfile);
  }
}

void TrapConfig::configureOnlineGains()
{
  // we dont want to do this anymore .... but here for future reference.
  /* if (hasOnlineFilterGain()) {
    const int nDets = MAXCHAMBER;
    const int nMcms = Geometry::MCMmax();
    const int nChs = Geometry::ADCmax();

    for (int ch = 0; ch < nChs; ++ch) {
      TrapConfig::TrapReg_t regFGAN = (TrapConfig::TrapReg_t)(TrapConfig::kFGA0 + ch);
      TrapConfig::TrapReg_t regFGFN = (TrapConfig::TrapReg_t)(TrapConfig::kFGF0 + ch);
      mTrapConfig->setTrapRegAlloc(regFGAN, TrapConfig::kAllocByMCM);
      mTrapConfig->setTrapRegAlloc(regFGFN, TrapConfig::kAllocByMCM);
    }

    for (int iDet = 0; iDet < nDets; ++iDet) {
      //const int MaxRows = Geometry::getStack(iDet) == 2 ? NROWC0 : NROWC1;
      int MaxCols = NCOLUMN;
      //CalOnlineGainTableROC gainTbl = mGainTable.getGainTableROC(iDet);
      const int nRobs = Geometry::getStack(iDet) == 2 ? Geometry::ROBmaxC0() : Geometry::ROBmaxC1();

      for (int rob = 0; rob < nRobs; ++rob) {
        for (int mcm = 0; mcm < nMcms; ++mcm) {
          //          for (int row = 0; row < MaxRows; row++) {
          //            for (int col = 0; col < MaxCols; col++) {
          //TODO  set ADC reference voltage
          mTrapConfig->setTrapReg(TrapConfig::kADCDAC, mTrapConfig.getAdcdacrm(iDet, rob, mcm), iDet, rob, mcm);

          // set constants channel-wise
          for (int ch = 0; ch < nChs; ++ch) {
            TrapConfig::TrapReg_t regFGAN = (TrapConfig::TrapReg_t)(TrapConfig::kFGA0 + ch);
            TrapConfig::TrapReg_t regFGFN = (TrapConfig::TrapReg_t)(TrapConfig::kFGF0 + ch);
            mTrapConfig->setTrapReg(regFGAN, mTrapConfig.getFGANrm(iDet, rob, mcm, ch), iDet, rob, mcm); //TODO can put these internal to TrapConfig
            mTrapConfig->setTrapReg(regFGFN, mTrapConfig.getFGFNrm(iDet, rob, mcm, ch), iDet, rob, mcm);
          }
        }
        //        }
        //    }
      }
    }
        }*/
}
