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

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRAP config as received from the trap chips in the form of            //
//  configuration events, configs are packed into digit events.           //
//  Header major version defines the config event payload follows as a    //
//  known block.                                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TrapRegisters.h"

#include <fairlogger/Logger.h>

#include <array>
#include <map>

using namespace o2::trd;

TrapRegisters::TrapRegisters()
{
  initialiseRegisters();
}

void TrapRegisters::initialiseRegisters()
{

  // initialize all TRAP registers
  mTrapRegisters[kTPL00].init("TPL00", 0x3180, 5, 0, 0, false, 5);
  mTrapRegisters[kTPL01].init("TPL01", 0x3181, 5, 0, 1, false, 5);
  mTrapRegisters[kTPL02].init("TPL02", 0x3182, 5, 0, 2, false, 5);
  mTrapRegisters[kTPL03].init("TPL03", 0x3183, 5, 0, 3, false, 5);
  mTrapRegisters[kTPL04].init("TPL04", 0x3184, 5, 0, 4, false, 5);
  mTrapRegisters[kTPL05].init("TPL05", 0x3185, 5, 0, 5, false, 5);
  mTrapRegisters[kTPL06].init("TPL06", 0x3186, 5, 0, 6, false, 5);
  mTrapRegisters[kTPL07].init("TPL07", 0x3187, 5, 0, 7, false, 5);
  mTrapRegisters[kTPL08].init("TPL08", 0x3188, 5, 0, 8, false, 5);
  mTrapRegisters[kTPL09].init("TPL09", 0x3189, 5, 0, 9, false, 5);
  mTrapRegisters[kTPL0A].init("TPL0A", 0x318A, 5, 0, 10, false, 5);
  mTrapRegisters[kTPL0B].init("TPL0B", 0x318B, 5, 0, 11, false, 5);
  mTrapRegisters[kTPL0C].init("TPL0C", 0x318C, 5, 0, 12, false, 5);
  mTrapRegisters[kTPL0D].init("TPL0D", 0x318D, 5, 0, 13, false, 5);
  mTrapRegisters[kTPL0E].init("TPL0E", 0x318E, 5, 0, 14, false, 5);
  mTrapRegisters[kTPL0F].init("TPL0F", 0x318F, 5, 0, 15, false, 5);
  mTrapRegisters[kTPL10].init("TPL10", 0x3190, 5, 0, 16, false, 5);
  mTrapRegisters[kTPL11].init("TPL11", 0x3191, 5, 0, 17, false, 5);
  mTrapRegisters[kTPL12].init("TPL12", 0x3192, 5, 0, 18, false, 5);
  mTrapRegisters[kTPL13].init("TPL13", 0x3193, 5, 0, 19, false, 5);
  mTrapRegisters[kTPL14].init("TPL14", 0x3194, 5, 0, 20, false, 5);
  mTrapRegisters[kTPL15].init("TPL15", 0x3195, 5, 0, 21, false, 5);
  mTrapRegisters[kTPL16].init("TPL16", 0x3196, 5, 0, 22, false, 5);
  mTrapRegisters[kTPL17].init("TPL17", 0x3197, 5, 0, 23, false, 5);
  mTrapRegisters[kTPL18].init("TPL18", 0x3198, 5, 0, 24, false, 5);
  mTrapRegisters[kTPL19].init("TPL19", 0x3199, 5, 0, 25, false, 5);
  mTrapRegisters[kTPL1A].init("TPL1A", 0x319A, 5, 0, 26, false, 5);
  mTrapRegisters[kTPL1B].init("TPL1B", 0x319B, 5, 0, 27, false, 5);
  mTrapRegisters[kTPL1C].init("TPL1C", 0x319C, 5, 0, 28, false, 5);
  mTrapRegisters[kTPL1D].init("TPL1D", 0x319D, 5, 0, 29, false, 5);
  mTrapRegisters[kTPL1E].init("TPL1E", 0x319E, 5, 0, 30, false, 5);
  mTrapRegisters[kTPL1F].init("TPL1F", 0x319F, 5, 0, 31, false, 5);
  mTrapRegisters[kTPL20].init("TPL20", 0x31A0, 5, 0, 32, false, 5);
  mTrapRegisters[kTPL21].init("TPL21", 0x31A1, 5, 0, 33, false, 5);
  mTrapRegisters[kTPL22].init("TPL22", 0x31A2, 5, 0, 34, false, 5);
  mTrapRegisters[kTPL23].init("TPL23", 0x31A3, 5, 0, 35, false, 5);
  mTrapRegisters[kTPL24].init("TPL24", 0x31A4, 5, 0, 36, false, 5);
  mTrapRegisters[kTPL25].init("TPL25", 0x31A5, 5, 0, 37, false, 5);
  mTrapRegisters[kTPL26].init("TPL26", 0x31A6, 5, 0, 38, false, 5);
  mTrapRegisters[kTPL27].init("TPL27", 0x31A7, 5, 0, 39, false, 5);
  mTrapRegisters[kTPL28].init("TPL28", 0x31A8, 5, 0, 40, false, 5);
  mTrapRegisters[kTPL29].init("TPL29", 0x31A9, 5, 0, 41, false, 5);
  mTrapRegisters[kTPL2A].init("TPL2A", 0x31AA, 5, 0, 42, false, 5);
  mTrapRegisters[kTPL2B].init("TPL2B", 0x31AB, 5, 0, 43, false, 5);
  mTrapRegisters[kTPL2C].init("TPL2C", 0x31AC, 5, 0, 44, false, 5);
  mTrapRegisters[kTPL2D].init("TPL2D", 0x31AD, 5, 0, 45, false, 5);
  mTrapRegisters[kTPL2E].init("TPL2E", 0x31AE, 5, 0, 46, false, 5);
  mTrapRegisters[kTPL2F].init("TPL2F", 0x31AF, 5, 0, 47, false, 5);
  mTrapRegisters[kTPL30].init("TPL30", 0x31B0, 5, 0, 48, false, 5);
  mTrapRegisters[kTPL31].init("TPL31", 0x31B1, 5, 0, 49, false, 5);
  mTrapRegisters[kTPL32].init("TPL32", 0x31B2, 5, 0, 50, false, 5);
  mTrapRegisters[kTPL33].init("TPL33", 0x31B3, 5, 0, 51, false, 5);
  mTrapRegisters[kTPL34].init("TPL34", 0x31B4, 5, 0, 52, false, 5);
  mTrapRegisters[kTPL35].init("TPL35", 0x31B5, 5, 0, 53, false, 5);
  mTrapRegisters[kTPL36].init("TPL36", 0x31B6, 5, 0, 54, false, 5);
  mTrapRegisters[kTPL37].init("TPL37", 0x31B7, 5, 0, 55, false, 5);
  mTrapRegisters[kTPL38].init("TPL38", 0x31B8, 5, 0, 56, false, 5);
  mTrapRegisters[kTPL39].init("TPL39", 0x31B9, 5, 0, 57, false, 5);
  mTrapRegisters[kTPL3A].init("TPL3A", 0x31BA, 5, 0, 58, false, 5);
  mTrapRegisters[kTPL3B].init("TPL3B", 0x31BB, 5, 0, 59, false, 5);
  mTrapRegisters[kTPL3C].init("TPL3C", 0x31BC, 5, 0, 60, false, 5);
  mTrapRegisters[kTPL3D].init("TPL3D", 0x31BD, 5, 0, 61, false, 5);
  mTrapRegisters[kTPL3E].init("TPL3E", 0x31BE, 5, 0, 62, false, 5);
  mTrapRegisters[kTPL3F].init("TPL3F", 0x31BF, 5, 0, 63, false, 5);
  mTrapRegisters[kTPL40].init("TPL40", 0x31C0, 5, 0, 64, false, 5);
  mTrapRegisters[kTPL41].init("TPL41", 0x31C1, 5, 0, 65, false, 5);
  mTrapRegisters[kTPL42].init("TPL42", 0x31C2, 5, 0, 66, false, 5);
  mTrapRegisters[kTPL43].init("TPL43", 0x31C3, 5, 0, 67, false, 5);
  mTrapRegisters[kTPL44].init("TPL44", 0x31C4, 5, 0, 68, false, 5);
  mTrapRegisters[kTPL45].init("TPL45", 0x31C5, 5, 0, 69, false, 5);
  mTrapRegisters[kTPL46].init("TPL46", 0x31C6, 5, 0, 70, false, 5);
  mTrapRegisters[kTPL47].init("TPL47", 0x31C7, 5, 0, 71, false, 5);
  mTrapRegisters[kTPL48].init("TPL48", 0x31C8, 5, 0, 72, false, 5);
  mTrapRegisters[kTPL49].init("TPL49", 0x31C9, 5, 0, 73, false, 5);
  mTrapRegisters[kTPL4A].init("TPL4A", 0x31CA, 5, 0, 74, false, 5);
  mTrapRegisters[kTPL4B].init("TPL4B", 0x31CB, 5, 0, 75, false, 5);
  mTrapRegisters[kTPL4C].init("TPL4C", 0x31CC, 5, 0, 76, false, 5);
  mTrapRegisters[kTPL4D].init("TPL4D", 0x31CD, 5, 0, 77, false, 5);
  mTrapRegisters[kTPL4E].init("TPL4E", 0x31CE, 5, 0, 78, false, 5);
  mTrapRegisters[kTPL4F].init("TPL4F", 0x31CF, 5, 0, 79, false, 5);
  mTrapRegisters[kTPL50].init("TPL50", 0x31D0, 5, 0, 80, false, 5);
  mTrapRegisters[kTPL51].init("TPL51", 0x31D1, 5, 0, 81, false, 5);
  mTrapRegisters[kTPL52].init("TPL52", 0x31D2, 5, 0, 82, false, 5);
  mTrapRegisters[kTPL53].init("TPL53", 0x31D3, 5, 0, 83, false, 5);
  mTrapRegisters[kTPL54].init("TPL54", 0x31D4, 5, 0, 84, false, 5);
  mTrapRegisters[kTPL55].init("TPL55", 0x31D5, 5, 0, 85, false, 5);
  mTrapRegisters[kTPL56].init("TPL56", 0x31D6, 5, 0, 86, false, 5);
  mTrapRegisters[kTPL57].init("TPL57", 0x31D7, 5, 0, 87, false, 5);
  mTrapRegisters[kTPL58].init("TPL58", 0x31D8, 5, 0, 88, false, 5);
  mTrapRegisters[kTPL59].init("TPL59", 0x31D9, 5, 0, 89, false, 5);
  mTrapRegisters[kTPL5A].init("TPL5A", 0x31DA, 5, 0, 90, false, 5);
  mTrapRegisters[kTPL5B].init("TPL5B", 0x31DB, 5, 0, 91, false, 5);
  mTrapRegisters[kTPL5C].init("TPL5C", 0x31DC, 5, 0, 92, false, 5);
  mTrapRegisters[kTPL5D].init("TPL5D", 0x31DD, 5, 0, 93, false, 5);
  mTrapRegisters[kTPL5E].init("TPL5E", 0x31DE, 5, 0, 94, false, 5);
  mTrapRegisters[kTPL5F].init("TPL5F", 0x31DF, 5, 0, 95, false, 5);
  mTrapRegisters[kTPL60].init("TPL60", 0x31E0, 5, 0, 96, false, 5);
  mTrapRegisters[kTPL61].init("TPL61", 0x31E1, 5, 0, 97, false, 5);
  mTrapRegisters[kTPL62].init("TPL62", 0x31E2, 5, 0, 98, false, 5);
  mTrapRegisters[kTPL63].init("TPL63", 0x31E3, 5, 0, 99, false, 5);
  mTrapRegisters[kTPL64].init("TPL64", 0x31E4, 5, 0, 100, false, 5);
  mTrapRegisters[kTPL65].init("TPL65", 0x31E5, 5, 0, 101, false, 5);
  mTrapRegisters[kTPL66].init("TPL66", 0x31E6, 5, 0, 102, false, 5);
  mTrapRegisters[kTPL67].init("TPL67", 0x31E7, 5, 0, 103, false, 5);
  mTrapRegisters[kTPL68].init("TPL68", 0x31E8, 5, 0, 104, false, 5);
  mTrapRegisters[kTPL69].init("TPL69", 0x31E9, 5, 0, 105, false, 5);
  mTrapRegisters[kTPL6A].init("TPL6A", 0x31EA, 5, 0, 106, false, 5);
  mTrapRegisters[kTPL6B].init("TPL6B", 0x31EB, 5, 0, 107, false, 5);
  mTrapRegisters[kTPL6C].init("TPL6C", 0x31EC, 5, 0, 108, false, 5);
  mTrapRegisters[kTPL6D].init("TPL6D", 0x31ED, 5, 0, 109, false, 5);
  mTrapRegisters[kTPL6E].init("TPL6E", 0x31EE, 5, 0, 110, false, 5);
  mTrapRegisters[kTPL6F].init("TPL6F", 0x31EF, 5, 0, 111, false, 5);
  mTrapRegisters[kTPL70].init("TPL70", 0x31F0, 5, 0, 112, false, 5);
  mTrapRegisters[kTPL71].init("TPL71", 0x31F1, 5, 0, 113, false, 5);
  mTrapRegisters[kTPL72].init("TPL72", 0x31F2, 5, 0, 114, false, 5);
  mTrapRegisters[kTPL73].init("TPL73", 0x31F3, 5, 0, 115, false, 5);
  mTrapRegisters[kTPL74].init("TPL74", 0x31F4, 5, 0, 116, false, 5);
  mTrapRegisters[kTPL75].init("TPL75", 0x31F5, 5, 0, 117, false, 5);
  mTrapRegisters[kTPL76].init("TPL76", 0x31F6, 5, 0, 118, false, 5);
  mTrapRegisters[kTPL77].init("TPL77", 0x31F7, 5, 0, 119, false, 5);
  mTrapRegisters[kTPL78].init("TPL78", 0x31F8, 5, 0, 120, false, 5);
  mTrapRegisters[kTPL79].init("TPL79", 0x31F9, 5, 0, 121, false, 5);
  mTrapRegisters[kTPL7A].init("TPL7A", 0x31FA, 5, 0, 122, false, 5);
  mTrapRegisters[kTPL7B].init("TPL7B", 0x31FB, 5, 0, 123, false, 5);
  mTrapRegisters[kTPL7C].init("TPL7C", 0x31FC, 5, 0, 124, false, 5);
  mTrapRegisters[kTPL7D].init("TPL7D", 0x31FD, 5, 0, 125, false, 5);
  mTrapRegisters[kTPL7E].init("TPL7E", 0x31FE, 5, 0, 126, false, 5);
  mTrapRegisters[kTPL7F].init("TPL7F", 0x31FF, 5, 0, 127, false, 5);
  mTrapRegisters[kFGA0].init("FGA0", 0x30A0, 6, 22, 0, false, 6);
  mTrapRegisters[kFGA1].init("FGA1", 0x30A1, 6, 22, 1, false, 6);
  mTrapRegisters[kFGA2].init("FGA2", 0x30A2, 6, 22, 2, false, 6);
  mTrapRegisters[kFGA3].init("FGA3", 0x30A3, 6, 22, 3, false, 6);
  mTrapRegisters[kFGA4].init("FGA4", 0x30A4, 6, 22, 4, false, 6);
  mTrapRegisters[kFGA5].init("FGA5", 0x30A5, 6, 22, 5, false, 6);
  mTrapRegisters[kFGA6].init("FGA6", 0x30A6, 6, 22, 6, false, 6);
  mTrapRegisters[kFGA7].init("FGA7", 0x30A7, 6, 22, 7, false, 6);
  mTrapRegisters[kFGA8].init("FGA8", 0x30A8, 6, 22, 8, false, 6);
  mTrapRegisters[kFGA9].init("FGA9", 0x30A9, 6, 22, 9, false, 6);
  mTrapRegisters[kFGA10].init("FGA10", 0x30AA, 6, 22, 10, false, 6);
  mTrapRegisters[kFGA11].init("FGA11", 0x30AB, 6, 22, 11, false, 6);
  mTrapRegisters[kFGA12].init("FGA12", 0x30AC, 6, 22, 12, false, 6);
  mTrapRegisters[kFGA13].init("FGA13", 0x30AD, 6, 22, 13, false, 6);
  mTrapRegisters[kFGA14].init("FGA14", 0x30AE, 6, 22, 14, false, 6);
  mTrapRegisters[kFGA15].init("FGA15", 0x30AF, 6, 22, 15, false, 6);
  mTrapRegisters[kFGA16].init("FGA16", 0x30B0, 6, 22, 16, false, 6);
  mTrapRegisters[kFGA17].init("FGA17", 0x30B1, 6, 22, 17, false, 6);
  mTrapRegisters[kFGA18].init("FGA18", 0x30B2, 6, 22, 18, false, 6);
  mTrapRegisters[kFGA19].init("FGA19", 0x30B3, 6, 22, 19, false, 6);
  mTrapRegisters[kFGA20].init("FGA20", 0x30B4, 6, 22, 20, false, 6);
  mTrapRegisters[kFGF0].init("FGF0", 0x3080, 10, 27, 0, false, 10);
  mTrapRegisters[kFGF1].init("FGF1", 0x3081, 10, 27, 1, false, 10);
  mTrapRegisters[kFGF2].init("FGF2", 0x3082, 10, 27, 2, false, 10);
  mTrapRegisters[kFGF3].init("FGF3", 0x3083, 10, 27, 3, false, 10);
  mTrapRegisters[kFGF4].init("FGF4", 0x3084, 10, 27, 4, false, 10);
  mTrapRegisters[kFGF5].init("FGF5", 0x3085, 10, 27, 5, false, 10);
  mTrapRegisters[kFGF6].init("FGF6", 0x3086, 10, 27, 6, false, 10);
  mTrapRegisters[kFGF7].init("FGF7", 0x3087, 10, 27, 7, false, 10);
  mTrapRegisters[kFGF8].init("FGF8", 0x3088, 10, 27, 8, false, 10);
  mTrapRegisters[kFGF9].init("FGF9", 0x3089, 10, 27, 9, false, 10);
  mTrapRegisters[kFGF10].init("FGF10", 0x308A, 10, 27, 10, false, 10);
  mTrapRegisters[kFGF11].init("FGF11", 0x308B, 10, 27, 11, false, 10);
  mTrapRegisters[kFGF12].init("FGF12", 0x308C, 10, 27, 12, false, 10);
  mTrapRegisters[kFGF13].init("FGF13", 0x308D, 10, 27, 13, false, 10);
  mTrapRegisters[kFGF14].init("FGF14", 0x308E, 10, 27, 14, false, 10);
  mTrapRegisters[kFGF15].init("FGF15", 0x308F, 10, 27, 15, false, 10);
  mTrapRegisters[kFGF16].init("FGF16", 0x3090, 10, 27, 16, false, 10);
  mTrapRegisters[kFGF17].init("FGF17", 0x3091, 10, 27, 17, false, 10);
  mTrapRegisters[kFGF18].init("FGF18", 0x3092, 10, 27, 18, false, 10);
  mTrapRegisters[kFGF19].init("FGF19", 0x3093, 10, 27, 19, false, 10);
  mTrapRegisters[kFGF20].init("FGF20", 0x3094, 10, 27, 20, false, 10);
  mTrapRegisters[kCPU0CLK].init("CPU0CLK", 0x0A20, 5, 34, 0, false, 5);
  mTrapRegisters[kCPU1CLK].init("CPU1CLK", 0x0A22, 5, 34, 1, false, 5);
  mTrapRegisters[kCPU2CLK].init("CPU2CLK", 0x0A24, 5, 34, 2, false, 5);
  mTrapRegisters[kCPU3CLK].init("CPU3CLK", 0x0A26, 5, 34, 3, false, 5);
  mTrapRegisters[kNICLK].init("NICLK", 0x0A28, 5, 34, 4, false, 5);
  mTrapRegisters[kFILCLK].init("FILCLK", 0x0A2A, 5, 34, 5, false, 5);
  mTrapRegisters[kPRECLK].init("PRECLK", 0x0A2C, 5, 34, 6, false, 5);
  mTrapRegisters[kADCEN].init("ADCEN", 0x0A2E, 5, 34, 7, false, 5);
  mTrapRegisters[kNIODE].init("NIODE", 0x0A30, 5, 34, 8, false, 5);
  mTrapRegisters[kNIOCE].init("NIOCE", 0x0A32, 5, 34, 9, false, 5);
  mTrapRegisters[kNIIDE].init("NIIDE", 0x0A34, 5, 34, 10, false, 5);
  mTrapRegisters[kNIICE].init("NIICE", 0x0A36, 5, 34, 11, false, 5);
  mTrapRegisters[kEBIS].init("EBIS", 0x3014, 15, 36, 0, false, 15);
  mTrapRegisters[kEBIT].init("EBIT", 0x3015, 15, 36, 1, false, 15);
  mTrapRegisters[kEBIL].init("EBIL", 0x3016, 15, 36, 2, false, 15);
  mTrapRegisters[kTPVT].init("TPVT", 0x3042, 6, 38, 0, false, 6);
  mTrapRegisters[kTPVBY].init("TPVBY", 0x3043, 6, 38, 1, false, 6);
  mTrapRegisters[kTPCT].init("TPCT", 0x3044, 6, 38, 2, false, 6);
  mTrapRegisters[kTPCL].init("TPCL", 0x3045, 6, 38, 3, false, 6);
  mTrapRegisters[kTPCBY].init("TPCBY", 0x3046, 6, 38, 4, false, 6);
  mTrapRegisters[kTPD].init("TPD", 0x3047, 6, 38, 5, false, 6);
  mTrapRegisters[kTPCI0].init("TPCI0", 0x3048, 6, 38, 6, false, 6);
  mTrapRegisters[kTPCI1].init("TPCI1", 0x3049, 6, 38, 7, false, 6);
  mTrapRegisters[kTPCI2].init("TPCI2", 0x304A, 6, 38, 8, false, 6);
  mTrapRegisters[kTPCI3].init("TPCI3", 0x304B, 6, 38, 9, false, 6);
  mTrapRegisters[kEBIN].init("EBIN", 0x3017, 5, 40, 0, false, 5);
  mTrapRegisters[kFLBY].init("FLBY", 0x3018, 5, 40, 1, false, 5);
  mTrapRegisters[kFPBY].init("FPBY", 0x3019, 5, 40, 2, false, 5);
  mTrapRegisters[kFGBY].init("FGBY", 0x301A, 5, 40, 3, false, 5);
  mTrapRegisters[kFTBY].init("FTBY", 0x301B, 5, 40, 4, false, 5);
  mTrapRegisters[kFCBY].init("FCBY", 0x301C, 5, 40, 5, false, 5);
  mTrapRegisters[kFLL00].init("FLL00", 0x3100, 6, 41, 0, false, 6);
  mTrapRegisters[kFLL01].init("FLL01", 0x3101, 6, 41, 1, false, 6);
  mTrapRegisters[kFLL02].init("FLL02", 0x3102, 6, 41, 2, false, 6);
  mTrapRegisters[kFLL03].init("FLL03", 0x3103, 6, 41, 3, false, 6);
  mTrapRegisters[kFLL04].init("FLL04", 0x3104, 6, 41, 4, false, 6);
  mTrapRegisters[kFLL05].init("FLL05", 0x3105, 6, 41, 5, false, 6);
  mTrapRegisters[kFLL06].init("FLL06", 0x3106, 6, 41, 6, false, 6);
  mTrapRegisters[kFLL07].init("FLL07", 0x3107, 6, 41, 7, false, 6);
  mTrapRegisters[kFLL08].init("FLL08", 0x3108, 6, 41, 8, false, 6);
  mTrapRegisters[kFLL09].init("FLL09", 0x3109, 6, 41, 9, false, 6);
  mTrapRegisters[kFLL0A].init("FLL0A", 0x310A, 6, 41, 10, false, 6);
  mTrapRegisters[kFLL0B].init("FLL0B", 0x310B, 6, 41, 11, false, 6);
  mTrapRegisters[kFLL0C].init("FLL0C", 0x310C, 6, 41, 12, false, 6);
  mTrapRegisters[kFLL0D].init("FLL0D", 0x310D, 6, 41, 13, false, 6);
  mTrapRegisters[kFLL0E].init("FLL0E", 0x310E, 6, 41, 14, false, 6);
  mTrapRegisters[kFLL0F].init("FLL0F", 0x310F, 6, 41, 15, false, 6);
  mTrapRegisters[kFLL10].init("FLL10", 0x3110, 6, 41, 16, false, 6);
  mTrapRegisters[kFLL11].init("FLL11", 0x3111, 6, 41, 17, false, 6);
  mTrapRegisters[kFLL12].init("FLL12", 0x3112, 6, 41, 18, false, 6);
  mTrapRegisters[kFLL13].init("FLL13", 0x3113, 6, 41, 19, false, 6);
  mTrapRegisters[kFLL14].init("FLL14", 0x3114, 6, 41, 20, false, 6);
  mTrapRegisters[kFLL15].init("FLL15", 0x3115, 6, 41, 21, false, 6);
  mTrapRegisters[kFLL16].init("FLL16", 0x3116, 6, 41, 22, false, 6);
  mTrapRegisters[kFLL17].init("FLL17", 0x3117, 6, 41, 23, false, 6);
  mTrapRegisters[kFLL18].init("FLL18", 0x3118, 6, 41, 24, false, 6);
  mTrapRegisters[kFLL19].init("FLL19", 0x3119, 6, 41, 25, false, 6);
  mTrapRegisters[kFLL1A].init("FLL1A", 0x311A, 6, 41, 26, false, 6);
  mTrapRegisters[kFLL1B].init("FLL1B", 0x311B, 6, 41, 27, false, 6);
  mTrapRegisters[kFLL1C].init("FLL1C", 0x311C, 6, 41, 28, false, 6);
  mTrapRegisters[kFLL1D].init("FLL1D", 0x311D, 6, 41, 29, false, 6);
  mTrapRegisters[kFLL1E].init("FLL1E", 0x311E, 6, 41, 30, false, 6);
  mTrapRegisters[kFLL1F].init("FLL1F", 0x311F, 6, 41, 31, false, 6);
  mTrapRegisters[kFLL20].init("FLL20", 0x3120, 6, 41, 32, false, 6);
  mTrapRegisters[kFLL21].init("FLL21", 0x3121, 6, 41, 33, false, 6);
  mTrapRegisters[kFLL22].init("FLL22", 0x3122, 6, 41, 34, false, 6);
  mTrapRegisters[kFLL23].init("FLL23", 0x3123, 6, 41, 35, false, 6);
  mTrapRegisters[kFLL24].init("FLL24", 0x3124, 6, 41, 36, false, 6);
  mTrapRegisters[kFLL25].init("FLL25", 0x3125, 6, 41, 37, false, 6);
  mTrapRegisters[kFLL26].init("FLL26", 0x3126, 6, 41, 38, false, 6);
  mTrapRegisters[kFLL27].init("FLL27", 0x3127, 6, 41, 39, false, 6);
  mTrapRegisters[kFLL28].init("FLL28", 0x3128, 6, 41, 40, false, 6);
  mTrapRegisters[kFLL29].init("FLL29", 0x3129, 6, 41, 41, false, 6);
  mTrapRegisters[kFLL2A].init("FLL2A", 0x312A, 6, 41, 42, false, 6);
  mTrapRegisters[kFLL2B].init("FLL2B", 0x312B, 6, 41, 43, false, 6);
  mTrapRegisters[kFLL2C].init("FLL2C", 0x312C, 6, 41, 44, false, 6);
  mTrapRegisters[kFLL2D].init("FLL2D", 0x312D, 6, 41, 45, false, 6);
  mTrapRegisters[kFLL2E].init("FLL2E", 0x312E, 6, 41, 46, false, 6);
  mTrapRegisters[kFLL2F].init("FLL2F", 0x312F, 6, 41, 47, false, 6);
  mTrapRegisters[kFLL30].init("FLL30", 0x3130, 6, 41, 48, false, 6);
  mTrapRegisters[kFLL31].init("FLL31", 0x3131, 6, 41, 49, false, 6);
  mTrapRegisters[kFLL32].init("FLL32", 0x3132, 6, 41, 50, false, 6);
  mTrapRegisters[kFLL33].init("FLL33", 0x3133, 6, 41, 51, false, 6);
  mTrapRegisters[kFLL34].init("FLL34", 0x3134, 6, 41, 52, false, 6);
  mTrapRegisters[kFLL35].init("FLL35", 0x3135, 6, 41, 53, false, 6);
  mTrapRegisters[kFLL36].init("FLL36", 0x3136, 6, 41, 54, false, 6);
  mTrapRegisters[kFLL37].init("FLL37", 0x3137, 6, 41, 55, false, 6);
  mTrapRegisters[kFLL38].init("FLL38", 0x3138, 6, 41, 56, false, 6);
  mTrapRegisters[kFLL39].init("FLL39", 0x3139, 6, 41, 57, false, 6);
  mTrapRegisters[kFLL3A].init("FLL3A", 0x313A, 6, 41, 58, false, 6);
  mTrapRegisters[kFLL3B].init("FLL3B", 0x313B, 6, 41, 59, false, 6);
  mTrapRegisters[kFLL3C].init("FLL3C", 0x313C, 6, 41, 60, false, 6);
  mTrapRegisters[kFLL3D].init("FLL3D", 0x313D, 6, 41, 61, false, 6);
  mTrapRegisters[kFLL3E].init("FLL3E", 0x313E, 6, 41, 62, false, 6);
  mTrapRegisters[kFLL3F].init("FLL3F", 0x313F, 6, 41, 63, false, 6);
  mTrapRegisters[kTPPT0].init("TPPT0", 0x3000, 7, 54, 0, false, 7);
  mTrapRegisters[kTPFS].init("TPFS", 0x3001, 7, 54, 1, false, 7);
  mTrapRegisters[kTPFE].init("TPFE", 0x3002, 7, 54, 2, false, 7);
  mTrapRegisters[kTPPGR].init("TPPGR", 0x3003, 7, 54, 3, false, 7);
  mTrapRegisters[kTPPAE].init("TPPAE", 0x3004, 7, 54, 4, false, 7);
  mTrapRegisters[kTPQS0].init("TPQS0", 0x3005, 7, 54, 5, false, 7);
  mTrapRegisters[kTPQE0].init("TPQE0", 0x3006, 7, 54, 6, false, 7);
  mTrapRegisters[kTPQS1].init("TPQS1", 0x3007, 7, 54, 7, false, 7);
  mTrapRegisters[kTPQE1].init("TPQE1", 0x3008, 7, 54, 8, false, 7);
  mTrapRegisters[kEBD].init("EBD", 0x3009, 7, 54, 9, false, 7);
  mTrapRegisters[kEBAQA].init("EBAQA", 0x300A, 7, 54, 10, false, 7);
  mTrapRegisters[kEBSIA].init("EBSIA", 0x300B, 7, 54, 11, false, 7);
  mTrapRegisters[kEBSF].init("EBSF", 0x300C, 7, 54, 12, false, 7);
  mTrapRegisters[kEBSIM].init("EBSIM", 0x300D, 7, 54, 13, false, 7);
  mTrapRegisters[kEBPP].init("EBPP", 0x300E, 7, 54, 14, false, 7);
  mTrapRegisters[kEBPC].init("EBPC", 0x300F, 7, 54, 15, false, 7);
  mTrapRegisters[kFPTC].init("FPTC", 0x3020, 10, 58, 0, false, 10);
  mTrapRegisters[kFPNP].init("FPNP", 0x3021, 10, 58, 1, false, 10);
  mTrapRegisters[kFPCL].init("FPCL", 0x3022, 10, 58, 2, false, 10);
  mTrapRegisters[kFGTA].init("FGTA", 0x3028, 15, 59, 0, false, 15);
  mTrapRegisters[kFGTB].init("FGTB", 0x3029, 15, 59, 1, false, 15);
  mTrapRegisters[kFGCL].init("FGCL", 0x302A, 15, 59, 2, false, 15);
  mTrapRegisters[kFTAL].init("FTAL", 0x3030, 10, 61, 0, false, 10);
  mTrapRegisters[kFTLL].init("FTLL", 0x3031, 10, 61, 1, false, 10);
  mTrapRegisters[kFTLS].init("FTLS", 0x3032, 10, 61, 2, false, 10);
  mTrapRegisters[kFCW1].init("FCW1", 0x3038, 10, 62, 0, false, 10);
  mTrapRegisters[kFCW2].init("FCW2", 0x3039, 10, 62, 1, false, 10);
  mTrapRegisters[kFCW3].init("FCW3", 0x303A, 10, 62, 2, false, 10);
  mTrapRegisters[kFCW4].init("FCW4", 0x303B, 10, 62, 3, false, 10);
  mTrapRegisters[kFCW5].init("FCW5", 0x303C, 10, 62, 4, false, 10);
  mTrapRegisters[kTPFP].init("TPFP", 0x3040, 15, 64, 0, false, 15);
  mTrapRegisters[kTPHT].init("TPHT", 0x3041, 15, 64, 1, false, 15);
  mTrapRegisters[kADCMSK].init("ADCMSK", 0x3050, 32, 65, 0, false, 32);
  mTrapRegisters[kADCINB].init("ADCINB", 0x3051, 5, 66, 0, false, 5);
  mTrapRegisters[kADCDAC].init("ADCDAC", 0x3052, 5, 66, 1, false, 5);
  mTrapRegisters[kADCPAR].init("ADCPAR", 0x3053, 32, 67, 0, false, 32);
  mTrapRegisters[kADCTST].init("ADCTST", 0x3054, 5, 68, 0, false, 5);
  mTrapRegisters[kSADCAZ].init("SADCAZ", 0x3055, 5, 68, 1, false, 5);
  mTrapRegisters[kPASADEL].init("PASADEL", 0x3158, 10, 69, 0, false, 10);
  mTrapRegisters[kPASAPHA].init("PASAPHA", 0x3159, 10, 69, 1, false, 10);
  mTrapRegisters[kPASAPRA].init("PASAPRA", 0x315A, 10, 69, 2, false, 10);
  mTrapRegisters[kPASADAC].init("PASADAC", 0x315B, 10, 69, 3, false, 10);
  mTrapRegisters[kPASASTL].init("PASASTL", 0x315D, 10, 71, 0, false, 10);
  mTrapRegisters[kPASAPR1].init("PASAPR1", 0x315E, 10, 71, 1, false, 10);
  mTrapRegisters[kPASAPR0].init("PASAPR0", 0x315F, 10, 71, 2, false, 10);
  mTrapRegisters[kSADCTRG].init("SADCTRG", 0x3161, 5, 72, 0, false, 5);
  mTrapRegisters[kSADCRUN].init("SADCRUN", 0x3162, 5, 72, 1, false, 5);
  mTrapRegisters[kSADCPWR].init("SADCPWR", 0x3163, 5, 72, 2, false, 5);
  mTrapRegisters[kL0TSIM].init("L0TSIM", 0x3165, 15, 73, 0, false, 15);
  mTrapRegisters[kSADCEC].init("SADCEC", 0x3166, 15, 73, 1, false, 15);
  mTrapRegisters[kSADCMC].init("SADCMC", 0x3170, 10, 74, 0, false, 10);
  mTrapRegisters[kSADCOC].init("SADCOC", 0x3171, 10, 74, 1, false, 10);
  mTrapRegisters[kSADCGTB].init("SADCGTB", 0x3172, 32, 75, 0, false, 32);
  mTrapRegisters[kSEBDEN].init("SEBDEN", 0x3178, 5, 76, 0, false, 5);
  mTrapRegisters[kSEBDOU].init("SEBDOU", 0x3179, 5, 76, 1, false, 5);
  mTrapRegisters[kSML0].init("SML0", 0x0A00, 15, 77, 0, false, 15);
  mTrapRegisters[kSML1].init("SML1", 0x0A01, 15, 77, 1, false, 15);
  mTrapRegisters[kSML2].init("SML2", 0x0A02, 15, 77, 2, false, 15);
  mTrapRegisters[kSMMODE].init("SMMODE", 0x0A03, 16, 79, 0, false, 16);
  mTrapRegisters[kNITM0].init("NITM0", 0x0A08, 15, 80, 0, false, 12);
  mTrapRegisters[kNITM1].init("NITM1", 0x0A09, 15, 80, 1, false, 12);
  mTrapRegisters[kNITM2].init("NITM2", 0x0A0A, 15, 80, 2, false, 12);
  mTrapRegisters[kNIP4D].init("NIP4D", 0x0A0B, 15, 80, 3, false, 12);
  mTrapRegisters[kARBTIM].init("ARBTIM", 0x0A3F, 16, 82, 0, false, 12);
  mTrapRegisters[kIA0IRQ0].init("IA0IRQ0", 0x0B00, 15, 83, 0, true, 12);
  mTrapRegisters[kIA0IRQ1].init("IA0IRQ1", 0x0B01, 15, 83, 1, true, 12);
  mTrapRegisters[kIA0IRQ2].init("IA0IRQ2", 0x0B02, 15, 83, 2, true, 12);
  mTrapRegisters[kIA0IRQ3].init("IA0IRQ3", 0x0B03, 15, 83, 3, true, 12);
  mTrapRegisters[kIA0IRQ4].init("IA0IRQ4", 0x0B04, 15, 83, 4, true, 12);
  mTrapRegisters[kIA0IRQ5].init("IA0IRQ5", 0x0B05, 15, 83, 5, true, 12);
  mTrapRegisters[kIA0IRQ6].init("IA0IRQ6", 0x0B06, 15, 83, 6, true, 12);
  mTrapRegisters[kIA0IRQ7].init("IA0IRQ7", 0x0B07, 15, 83, 7, true, 12);
  mTrapRegisters[kIA0IRQ8].init("IA0IRQ8", 0x0B08, 15, 83, 8, true, 12);
  mTrapRegisters[kIA0IRQ9].init("IA0IRQ9", 0x0B09, 15, 83, 9, true, 12);
  mTrapRegisters[kIA0IRQA].init("IA0IRQA", 0x0B0A, 15, 83, 10, true, 12);
  mTrapRegisters[kIA0IRQB].init("IA0IRQB", 0x0B0B, 15, 83, 11, true, 12);
  mTrapRegisters[kIA0IRQC].init("IA0IRQC", 0x0B0C, 15, 83, 12, true, 12);
  mTrapRegisters[kIRQSW0].init("IRQSW0", 0x0B0D, 15, 83, 13, true, 13);
  mTrapRegisters[kIRQHW0].init("IRQHW0", 0x0B0E, 15, 83, 14, true, 13);
  mTrapRegisters[kIRQHL0].init("IRQHL0", 0x0B0F, 15, 83, 15, true, 13);
  mTrapRegisters[kIA1IRQ0].init("IA1IRQ0", 0x0B20, 15, 91, 0, true, 12);
  mTrapRegisters[kIA1IRQ1].init("IA1IRQ1", 0x0B21, 15, 91, 1, true, 12);
  mTrapRegisters[kIA1IRQ2].init("IA1IRQ2", 0x0B22, 15, 91, 2, true, 12);
  mTrapRegisters[kIA1IRQ3].init("IA1IRQ3", 0x0B23, 15, 91, 3, true, 12);
  mTrapRegisters[kIA1IRQ4].init("IA1IRQ4", 0x0B24, 15, 91, 4, true, 12);
  mTrapRegisters[kIA1IRQ5].init("IA1IRQ5", 0x0B25, 15, 91, 5, true, 12);
  mTrapRegisters[kIA1IRQ6].init("IA1IRQ6", 0x0B26, 15, 91, 6, true, 12);
  mTrapRegisters[kIA1IRQ7].init("IA1IRQ7", 0x0B27, 15, 91, 7, true, 12);
  mTrapRegisters[kIA1IRQ8].init("IA1IRQ8", 0x0B28, 15, 91, 8, true, 12);
  mTrapRegisters[kIA1IRQ9].init("IA1IRQ9", 0x0B29, 15, 91, 9, true, 12);
  mTrapRegisters[kIA1IRQA].init("IA1IRQA", 0x0B2A, 15, 91, 10, true, 12);
  mTrapRegisters[kIA1IRQB].init("IA1IRQB", 0x0B2B, 15, 91, 11, true, 12);
  mTrapRegisters[kIA1IRQC].init("IA1IRQC", 0x0B2C, 15, 91, 12, true, 12);
  mTrapRegisters[kIRQSW1].init("IRQSW1", 0x0B2D, 15, 91, 13, true, 13);
  mTrapRegisters[kIRQHW1].init("IRQHW1", 0x0B2E, 15, 91, 14, true, 13);
  mTrapRegisters[kIRQHL1].init("IRQHL1", 0x0B2F, 15, 91, 15, true, 13);
  mTrapRegisters[kIA2IRQ0].init("IA2IRQ0", 0x0B40, 15, 99, 0, true, 12);
  mTrapRegisters[kIA2IRQ1].init("IA2IRQ1", 0x0B41, 15, 99, 1, true, 12);
  mTrapRegisters[kIA2IRQ2].init("IA2IRQ2", 0x0B42, 15, 99, 2, true, 12);
  mTrapRegisters[kIA2IRQ3].init("IA2IRQ3", 0x0B43, 15, 99, 3, true, 12);
  mTrapRegisters[kIA2IRQ4].init("IA2IRQ4", 0x0B44, 15, 99, 4, true, 12);
  mTrapRegisters[kIA2IRQ5].init("IA2IRQ5", 0x0B45, 15, 99, 5, true, 12);
  mTrapRegisters[kIA2IRQ6].init("IA2IRQ6", 0x0B46, 15, 99, 6, true, 12);
  mTrapRegisters[kIA2IRQ7].init("IA2IRQ7", 0x0B47, 15, 99, 7, true, 12);
  mTrapRegisters[kIA2IRQ8].init("IA2IRQ8", 0x0B48, 15, 99, 8, true, 12);
  mTrapRegisters[kIA2IRQ9].init("IA2IRQ9", 0x0B49, 15, 99, 9, true, 12);
  mTrapRegisters[kIA2IRQA].init("IA2IRQA", 0x0B4A, 15, 99, 10, true, 12);
  mTrapRegisters[kIA2IRQB].init("IA2IRQB", 0x0B4B, 15, 99, 11, true, 12);
  mTrapRegisters[kIA2IRQC].init("IA2IRQC", 0x0B4C, 15, 99, 12, true, 12);
  mTrapRegisters[kIRQSW2].init("IRQSW2", 0x0B4D, 15, 99, 13, true, 13);
  mTrapRegisters[kIRQHW2].init("IRQHW2", 0x0B4E, 15, 99, 14, true, 13);
  mTrapRegisters[kIRQHL2].init("IRQHL2", 0x0B4F, 15, 99, 15, true, 13);
  mTrapRegisters[kIA3IRQ0].init("IA3IRQ0", 0x0B60, 15, 107, 0, true, 12);
  mTrapRegisters[kIA3IRQ1].init("IA3IRQ1", 0x0B61, 15, 107, 1, true, 12);
  mTrapRegisters[kIA3IRQ2].init("IA3IRQ2", 0x0B62, 15, 107, 2, true, 12);
  mTrapRegisters[kIA3IRQ3].init("IA3IRQ3", 0x0B63, 15, 107, 3, true, 12);
  mTrapRegisters[kIA3IRQ4].init("IA3IRQ4", 0x0B64, 15, 107, 4, true, 12);
  mTrapRegisters[kIA3IRQ5].init("IA3IRQ5", 0x0B65, 15, 107, 5, true, 12);
  mTrapRegisters[kIA3IRQ6].init("IA3IRQ6", 0x0B66, 15, 107, 6, true, 12);
  mTrapRegisters[kIA3IRQ7].init("IA3IRQ7", 0x0B67, 15, 107, 7, true, 12);
  mTrapRegisters[kIA3IRQ8].init("IA3IRQ8", 0x0B68, 15, 107, 8, true, 12);
  mTrapRegisters[kIA3IRQ9].init("IA3IRQ9", 0x0B69, 15, 107, 9, true, 12);
  mTrapRegisters[kIA3IRQA].init("IA3IRQA", 0x0B6A, 15, 107, 10, true, 12);
  mTrapRegisters[kIA3IRQB].init("IA3IRQB", 0x0B6B, 15, 107, 11, true, 12);
  mTrapRegisters[kIA3IRQC].init("IA3IRQC", 0x0B6C, 15, 107, 12, true, 12);
  mTrapRegisters[kIRQSW3].init("IRQSW3", 0x0B6D, 15, 107, 13, true, 13);
  mTrapRegisters[kIRQHW3].init("IRQHW3", 0x0B6E, 15, 107, 14, true, 13);
  mTrapRegisters[kIRQHL3].init("IRQHL3", 0x0B6F, 15, 107, 15, true, 13);
  mTrapRegisters[kCTGDINI].init("CTGDINI", 0x0B80, 32, 115, 0, false, 32);
  mTrapRegisters[kCTGCTRL].init("CTGCTRL", 0x0B81, 16, 116, 0, false, 16);
  mTrapRegisters[kMEMRW].init("MEMRW", 0xD000, 10, 117, 0, false, 10);
  mTrapRegisters[kMEMCOR].init("MEMCOR", 0xD001, 10, 117, 1, false, 10);
  mTrapRegisters[kDMDELA].init("DMDELA", 0xD002, 10, 117, 2, false, 10);
  mTrapRegisters[kDMDELS].init("DMDELS", 0xD003, 10, 117, 3, false, 10);
  mTrapRegisters[kNMOD].init("NMOD", 0x0D40, 31, 119, 0, false, 31);
  mTrapRegisters[kNDLY].init("NDLY", 0x0D41, 31, 119, 1, false, 31);
  mTrapRegisters[kNED].init("NED", 0x0D42, 31, 119, 2, false, 31);
  mTrapRegisters[kNTRO].init("NTRO", 0x0D43, 31, 119, 3, false, 31);
  mTrapRegisters[kNRRO].init("NRRO", 0x0D44, 31, 119, 4, false, 31);
  mTrapRegisters[kNBND].init("NBND", 0x0D47, 16, 124, 0, false, 16);
  mTrapRegisters[kNP0].init("NP0", 0x0D48, 15, 125, 0, false, 15);
  mTrapRegisters[kNP1].init("NP1", 0x0D49, 15, 125, 1, false, 15);
  mTrapRegisters[kNP2].init("NP2", 0x0D4A, 15, 125, 2, false, 15);
  mTrapRegisters[kNP3].init("NP3", 0x0D4B, 15, 125, 3, false, 15);
  mTrapRegisters[kC08CPU0].init("C08CPU0", 0x0C00, 32, 126, 0, true, 32);
  mTrapRegisters[kQ2VINFO].init("Q2VINFO", 0x0C01, 32, 127, 0, false, 32); // Q2 start, end and tracklet format
  mTrapRegisters[kC10CPU0].init("C10CPU0", 0x0C02, 32, 128, 0, true, 32);
  mTrapRegisters[kC11CPU0].init("C11CPU0", 0x0C03, 32, 129, 0, true, 32);
  mTrapRegisters[kC12CPUA].init("C12CPUA", 0x0C04, 32, 130, 0, true, 32);
  mTrapRegisters[kC13CPUA].init("C13CPUA", 0x0C05, 32, 131, 0, true, 32);
  mTrapRegisters[kC14CPUA].init("C14CPUA", 0x0C06, 32, 132, 0, true, 32);
  mTrapRegisters[kC15CPUA].init("C15CPUA", 0x0C07, 32, 133, 0, true, 32);
  mTrapRegisters[kC08CPU1].init("C08CPU1", 0x0C08, 32, 134, 0, true, 32);
  mTrapRegisters[kVINFO].init("VINFO", 0x0C09, 32, 135, 0, false, 32); // source version 24bit version + # of commits
  mTrapRegisters[kC10CPU1].init("C10CPU1", 0x0C0A, 32, 136, 0, true, 32);
  mTrapRegisters[kC11CPU1].init("C11CPU1", 0x0C0B, 32, 137, 0, true, 32);
  mTrapRegisters[kC08CPU2].init("C08CPU2", 0x0C10, 32, 138, 0, true, 32);
  mTrapRegisters[kNDRIFT].init("NDRIFT", 0x0C11, 32, 139, 0, false, 32); // was called C09CPU2
  mTrapRegisters[kC10CPU2].init("C10CPU2", 0x0C12, 32, 140, 0, true, 32);
  mTrapRegisters[kC11CPU2].init("C11CPU2", 0x0C13, 32, 141, 0, true, 32);
  mTrapRegisters[kC08CPU3].init("C08CPU3", 0x0C18, 32, 142, 0, true, 32);
  mTrapRegisters[kYCORR].init("YCORR", 0x0C19, 32, 143, 0, false, 32); // was called C09CPU3
  mTrapRegisters[kC10CPU3].init("C10CPU3", 0x0C1A, 32, 144, 0, true, 32);
  mTrapRegisters[kC11CPU3].init("C11CPU3", 0x0C1B, 32, 145, 0, true, 32);
  mTrapRegisters[kNES].init("NES", 0x0D45, 31, 146, 0, false, 31);
  mTrapRegisters[kNTP].init("NTP", 0x0D46, 31, 147, 0, false, 31);
  mTrapRegisters[kNCUT].init("NCUT", 0x0D4C, 32, 148, 0, false, 32);
  mTrapRegisters[kPASACHM].init("PASACHM", 0x315C, 32, 149, 0, false, 19);
}

TrapRegInfo& TrapRegisters::operator[](uint32_t regid)
{
  return mTrapRegisters[regid];
}

int32_t TrapRegisters::getRegIndexByName(const std::string& name)
{
  // there is no index for this but its not used online
  int counter = 0;
  for (auto& reg : mTrapRegisters) {
    if (reg.getName() == name) {
      return counter;
    }
    counter++;
  }
  return -1; // error condition
}

int32_t TrapRegisters::getRegAddrByName(const std::string& name)
{
  // there is no index for this but its not used online
  for (auto& reg : mTrapRegisters) {
    if (reg.getName() == name) {
      return reg.getAddr();
    }
  }
  return -1; // error condition
}
