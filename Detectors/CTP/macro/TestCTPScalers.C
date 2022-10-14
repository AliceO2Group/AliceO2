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

/// \file TestCTPScalers.C
/// \brief create CTP scalers, test it and add to database
/// \author Roman Lietava
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Scalers.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
void TestCTPScalers(long tmin = 0, long tmax = -1, std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  /// Demo scalers
  std::string scalersstr = "0\n";      // version
  scalersstr += "333 5 0 2 8 32 63\n"; // runnumber nclasses [classes list]]
  scalersstr += "4 3563 4 5  \n";      // orbit bc secs usecs
  scalersstr += "0 0 0 0 0 0 \n";      // class 0
  scalersstr += "0 0 0 0 0 0 \n";      // class 2
  scalersstr += "0 0 0 0 0 0 \n";      // class 8
  scalersstr += "0 0 0 0 0 0 \n";      // class 32
  scalersstr += "0 0 0 0 0 0 \n";      // class 63
  scalersstr += "1000 3563 5 5\n";     // orbit bc secs usecs
  scalersstr += "1 1 1 1 1 1 \n";      // class 0
  scalersstr += "1 1 1 1 1 1 \n";      // class 2
  scalersstr += "1 1 1 1 1 1 \n";      // class 8
  scalersstr += "1 1 1 1 1 1 \n";      // class 32
  scalersstr += "1 1 1 1 1 1 \n";      // class 63
  //
  /// Demo scalers: LMB >= LMA >= L0B >= L0A >= L1B >= L1A error
  std::string scalersstr_e1 = "0\n";       // version
  scalersstr_e1 += "333 5 0 2 8 32 63\n";  // runnumber nclasses [classes list]]
  scalersstr_e1 += "4 3563 4 5  \n";       // orbit bc secs usecs
  scalersstr_e1 += "20 20 20 20 20 20 \n"; // class 0
  scalersstr_e1 += "0 0 0 0 0 0 \n";       // class 2
  scalersstr_e1 += "0 0 0 0 0 0 \n";       // class 8
  scalersstr_e1 += "0 0 0 0 0 0 \n";       // class 32
  scalersstr_e1 += "0 0 0 0 0 0 \n";       // class 63
  scalersstr_e1 += "1000 3563 5 5\n";      // orbit bc secs usecs
  scalersstr_e1 += "1 2 3 4 5 6 \n";       // class 0
  scalersstr_e1 += "1 1 1 1 1 1 \n";       // class 2
  scalersstr_e1 += "1 1 1 1 1 1 \n";       // class 8
  scalersstr_e1 += "1 1 1 1 1 1 \n";       // class 32
  scalersstr_e1 += "1 1 1 1 1 1 \n";       // class 63
  //
  CTPRunScalers ctpscalers;
  int ret = ctpscalers.readScalers(scalersstr_e1);
  if (ret != 0) {
    return;
  }
  //
  ctpscalers.convertRawToO2();
  ctpscalers.printStream(std::cout);
}
