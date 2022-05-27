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

/// \file CreateCTPConfig.C
/// \brief create CTP config, test it and add to database
/// \author Roman Lietava

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "FairLogger.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
CTPConfiguration CreateCTPConfigRun2()
{
  /// Demo configuration
  CTPConfiguration ctpcfg;
  std::string cfgstr = "PARTITION: TEST \n";
  cfgstr += "VERSION:0 \n";
  cfgstr += "INPUTS: \n";
  cfgstr += "MFV0MB FV0 M 0x1 \n";
  cfgstr += "MFV0MBInner FV0 M 0x2 \n";
  cfgstr += "MFV0MBOuter FV0 M 0x4 \n";
  cfgstr += "MFV0HM FV0 M 0x8 \n";
  cfgstr += "MFT0A FT0 M 0x10 \n";
  cfgstr += "MFT0B FT0 M 0x20 \n";
  cfgstr += "MFT0Vertex FT0 M 0x40 \n";
  cfgstr += "MFT0Cent FT0 M 0x80 \n";
  cfgstr += "MFT0SemiCent FT0 M 0x100 \n";
  cfgstr += "DESCRIPTORS: \n";
  cfgstr += "DV0MB MFV0MB \n";
  cfgstr += "DV0MBInner MFV0MBInner \n";
  cfgstr += "DV0MBOuter MFV0MBOuter \n";
  cfgstr += "DT0AND MFT0A MFT0B \n";
  cfgstr += "DT0A MFT0A \n";
  cfgstr += "DT0B MFT0B \n";
  cfgstr += "DINTV0T0 MFV0MB MFT0Vertex \n";
  cfgstr += "DINT4 MFV0MB MFT0A MFT0B \n";
  cfgstr += "DV0HM MFV0HM \n";
  cfgstr += "DT0HM MFT0Cent \n";
  cfgstr += "DHM MFV0HM MFT0Cent \n";
  cfgstr += "CLUSTERS: ALL\n";
  cfgstr += "ALL FV0 FT0 TPC \n";
  cfgstr += "CLASSES:\n";
  cfgstr += "CMBV0 0 DV0MB ALL \n";
  cfgstr += "CMBT0 1 DT0AND ALL \n";
  cfgstr += "CINT4 2 DINT4 ALL \n";
  cfgstr += "CINTV0T0 3 DINTV0T0 ALL \n";
  cfgstr += "CT0A 4 DT0A ALL \n";
  cfgstr += "CT0B 62 DT0B ALL \n";
  cfgstr += "CINTHM 63 DHM ALL \n";
  //
  ctpcfg.loadConfiguration(cfgstr);
  ctpcfg.printStream(std::cout);
  return ctpcfg;
}
