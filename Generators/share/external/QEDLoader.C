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

//< Loader macro to run QED background generator from QEDepem.C macro, use it as e.g.
//< o2-sim -n10000 -m PIPE ITS T0 MFT  --noemptyevents -g external --configKeyValues "GeneratorExternal.fileName=QEDloader.C"
#include <fairlogger/Logger.h>

FairGenerator* fg = nullptr;

FairGenerator* QEDLoader()
{
  const TString macroName = "QEDepem";
  gSystem->Load("libTEPEMGEN");

  // the path of the macro to load depends on where it was installed, we assume that its installation
  // directory is the same as of the loader macro
  std::ostringstream mstr;
  mstr << __FILE__;
  TString macroFullName = Form("%s/%s.C", gSystem->DirName(mstr.str().c_str()), macroName.Data());
  LOG(info) << "\nLoading " << macroFullName.Data() << "\n";

  gROOT->LoadMacro(macroFullName.Data());
  gInterpreter->ProcessLine(Form("%s()", macroName.Data()));
  return fg;
}
