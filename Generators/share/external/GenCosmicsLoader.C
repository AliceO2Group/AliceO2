// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//< Loader macro to run QED background generator from QEDepem.C macro, use it as e.g.
//< o2-sim -n10000 -m PIPE ITS TPC -g extgen --configKeyValues "GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C"
//< Generation options can be changed by providing --configKeyValues "cosmics.maxAngle=30.;cosmics.accept=ITS0" etc.
//< See GenCosmicsParam for available options

#include <FairLogger.h>

FairGenerator* fg = nullptr;

FairGenerator* GenCosmicsLoader()
{
  const TString macroName = "GenCosmics";
  gSystem->Load("libGeneratorCosmics.so");

  // the path of the macro to load depends on where it was installed, we assume that its installation
  // directory is the same as of the loader macro
  std::ostringstream mstr;
  mstr << __FILE__;
  TString macroFullName = Form("%s/%s.C", gSystem->DirName(mstr.str().c_str()), macroName.Data());
  LOG(INFO) << "\nLoading " << macroFullName.Data() << "\n";

  gROOT->LoadMacro(macroFullName.Data());
  gInterpreter->ProcessLine(Form("%s()", macroName.Data()));
  return fg;
}
