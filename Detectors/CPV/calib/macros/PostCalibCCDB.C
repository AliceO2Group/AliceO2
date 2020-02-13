// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TH2F.h"
#include "CCDB/CcdbApi.h"
#include "CPVCalib/CalibParams.h"
#include "CPVBase/Geometry.h"
#include <iostream>
#endif
void PostCalibCCDB()
{

  //Post test calibration parameters for CPV to test CCDB
  //Input are files which can be produced with macros PlotOCDB.C

  o2::ccdb::CcdbApi ccdb;
  std::map<std::string, std::string> metadata; // do we want to store any meta data?
  ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation

  auto o2cpvCalib = new o2::cpv::CalibParams();

  TFile* fHGLGratio = new TFile("Run2_HGLH.root");
  // TFile * fTimeCalib = new TFile("Run2_TimeCalib.root") ;
  TFile* fGains = new TFile("Run2_CPVCalib.root");
  TH2F* hGains[5];
  for (Int_t mod = 2; mod < 5; mod++) {

    hGains[mod] = (TH2F*)fGains->Get(Form("mod%d", mod));
    if (!o2cpvCalib->setGain(hGains[mod], mod)) {
      cout << " Can not set gain for module " << mod << endl;
      return;
    }
  }

  ccdb.storeAsTFileAny(o2cpvCalib, "CPV/Calib", metadata, 1, 1670700184549); // one year validity time
}
