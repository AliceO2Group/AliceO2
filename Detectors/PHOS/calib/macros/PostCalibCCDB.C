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
#include "PHOSCalib/CalibParams.h"
#include "PHOSBase/Geometry.h"
#include <iostream>
#endif
void PostCalibCCDB()
{

  //Post test calibration parameters for PHOS to test CCDB
  //Input are files which can be produced with macros PlotOCDB.C

  o2::ccdb::CcdbApi ccdb;
  std::map<std::string, std::string> metadata; // do we want to store any meta data?
  ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation

  auto o2phosCalib = new o2::phos::CalibParams();

  o2::phos::Geometry* geom = o2::phos::Geometry::GetInstance("Run3"); // Needed for tranforming 2D histograms to channel ID

  TFile* fHGLGratio = new TFile("Run2_HGLH.root");
  // TFile * fTimeCalib = new TFile("Run2_TimeCalib.root") ;
  TFile* fGains = new TFile("Run2_PHOSCalib.root");
  TH2F *hHGLG[5], *hTimeHG[5], *hTimeLG[5], *hGains[5];
  for (Int_t mod = 1; mod < 5; mod++) {
    hHGLG[mod] = (TH2F*)fHGLGratio->Get(Form("LGHGm%d", mod));

    if (!o2phosCalib->setHGLGRatio(hHGLG[mod], mod)) {
      std::cout << " Can not set LG/HG ratio for module " << mod << std::endl;
      return;
    }

    hTimeHG[mod] = (TH2F*)fGains->Get(Form("Tmod%d", mod));
    hTimeLG[mod] = (TH2F*)fGains->Get(Form("Tlmod%d", mod));
    if (!o2phosCalib->setHGTimeCalib(hTimeHG[mod], mod)) {
      std::cout << " Can not set HG time for module " << mod << std::endl;
      return;
    }
    if (!o2phosCalib->setLGTimeCalib(hTimeLG[mod], mod)) {
      std::cout << " Can not set LG time for module " << mod << std::endl;
      return;
    }

    hGains[mod] = (TH2F*)fGains->Get(Form("mod%d", mod));
    if (!o2phosCalib->setGain(hGains[mod], mod)) {
      std::cout << " Can not set gain for module " << mod << std::endl;
      return;
    }
  }

  ccdb.storeAsTFileAny(o2phosCalib, "PHOS/Calib", metadata, 1, 1670700184549); // one year validity time
}