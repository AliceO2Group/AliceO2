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

void testGeometryRowColIndexing()
{
  auto geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  TH1* habsid = new TH1D("hAbsID", "Cell abs ID", 20001, -0.5, 20000.5);
  TH1* hsmod = new TH1D("hsmod", "Supermodule ID", 21, -0.5, 20.5);
  for (int icol = 0; icol < 96; icol++) {
    for (int irow = 0; irow < 208; irow++) {
      // exclude PHOS hole
      if (icol >= 32 && icol < 64 && irow >= 128 && irow < 200)
        continue;
      int absID = geo->GetCellAbsIDFromGlobalRowCol(irow, icol);
      habsid->Fill(absID);
      auto [smod, mod, iphi, ieta] = geo->GetCellIndexFromGlobalRowCol(irow, icol);
      hsmod->Fill(smod);
      std::cout << "Col " << icol << ", row " << irow << ": ID " << absID << ", sm " << smod << ", module " << mod << ", iphi " << iphi << ", ieta " << ieta << std::endl;
    }
  }
  auto plot = new TCanvas("geotest", "Geometry test", 1200, 700);
  plot->Divide(2, 1);
  plot->cd(1);
  habsid->Draw();
  plot->cd(2);
  hsmod->Draw();
  plot->cd();
  plot->Update();
}