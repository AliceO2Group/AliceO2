#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsCPV/CalibParams.h"
#include "CPVBase/Geometry.h"
#include <iostream>
#endif

o2::cpv::CalibParams* readCalibParamsFromCCDB(long timeStamp = 0, const char* ccdbURI = "http://ccdb-test.cern.ch:8080")
{
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL(ccdbURI);
  if (!ccdbMgr.isHostReachable()) {
    std::cerr << ccdbURI << " is not reachable!" << std::endl;
    return 0x0;
  }
  if (timeStamp == 0) {
    timeStamp = o2::ccdb::getCurrentTimestamp();
  }
  ccdbMgr.setTimestamp(timeStamp);
  o2::cpv::CalibParams* gains = ccdbMgr.get<o2::cpv::CalibParams>("CPV/Calib/Gains");
  if (!gains) {
    std::cerr << "Cannot get gains from CCDB/CPV/Calib/Gains!" << std::endl;
    return 0x0;
  }

  TH2F* hGains[3];
  o2::cpv::Geometry geo;
  short relId[3];
  TCanvas* can = new TCanvas("Modules", "Modules");
  can->Divide(3, 1);
  for (int iMod = 0; iMod < 3; iMod++) {
    hGains[iMod] = new TH2F(Form("hGainsM%d", iMod + 2),
                            Form("Gains in M%d", iMod + 2),
                            128, 0., 128., 60, 0., 60);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      geo.absToRelNumbering(iCh, relId);
      hGains[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, gains->getGain(iCh));
    }
    // TCanvas* can = new TCanvas(Form("canM%d", iMod + 2), Form("module M%d", iMod + 2), 10 * iMod, 0, 1000 + 10 * iMod, 1000);
    can->cd(iMod + 1);
    hGains[iMod]->GetXaxis()->SetTitle("X pad");
    hGains[iMod]->GetYaxis()->SetTitle("Z pad");
    hGains[iMod]->Draw("colz");
  }
  return gains;
}
