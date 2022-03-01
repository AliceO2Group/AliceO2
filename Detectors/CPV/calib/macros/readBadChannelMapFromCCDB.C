#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsCPV/BadChannelMap.h"
#include "CPVBase/Geometry.h"
#include <iostream>
#endif

o2::cpv::BadChannelMap* readBadChannelMapFromCCDB(long timeStamp = 0, const char* ccdbURI = "http://ccdb-test.cern.ch:8080")
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
  o2::cpv::BadChannelMap* badMap = ccdbMgr.get<o2::cpv::BadChannelMap>("CPV/Calib/BadChannelMap");
  if (!badMap) {
    std::cerr << "Cannot get badMap from CCDB/CPV/Calib/BadChannelMap!" << std::endl;
    return 0x0;
  }

  TH2F* hBadMap[3];
  o2::cpv::Geometry geo;
  short relId[3];
  for (int iMod = 0; iMod < 3; iMod++) {
    hBadMap[iMod] = new TH2F(Form("hBadMapM%d", iMod + 2),
                             Form("Bad Channel Map in M%d", iMod + 2),
                             128, 0., 128., 60, 0., 60);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      geo.absToRelNumbering(iCh, relId);
      hBadMap[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, !badMap->isChannelGood(iCh));
    }
    TCanvas* can = new TCanvas(Form("canM%d", iMod + 2), Form("module M%d", iMod + 2), 10 * iMod, 0, 1000 + 10 * iMod, 1000);
    can->cd(1);
    hBadMap[iMod]->Draw("colz");
  }
  return badMap;
}
