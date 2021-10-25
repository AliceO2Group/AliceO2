#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CPVBase/Geometry.h"
#endif

void readPedEfficienciesFromCCDB(const char* ccdbURI = "http://ccdb-test.cern.ch:8080", long timeStamp = 0)
{
  // ccdbURI -> CCDB instance
  // timeStamp -> time in milliseconds (wow!) (starting at 00:00 on 1.1.1970)
  // timeStamp == 0 -> current time
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL(ccdbURI);
  if (!ccdbMgr.isHostReachable()) {
    std::cerr << ccdbURI << " is not reachable!" << std::endl;
    return;
  }
  if (timeStamp == 0) {
    timeStamp = o2::ccdb::getCurrentTimestamp();
  }
  ccdbMgr.setTimestamp(timeStamp);
  std::vector<float>* effs = ccdbMgr.get<std::vector<float>>("CPV/PedestalRun/ChannelEfficiencies");
  if (!effs) {
    std::cerr << "Cannot get FEE thresholds from CCDB/CPV/PedestalRun/ChannelEfficiencies!" << std::endl;
    return;
  }

  TH2F* hEfficiencies[3];
  o2::cpv::Geometry geo;
  short relId[3];
  for (int iMod = 0; iMod < 3; iMod++) {
    hEfficiencies[iMod] = new TH2F(Form("hEfficienciesM%d", iMod + 2),
                                   Form("Channel efficiencies in M%d", iMod + 2),
                                   128, 0., 128., 60, 0., 60);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      geo.absToRelNumbering(iCh, relId);
      hEfficiencies[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, effs->at(iCh));
    }
    TCanvas* can = new TCanvas(Form("canM%d", iMod + 2), Form("module M%d", iMod + 2), 10 * iMod, 0, 1000 + 10 * iMod, 1000);
    can->Divide(1, 1);
    can->cd(1);
    hEfficiencies[iMod]->Draw("colz");
  }
}
