#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2I.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CPVBase/Geometry.h"
#include "DataFormatsCPV/BadChannelMap.h"
#endif

void makeBadMapFromPedestalRun(const char* ccdbURI = "http://ccdb-test.cern.ch:8080", long timeStamp = 0)
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
    std::cerr << "Cannot get Efficiencies from CCDB/CPV/PedestalRun/ChannelEfficiencies!" << std::endl;
  }

  std::vector<int>* dead = ccdbMgr.get<std::vector<int>>("CPV/PedestalRun/DeadChannels");
  if (!dead) {
    std::cerr << "Cannot get dead channels from CCDB/CPV/PedestalRun/DeadChannels!" << std::endl;
  }

  std::vector<int>* highPed = ccdbMgr.get<std::vector<int>>("CPV/PedestalRun/HighPedChannels");
  if (!highPed) {
    std::cerr << "Cannot get high ped channels from CCDB/CPV/PedestalRun/HighPedChannels!" << std::endl;
  }

  bool badMapBool[23040] = {false};
  for (int i = 0; i < 23040; i++) {
    badMapBool[i] = false;
    if (effs->at(i) > 1.1) {
      badMapBool[i] = true;
    }
  }

  for (int i = 0; i < dead->size(); i++) {
    badMapBool[dead->at(i)] = true;
  }

  for (int i = 0; i < highPed->size(); i++) {
    badMapBool[highPed->at(i)] = true;
  }

  o2::cpv::Geometry geo;
  short relId[3];
  o2::cpv::BadChannelMap badMap(1);
  TH2I* hBadMap[3];
  for (int iMod = 0; iMod < 3; iMod++) {
    hBadMap[iMod] = new TH2I(Form("hBadMapM%d", iMod + 2),
                             Form("Bad channels in M%d", iMod + 2),
                             128, 0., 128., 60, 0., 60);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      if (badMapBool[iCh]) {
        geo.absToRelNumbering(iCh, relId);
        hBadMap[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, 1);
        badMap.addBadChannel(iCh);
      }
    }
    TCanvas* can = new TCanvas(Form("canM%d", iMod + 2), Form("module M%d", iMod + 2), 10 * iMod, 0, 1000 + 10 * iMod, 1000);
    can->Divide(1, 1);
    can->cd(1);
    hBadMap[iMod]->Draw("colz");
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbURI);            // or http://localhost:8080 for a local installation
  api.storeAsTFileAny(&badMap, "CPV/Calib/BadChannelMap", metadata, timeStamp, timeStamp + 31536000000);
}
