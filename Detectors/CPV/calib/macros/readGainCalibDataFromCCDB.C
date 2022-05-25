#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CPVCalibration/GainCalibrator.h"
#include "CPVBase/Geometry.h"
#include <iostream>
#endif

o2::cpv::GainCalibData* readGainCalibDataFromCCDB(long timeStamp = 0, const char* ccdbURI = "http://ccdb-test.cern.ch:8080")
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
  o2::cpv::GainCalibData* gcd = ccdbMgr.get<o2::cpv::GainCalibData>("CPV/PhysicsRun/GainCalibData");
  if (!gcd) {
    std::cerr << "Cannot get GainCalibData from CPV/PhysicsRun/GainCalibData!" << std::endl;
    return 0x0;
  }
  gcd->print();

  cout << "I passed print" << endl;
  TH2F *hGCDEntries[3], *hGCDMean[3];
  o2::cpv::Geometry geo;
  short relId[3];
  TCanvas* can = new TCanvas("Modules", "Modules");
  can->Divide(3, 2);
  for (int iMod = 0; iMod < 3; iMod++) {
    hGCDEntries[iMod] = new TH2F(Form("hGCDEntriesM%d", iMod + 2),
                                 Form("GainCalibData entries in M%d", iMod + 2),
                                 128, 0., 128., 60, 0., 60);
    hGCDMean[iMod] = new TH2F(Form("hGCDMeanM%d", iMod + 2),
                              Form("GainCalibData means in M%d", iMod + 2),
                              128, 0., 128., 60, 0., 60);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      geo.absToRelNumbering(iCh, relId);
      hGCDEntries[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, gcd->mAmplitudeSpectra[iCh].getNEntries());
      if (gcd->mAmplitudeSpectra[iCh].getNEntries() > 0) {
        hGCDMean[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, gcd->mAmplitudeSpectra[iCh].getMean());
      }
    }
    can->cd(iMod + 1);
    hGCDEntries[iMod]->GetXaxis()->SetTitle("X pad");
    hGCDEntries[iMod]->GetYaxis()->SetTitle("Z pad");
    hGCDEntries[iMod]->Draw("colz");
    can->cd(iMod + 4);
    hGCDMean[iMod]->GetXaxis()->SetTitle("X pad");
    hGCDMean[iMod]->GetYaxis()->SetTitle("Z pad");
    hGCDMean[iMod]->Draw("colz");
  }
  return gcd;
}
