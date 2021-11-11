#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsCPV/Pedestals.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVSimParams.h"
#include <iostream>
#include <fstream>
#endif

o2::cpv::Pedestals* readPedestalsFromCCDB(const char* ccdbURI = "http://ccdb-test.cern.ch:8080", long timeStamp = 0)
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
  o2::cpv::Pedestals* peds = ccdbMgr.get<o2::cpv::Pedestals>("CPV/Calib/Pedestals");
  if (!peds) {
    std::cerr << "Cannot get pedestals from CCDB/CPV/Calib/Pedestals!" << std::endl;
    return 0x0;
  }

  TH2F* hPedValues[3];
  TH2F* hPedSigmas[3];
  TH1F *hPedValues1D[3], *hPedSigmas1D[3];
  o2::cpv::Geometry geo;
  short relId[3];
  for (int iMod = 0; iMod < 3; iMod++) {
    hPedValues[iMod] = new TH2F(Form("hPedValuesM%d", iMod + 2),
                                Form("Pedestal values in M%d", iMod + 2),
                                128, 0., 128., 60, 0., 60);
    hPedSigmas[iMod] = new TH2F(Form("hPedSigmasM%d", iMod + 2),
                                Form("Pedestal sigmas in M%d", iMod + 2),
                                128, 0., 128., 60, 0., 60);
    hPedValues1D[iMod] = new TH1F(Form("hPedValues1DM%d", iMod + 2),
                                  Form("Pedestal values in M%d", iMod + 2),
                                  1000, 0., 1000.);
    hPedSigmas1D[iMod] = new TH1F(Form("hPedSigmas1DM%d", iMod + 2),
                                  Form("Pedestal sigmas in M%d", iMod + 2),
                                  1000, 0., 1000.);
    for (int iCh = iMod * 7680; iCh < (iMod + 1) * 7680; iCh++) {
      geo.absToRelNumbering(iCh, relId);
      hPedValues[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, peds->getPedestal(iCh));
      hPedSigmas[iMod]->SetBinContent(relId[1] + 1, relId[2] + 1, peds->getPedSigma(iCh));
      hPedValues1D[iMod]->Fill(peds->getPedestal(iCh));
      hPedSigmas1D[iMod]->Fill(peds->getPedSigma(iCh));
    }
    TCanvas* can = new TCanvas(Form("canM%d", iMod + 2), Form("module M%d", iMod + 2), 10 * iMod, 0, 1000 + 10 * iMod, 1000);
    can->Divide(2, 2);
    can->cd(1);
    hPedValues[iMod]->Draw("colz");
    can->cd(2);
    hPedSigmas[iMod]->Draw("colz");
    can->cd(3);
    hPedValues1D[iMod]->Draw();
    can->cd(4);
    hPedSigmas1D[iMod]->Draw();
  }

  // write pedestal thresholds to text file for electronics
  std::ofstream pededstalsTxt;
  float nSigmas = o2::cpv::CPVSimParams::Instance().mZSnSigmas;
  pededstalsTxt.open("pedestals.txt");
  for (unsigned short iCh = 0; iCh < 23040; iCh++) {
    short threshold = peds->getPedestal(iCh) + std::floor(peds->getPedSigma(iCh) * nSigmas) + 1;
    if ((threshold <= 0) || (threshold > 511)) {
      threshold = 511; // set maximum threshold for suspisious channels
    }
    short ccId, dil, gas, pad;
    geo.absIdToHWaddress(iCh, ccId, dil, gas, pad);
    unsigned short addr = ccId * 4 * 5 * 64 + dil * 5 * 64 + gas * 64 + pad;
    pededstalsTxt << addr << " " << threshold << std::endl;
  }
  pededstalsTxt.close();

  return peds;
}
