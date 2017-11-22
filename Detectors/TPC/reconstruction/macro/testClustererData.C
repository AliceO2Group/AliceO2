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
#include <vector>
#include <memory>

#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TString.h"

#include "TPCReconstruction/HwClusterer.h"
#include "TPCReconstruction/ClusterContainer.h"
#include "TPCBase/Digit.h"
#include "TPCReconstruction/GBTFrameContainer.h"
#endif

using namespace o2::TPC;

void setupContainers(TString fileInfo);

std::vector<std::unique_ptr<GBTFrameContainer>> mGBTFrameContainers; //! raw reader pointer

void testClustererData(Int_t maxEvents=50, TString fileInfo="GBTx0_Run005:0:0;GBTx1_Run005:1:0", TString pedestalFile="/data/Work/software/alicesw/O2-work/testData/FE55/testPedestal.root", TString outputFileName="clusters.root")
{

  TFile f(pedestalFile);
  CalDet<float> *pedestal = nullptr;
  f.GetObject("Pedestals", pedestal);
  printf("pedestal: %.2f\n", pedestal->getValue(CRU(0), 0, 0));

  setupContainers(fileInfo);

  int mTimeBinsPerCall=500;

  // ===| output file and container |===========================================
  std::vector<o2::TPC::Cluster> arrCluster;
  TFile fout(outputFileName,"recreate");
  TTree t("clusters","clusters");
  t.Branch("cl", &arrCluster);

  // ===| cluster finder |======================================================
  HwClusterer cl(&arrCluster, nullptr);
  cl.setPedestalObject(pedestal);

  // ===| loop over all data |==================================================
  int events = 0;
  bool data = true;
  while (data && (events<maxEvents)) {
    std::vector<Digit> digits(80);

    for (auto& reader_ptr : mGBTFrameContainers) {
      auto reader = reader_ptr.get();
      for (int i=0; i<mTimeBinsPerCall; ++i) {
        data = reader->getData(digits);
        if (!data) break;
      }
      if (!data) break;
    }

    printf("Event %d, found digits: %zu\n", events, digits.size());

    // Review if this copy from digits to arr is still needed? (as it used to be when it was still a TClonesArray)
    float maxTime = 0;
    std::vector<Digit> arr; 
    for (auto& digi : digits) {
      if (digi.getRow() == 255 && digi.getPad() == 255) continue;
      arr.emplace_back(digi.getCRU(), digi.getChargeFloat(), digi.getRow(), digi.getPad(), digi.getTimeStamp());
      if (digi.getTimeStamp() > maxTime) maxTime = digi.getTimeStamp();
    }

    printf("Converted digits: %lu, max time: %.2f\n", arr.size(), maxTime);
    //for (Int_t i=0; i<10; ++i) {
      //printf("%.2f ", ((DigitMC*)arr.At(i))->getChargeFloat());
    //}
    //printf("\n");

    cl.Process(arr,nullptr,events);
    t.Fill();

    printf("Found clusters: %lu\n", arrCluster.size());
    arrCluster.clear();
    ++events;
  }


  fout.Write();
  fout.Close();

}

/// add GBT frame container to process
void addGBTFrameContainer(GBTFrameContainer *cont) { mGBTFrameContainers.push_back(std::unique_ptr<GBTFrameContainer>(cont)); }

void setupContainers(TString fileInfo)
{
  int iSize = 4000000;
  int iCRU = 0;
  int iLink = 0;

  //auto contPtr = std::unique_ptr<GBTFrameContainer>(new GBTFrameContainer(iSize,iCRU,iLink));
  // input data
  auto arrData = fileInfo.Tokenize("; ");
  for (auto o : *arrData) {
    const TString& data = static_cast<TObjString*>(o)->String();

    // get file info: file name, cru, link
    auto arrDataInfo = data.Tokenize(":");
    if (arrDataInfo->GetEntriesFast() != 3) {
      printf("Error, badly formatte input data string: %s, expected format is <filename:cru:link>\n", data.Data());
      delete arrDataInfo;
      continue;
    }

    TString& filename = static_cast<TObjString*>(arrDataInfo->At(0))->String();
    iCRU = static_cast<TObjString*>(arrDataInfo->At(1))->String().Atoi();
    iLink = static_cast<TObjString*>(arrDataInfo->At(2))->String().Atoi();

    auto cont = new GBTFrameContainer(iSize,iCRU,iLink);

    cont->setEnableAdcClockWarning(false);
    cont->setEnableSyncPatternWarning(false);
    cont->setEnableStoreGBTFrames(false);
    cont->setEnableCompileAdcValues(true);

    std::cout << "Read digits from file " << filename << " with cru " << iCRU << ", link " << iLink << "...\n";
    cont->addGBTFramesFromBinaryFile(filename.Data());
    std::cout << " ... done. Read " << cont->getSize() << "\n";

    addGBTFrameContainer(cont);
  }

  delete arrData;
}
