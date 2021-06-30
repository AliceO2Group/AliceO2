#if !defined(__CINT__) || defined(__MAKECINT__)

// ROOT
#include "TClonesArray.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"

// AliRoot
#include <AliRunLoader.h>
#include <AliLoader.h>
#include <AliDataLoader.h>
#include <AliTreeLoader.h>
#include <AliTRDarrayADC.h>
#include <AliRawReaderRoot.h>
#include <AliRawReaderDateOnline.h>
#include <AliTRDrawStream.h>
#include <AliRawReader.h>
#include <AliTRDdigitsManager.h>
#include <AliTRDCommonParam.h>
#include <AliTRDSignalIndex.h>
#include <AliTRDfeeParam.h>

// O2
#include "TRDBase/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

// other
#include <iostream>

#endif

using namespace std;
using namespace o2::trd;
using namespace o2::trd::constants;

// qa.root
// 18000283989033.808.root
// TRD.Digits.root
void convertRun2ToRun3Digits(TString qaOutPath = "",
                             TString rawDataInPath = "",
                             TString run2DigitsInPath = "",
                             TString run3DigitsOutPath = "trddigits.root",
                             int nRawEvents = 5000)
{
  vector<Digit> run3Digits;
  vector<TriggerRecord> triggerRecords;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;

  TH1F* hAdc = new TH1F("hADC", "ADC spectrum", 1024, -0.5, 1023.5);
  TH1F* hTBsum = new TH1F("hTBsum", "TBsum", 3000, -0.5, 2999.5);

  // convert raw data if path set
  if (rawDataInPath != "") {
    cout << "Converting RAW data..." << endl;
    AliRawReader* reader;
    if (rawDataInPath.Contains(".root")) {
      cout << "[I] Reading with ROOT" << endl;
      AliRawReaderRoot* readerDate = new AliRawReaderRoot(rawDataInPath);
      readerDate->SelectEquipment(0, 1024, 1024);
      readerDate->Select("TRD");
      //readerDate->SelectEvents(7);
      reader = (AliRawReader*)readerDate;

    } else if (rawDataInPath.Contains(":")) {
      cout << "[I] Reading DATE monitoring events" << endl;
      AliRawReaderDateOnline* readerRoot = new AliRawReaderDateOnline(rawDataInPath);
      readerRoot->SelectEquipment(0, 1024, 1041);
      readerRoot->Select("TRD");
      //readerRoot->SelectEvents(7);
      reader = (AliRawReader*)readerRoot;
    }

    AliTRDdigitsManager* digitMan = new AliTRDdigitsManager;
    digitMan->CreateArrays();

    AliTRDrawStream* rawStream = new AliTRDrawStream(reader);

    TClonesArray trkl("AliTRDtrackletMCM");
    rawStream->SetTrackletArray(&trkl);

    int ievent = 0;
    uint64_t triggerRecordsStart = 0;
    int recordSize = 0;
    while (reader->NextEvent()) {
      int eventtime = ievent * 12;

      if (ievent >= nRawEvents)
        break;

      //digitMan->ResetArrays();

      if (ievent % 10 == 0) {
        cout << "Event " << ievent << endl;
      }

      // hntrkl->Fill(trkl.GetEntries());
      while (rawStream->NextChamber(digitMan) >= 0) {
        //hptphase->Fill(digMan->GetDigitsParam()->GetPretriggerPhase());
      }

      for (int det = 0; det < AliTRDCommonParam::kNdet; det++) {
        AliTRDSignalIndex* idx = digitMan->GetIndexes(det);

        if (!idx)
          continue;
        if (!idx->HasEntry())
          continue;

        int row, col;
        while (idx->NextRCIndex(row, col)) {
          int tbsum = 0;
          ArrayADC adctimes;
          for (int timebin = 0; timebin < digitMan->GetDigits(det)->GetNtime(); timebin++) {
            int adc = digitMan->GetDigits(det)->GetData(row, col, timebin);
            hAdc->Fill(adc);
            tbsum += adc;

            adctimes[timebin] = adc;
          }

          if (tbsum > 0) {
            run3Digits.push_back(Digit(det, row, col, adctimes));
          }

          hTBsum->Fill(tbsum);
        }
      }
      trkl.Clear();
      recordSize = run3Digits.size() - triggerRecordsStart;
      triggerRecords.emplace_back(ievent, triggerRecordsStart, recordSize, 0, 0);
      triggerRecordsStart = run3Digits.size();
      ievent++;
    }

    delete rawStream;
    if (reader)
      delete reader;
  }

  // convert run2 digits if path set
  if (run2DigitsInPath != "") {
    cout << "Converting Run2 digits..." << endl;

    TFile run2DigitsFile(run2DigitsInPath);
    AliTRDdigitsManager* digitMan = new AliTRDdigitsManager;
    digitMan->CreateArrays();

    TIter next(run2DigitsFile.GetListOfKeys());

    uint64_t triggerRecordsStart = 0;
    int recordSize = 0;
    int ievent = 0;
    while (TObject* obj = next()) {
      cout << "Processing " << obj->GetName() << endl;

      // eventTime needs to be some increasing integer
      string eventNumber(obj->GetName(), 5, 3);
      int eventTime = stoi(eventNumber) * 12;

      TTree* tr = (TTree*)run2DigitsFile.Get(Form("%s/TreeD", obj->GetName()));

      for (int det = 0; det < AliTRDCommonParam::kNdet; det++) {
        digitMan->ClearArrays(det);
        digitMan->ClearIndexes(det);
      }

      digitMan->ReadDigits(tr);

      for (int det = 0; det < AliTRDCommonParam::kNdet; det++) {
        if (!digitMan->GetDigits(det))
          continue;

        int sector = det / 30;
        int stack = (det - sector * 30) / 6;

        digitMan->GetDigits(det)->Expand();

        int nrows = AliTRDfeeParam::GetNrowC1();
        if (stack == 2) {
          nrows = AliTRDfeeParam::GetNrowC0();
        }

        // cout << "det: " << det << " | " << "sector: " << sector << " | " << "stack: " << stack << " | " << "rows: " << nrows << endl;

        for (int row = 0; row < nrows; row++) {
          for (int col = 0; col < NCOLUMN; col++) {
            int side = (col < NCOLUMN / 2) ? 0 : 1;
            int rob = 2 * (row / NMCMROBINROW) + side;
            int mcm = side == 0 ? (row * NMCMROBINROW + col / NCOLMCM) % NMCMROB : (row * NMCMROBINROW + (col - NCOLUMN / 2) / NCOLMCM) % NMCMROB;
            int channel = 19 - (col % NCOLMCM);
            int tbsum = 0;
            ArrayADC adctimes;
            bool isSharedRight = false, isSharedLeft = false;
            if (col % NCOLMCM == 0 || col % NCOLMCM == 1) {
              isSharedRight = true;
            } else if (col % NCOLMCM == 17) {
              isSharedLeft = true;
            }

            for (int timebin = 0; timebin < digitMan->GetDigits(det)->GetNtime(); timebin++) {

              if (digitMan->GetDigits(det)->GetNtime() > 30) {
                cout << "----!!! --- number of times is greater than 30" << endl;
              }

              int adc = digitMan->GetDigitAmp(row, col, timebin, det);
              adctimes[timebin] = adc;

              // this value seems to indicate no digit -> skip
              if (adc == -7169)
                continue;

              hAdc->Fill(adc);
              tbsum += adc;
            }

            if (tbsum > 0) {
              run3Digits.push_back(Digit(det, rob, mcm, channel, adctimes));
              if (isSharedRight && col > 17) {
                if (mcm % NMCMROBINCOL == 0) {
                  // switch to the ROB on the right
                  run3Digits.push_back(Digit(det, rob - 1, mcm + 3, channel - NCOLMCM, adctimes));
                } else {
                  // we stay on the same ROB
                  run3Digits.push_back(Digit(det, rob, mcm - 1, channel - NCOLMCM, adctimes));
                }

              } else if (isSharedLeft && col < 126) {
                if (mcm % NMCMROBINCOL == 3) {
                  // switch to ROB on the left
                  run3Digits.push_back(Digit(det, rob + 1, mcm - 3, channel + NCOLMCM, adctimes));
                } else {
                  // we stay on the same ROB
                  run3Digits.push_back(Digit(det, rob, mcm + 1, channel + NCOLMCM, adctimes));
                }
              }
            }

            if (tbsum > 0) {
              hTBsum->Fill(tbsum);
            }
          }
        }
      }
      recordSize = run3Digits.size() - triggerRecordsStart;
      triggerRecords.emplace_back(ievent, triggerRecordsStart, recordSize, 0, 0);
      triggerRecordsStart = run3Digits.size();
      ievent++;
    }
  }

  // show and write QA
  if (qaOutPath != "") {
    hAdc->SetXTitle("ADC value");
    hAdc->SetYTitle("number of entries");

    TCanvas* cnv_adc = new TCanvas("cnv_adc", "cnv_adc");
    cnv_adc->SetLogy();
    hAdc->Draw();

    TCanvas* cnv_tbsum = new TCanvas("cnv_tbsum", "cnv_tbsum");
    cnv_adc->SetLogy();
    hTBsum->Draw();

    TFile* outFile = new TFile(qaOutPath, "RECREATE");
    hAdc->Write();
    hTBsum->Write();

    cout << "QA output written to: " << qaOutPath << endl;
  }

  // write run3 digits
  if (run3Digits.size() != 0) {
    TFile* digitsFile = new TFile(run3DigitsOutPath, "RECREATE");
    TTree* digitTree = new TTree("o2sim", "run2 digits");
    std::vector<Digit>* run3pdigits = &run3Digits;
    digitTree->Branch("TRDDigit", &run3pdigits);
    digitTree->Branch("TriggerRecord", &triggerRecords);
    digitTree->Branch("TRDMCLabels", &mcLabels);
    digitTree->Fill();
    cout << run3Digits.size() << " run3 digits written to: " << run3DigitsOutPath << endl;
    digitTree->Write();
    delete digitTree;
    delete digitsFile;
  }
}
