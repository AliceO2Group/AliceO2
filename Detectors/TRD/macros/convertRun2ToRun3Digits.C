#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TClonesArray.h>
// #include <TEveLine.h>
// #include <TEveManager.h>
// #include <TEveElement.h>

#include <AliRunLoader.h>
#include <AliLoader.h>
#include <AliDataLoader.h>
#include <AliTreeLoader.h>
#include <AliTRDarrayADC.h>
// #include <AliTRDtrackletWord.h>
// #include <AliTRDtrackletMCM.h>

#include <iostream>

#include "TH1F.h"
// #include "AliTRDtrackletMCM.h"
//#include "TRDDataFormat/TriggerRecord.h"
#include "TRDBase/Digit.h"
// #include "TRDBase/Tracklet.h"
#endif

using namespace o2;
using namespace trd;
using namespace std;


void convertRun2ToRun3Digits(TString infilename = "TRD.Digits.root",
              TString outfilename = "digitsqa.root")
{
    std::vector<o2::trd::Digit> run3Digits;
    
    // ======================================================================
    // Book histograms

    TFile *outFile = new TFile(outfilename, "RECREATE");

    TH1F *hAdc = new TH1F("hADC", "ADC spectrum", 1024, -0.5, 1023.5);
    hAdc->SetXTitle("ADC value");
    hAdc->SetYTitle("number of entries");

    TH1F *hTBsum = new TH1F("hTBsum", "TBsum", 3000, -0.5, 2999.5);

    TFile fd(infilename);

    AliTRDdigitsManager *digitMan = new AliTRDdigitsManager;
    digitMan->CreateArrays();

    TIter next(fd.GetListOfKeys());
    while (TObject *obj = next())
    {
        cout << "Processing " << obj->GetName() << endl;
        string eventNumber(obj->GetName(), 5, 3);
        // eventTime needs to be some increasing integer
        int eventTime = stoi(eventNumber) * 12;

        TTree *tr = (TTree *)fd.Get(Form("%s/TreeD", obj->GetName()));
        //tr->Print();

        for (int det = 0; det < 540; det++)
        {
            digitMan->ClearArrays(det);
            digitMan->ClearIndexes(det);
        }

        digitMan->ReadDigits(tr);

        for (int det = 0; det < 540; det++)
        {
            if (!digitMan->GetDigits(det))
                continue;

            digitMan->GetDigits(det)->Expand();

            AliTRDarrayADC* digits;
            digits = digitMan->GetDigits(det);
            digits->Expand();

            // TODO: check actual number of rows, from geometry
            // here: play it safe, assume 12 rows everywhere
            int nrows = 12;
            for (int row = 0; row < nrows; row++)
            {
                for (int col = 0; col < 144; col++)
                {
                    int tbsum = 0;
                    ArrayADC adctimes;
                    for (int timebin = 0; timebin < digitMan->GetDigits(det)->GetNtime(); timebin++)
                    {
                        // int adc = digitMan->GetDigits(det)->GetDataBits(row,col,timebin);
                        int adc = digitMan->GetDigitAmp(row, col, timebin, det);

                        if (digits->GetNtime() > 30)
                        {
                            cout << "----!!! --- number of times is greater than 30" << endl;
                        }
                        adctimes[timebin] = digits->GetData(row, col, timebin);

                        // this value seems to indicate no digit -> skip
                        if (adc == -7169)
                            continue;

                        hAdc->Fill(adc);
                        tbsum += adc;
                    }

                    run3Digits.push_back(o2::trd::Digit(det, row, col, adctimes, eventTime));

                    if (tbsum > 0)
                    {
                        hTBsum->Fill(tbsum);
                    }
                }
            }
        }
    }

    TCanvas *cnv_adc = new TCanvas("cnv_adc", "cnv_adc");
    cnv_adc->SetLogy();
    hAdc->Draw();

    TCanvas *cnv_tbsum = new TCanvas("cnv_tbsum", "cnv_tbsum");
    cnv_adc->SetLogy();
    hTBsum->Draw();

    outFile->Write();
    
    TFile *digitsFile = new TFile("convertedRun3Digits.root", "RECREATE");
    TTree *digitTree = new TTree("o2sim", "run2 digits");
    std::vector<o2::trd::Digit> *run3pdigits = &run3Digits;

    digitTree->Branch("TRDDigit", &run3pdigits);
    digitTree->Fill();
    cout << "Number of run3 digits is: " << run3Digits.size() << endl;
    digitTree->Write();
    delete digitTree;
    delete digitsFile;
}