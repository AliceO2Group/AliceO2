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


void convert_run2_to_run3_digits(TString infilename = "TRD.Digits.root",
              TString outfilename = "digitsqa.root")
{
    unsigned int digitscount = 0;
    std::vector<o2::trd::Digit> run3digits;
    
    // ======================================================================
    // Book histograms

    TFile *outfile = new TFile(outfilename, "RECREATE");

    TH1F *hAdc = new TH1F("hADC", "ADC spectrum", 1024, -0.5, 1023.5);
    hAdc->SetXTitle("ADC value");
    hAdc->SetYTitle("number of entries");

    TH1F *hTBsum = new TH1F("hTBsum", "TBsum", 3000, -0.5, 2999.5);

    TFile fd(infilename);

    AliTRDdigitsManager *digman = new AliTRDdigitsManager;
    digman->CreateArrays();

    TIter next(fd.GetListOfKeys());
    while (TObject *obj = next())
    {
        cout << "Processing " << obj->GetName() << endl;
        string name(obj->GetName(), 5, 3);
        int eventtime = stoi(name) * 12;

        TTree *tr = (TTree *)fd.Get(Form("%s/TreeD", obj->GetName()));
        //tr->Print();

        for (int det = 0; det < 540; det++)
        {
            digman->ClearArrays(det);
            digman->ClearIndexes(det);
        }

        digman->ReadDigits(tr);

        for (int det = 0; det < 540; det++)
        {
            if (!digman->GetDigits(det))
                continue;

            digman->GetDigits(det)->Expand();

            AliTRDarrayADC* digits;
            digits = digman->GetDigits(det);
            digits->Expand();

            // TODO: check actual number of rows, from geometry
            // here: play it safe, assume 12 rows everywhere
            int nrows = 12;
            for (int r = 0; r < nrows; r++)
            {
                for (int c = 0; c < 144; c++)
                {
                    int tbsum = 0;
                    ArrayADC adctimes;
                    for (int t = 0; t < digman->GetDigits(det)->GetNtime(); t++)
                    {
                        // int adc = digman->GetDigits(det)->GetDataBits(row,c,t);
                        int adc = digman->GetDigitAmp(r, c, t, det);

                        if (digits->GetNtime() > 30)
                        {
                            cout << "----!!! --- number of times is greater than 30" << endl;
                        }
                        adctimes[t] = digits->GetData(r, c, t);

                        // this value seems to indicate no digit -> skip
                        if (adc == -7169)
                            continue;

                        hAdc->Fill(adc);
                        tbsum += adc;
                    }

                    run3digits.push_back(o2::trd::Digit(det, r, c, adctimes, eventtime));
                    digitscount++;

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

    outfile->Write();
    
    TFile *digitsfile = new TFile("converted_run3_digits.root", "RECREATE");
    TTree *digittree = new TTree("o2sim", "run2 digits");
    std::vector<o2::trd::Digit> *run3pdigits = &run3digits;
    digittree->Branch("TRDDigit", &run3pdigits);
    digittree->Fill();
    cout << " run3digits is : " << run3digits.size() << endl;
    digittree->Write();
    delete digittree;
    delete digitsfile;
}