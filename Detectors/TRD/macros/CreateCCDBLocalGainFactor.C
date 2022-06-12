#include "TRDCalibration/PadCalibCCDBBuilder.h"

#include <ctime>
#include <sstream>
#include <string>

using namespace o2::trd;

void CreateCCDBLocalGainFactor( TString sOpenFile )
{
    PadCalibCCDBBuilder* krCalibToCCDB = new PadCalibCCDBBuilder();

    TFile* file = TFile::Open( sOpenFile );
    TTree* tree = (TTree*) file->Get("nt_Krypton");
    LocalGainFactor calObject; 

    for( int idet = 0; idet < 540; idet++ ) {
        TH2F* hDetDef = krCalibToCCDB->GetDetectorMap( tree, idet);
        krCalibToCCDB->SmoothenTheDetector(hDetDef);
        TH2F* hDetFilled = krCalibToCCDB->FillTheMap(hDetDef);
        delete hDetDef;
        TH2F* hDet = krCalibToCCDB->CreateNormalizedMap(hDetFilled);
        delete hDetFilled;
        for(int irow = 0; irow < hDet->GetNbinsY(); irow++ ) {
            for(int icol = 0; icol < hDet->GetNbinsX(); icol++ ) {
                float relativeGain = hDet->GetBinContent(hDet->GetXaxis()->FindBin(icol), hDet->GetYaxis()->FindBin(irow));
                calObject.setPadValue(idet, icol, irow, relativeGain);
            }
        }        
    }


    ///write map to CCDB
    o2::ccdb::CcdbApi ccdb;
    ccdb.init("http://ccdb-test.cern.ch:8080");
    // o2::base::NameConf::getCCDBServer()
    std::map<std::string, std::string> metadata;

    time_t start_time = 1631311200; // Fri Sep 10 2021 22:00:00 GMT+0000
    time_t end_time =  	2208985200; // Sat Dec 31 2039 23:00:00 GMT+0000
    auto timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::from_time_t(start_time).time_since_epoch()).count();
    auto timeStampEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::from_time_t(end_time).time_since_epoch()).count();
    ccdb.storeAsTFileAny(&calObject, "TRD/Calib/CalPadGain", metadata, timeStamp, timeStampEnd); 
    cout << "test completed" << endl;
}