#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <TFile.h>
#include <TSystem.h>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iostream>

//ClassImp(o2::itsmft::AlpideSimResponse);

void alpideResponse()
{
    o2::itsmft::AlpideSimResponse resp0, resp1;

    std::string mDataPath = "$O2_ROOT/share/Detectors/ITSMFT/data/alpideResponseData/";
    gSystem->Exec("mkdir -p mDataPath");
    std::string dataFile = mDataPath + "AlpideResponseData.root";

    resp0.initData(0);
    resp1.initData(1);

    auto file = TFile::Open(dataFile.data(),"recreate");
    file->WriteObjectAny(&resp0,"o2::itsmft::AlpideSimResponse","response0");
    file->WriteObjectAny(&resp1,"o2::itsmft::AlpideSimResponse","response1");
    file->Close();
}