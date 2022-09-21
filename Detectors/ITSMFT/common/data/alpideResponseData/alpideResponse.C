#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <TFile.h>
#include <TSystem.h>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iostream>

void alpideResponse()
{
    o2::itsmft::AlpideSimResponse resp0, resp1;

    std::string outputPath = "$O2_ROOT/share/Detectors/ITSMFT/data/alpideResponseData/";
    std::string mkdirCommand = "mkdir -p " + outputPath;
    gSystem->Exec(mkdirCommand.data());
    std::string responseFile = outputPath + "AlpideResponseData.root";

    std::string dataPath = "/home/abigot/alice/O2/Detectors/ITSMFT/common/data/alpideResponseData/";
    resp0.initData(0, dataPath.data());
    resp1.initData(1, dataPath.data());

    auto file = TFile::Open(responseFile.data(), "new");
    file->WriteObjectAny(&resp0,"o2::itsmft::AlpideSimResponse","response0");
    file->WriteObjectAny(&resp1,"o2::itsmft::AlpideSimResponse","response1");
    file->Close();
}