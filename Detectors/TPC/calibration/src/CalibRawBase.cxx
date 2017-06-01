// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CalibRawBase.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TObjString.h"
#include "TObjArray.h"

#include "TPCCalibration/CalibRawBase.h"

using namespace o2::TPC;

void CalibRawBase::setupContainers(TString fileInfo)
{
  int iSize = 4000000;
  int iCRU = 0;
  int iLink = 0;

  //auto contPtr = std::unique_ptr<GBTFrameContainer>(new GBTFrameContainer(iSize,iCRU,iLink));
  // input data
  TString rorcType="grorc";
  auto arrData = fileInfo.Tokenize("; ");
  for (auto o : *arrData) {
    const TString& data = static_cast<TObjString*>(o)->String();

    // get file info: file name, cru, link
    auto arrDataInfo = data.Tokenize(":");
    if (arrDataInfo->GetEntriesFast() == 1) {
      TString& rorcTypeTmp = static_cast<TObjString*>(arrDataInfo->At(0))->String();
      if (rorcTypeTmp=="trorc") rorcType=rorcTypeTmp;
      else if (rorcTypeTmp=="trorc2") rorcType=rorcTypeTmp;
      else {
        printf("Error, unrecognized option: %s\n", rorcTypeTmp.Data());
      }
      std::cout << "Found rorc type: " << rorcType << "\n";
      delete arrDataInfo;
      continue;
    }
    else if (arrDataInfo->GetEntriesFast() != 3) {
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
    cont->setEnableStoreGBTFrames(true);
    cont->setEnableCompileAdcValues(true);

    std::cout << "Read digits from file " << filename << " with cru " << iCRU << ", link " << iLink << ", rorc type " << rorcType << "...\n";
    cont->addGBTFramesFromBinaryFile(filename.Data(), -1, rorcType.Data());
    std::cout << " ... done. Read " << cont->getSize() << "\n";

    addGBTFrameContainer(cont);
  }

  delete arrData;
}

void CalibRawBase::RewindEvents()
{
  for (auto& c : mGBTFrameContainers) {
    c.get()->reProcessAllFrames();
  }
}
