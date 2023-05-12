// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CalibRawBase.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "TSystem.h"
#include "TGrid.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TPCBase/RDHUtils.h"

#include "TPCCalibration/CalibRawBase.h"

using namespace o2::tpc;

void CalibRawBase::setupContainers(TString fileInfo, uint32_t verbosity, uint32_t debugLevel)
{
  int iSize = 4000000;
  int iCRU = 0;
  int iLink = 0;
  int iSampaVersion = -1;

  // auto contPtr = std::unique_ptr<GBTFrameContainer>(new GBTFrameContainer(iSize,iCRU,iLink));
  //  input data
  TString rorcType = "cru";
  auto arrData = fileInfo.Tokenize("; ");

  std::shared_ptr<RawReaderEventSync> eventSync = std::make_shared<RawReaderEventSync>();

  for (auto o : *arrData) {
    const TString& data = static_cast<TObjString*>(o)->String();

    std::cout << " data: " << data << "\n";
    if (data.Contains(".root")) {
      rorcType = "digits";
      std::cout << "Setting digits\n";
    }

    // get file info: file name, cru, link
    auto arrDataInfo = data.Tokenize(":");
    if (arrDataInfo->GetEntriesFast() == 1) {
      TString& rorcTypeTmp = static_cast<TObjString*>(arrDataInfo->At(0))->String();
      if (rorcTypeTmp == "grorc") {
        rorcType = rorcTypeTmp;
      } else if (rorcTypeTmp == "trorc") {
        rorcType = rorcTypeTmp;
      } else if (rorcTypeTmp == "trorc2") {
        rorcType = rorcTypeTmp;
      } else if (rorcTypeTmp == "raw") {
        rorcType = rorcTypeTmp;
      } else if (rorcTypeTmp == "cru") {
        rorcType = rorcTypeTmp;
      } else {
        printf("Error, unrecognized option: %s\n", rorcTypeTmp.Data());
      }
      std::cout << "Found decoder type: " << rorcType << "\n";
      delete arrDataInfo;
      continue;
    } else if (rorcType == "cru") {
      TString files = gSystem->GetFromPipe(TString::Format("ls %s", arrDataInfo->At(0)->GetName()));
      const int timeBins = static_cast<TObjString*>(arrDataInfo->At(1))->String().Atoi();
      std::unique_ptr<TObjArray> arr(files.Tokenize("\n"));
      mRawReaderCRUManager.reset();
      mPresentEventNumber = 0; // reset event number for readers
      mRawReaderCRUManager.setDebugLevel(debugLevel);
      mRawReaderCRUManager.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> Int_t {
        Int_t timeBins = update(padROCPos, cru, data);
        mProcessedTimeBins = std::max(mProcessedTimeBins, size_t(timeBins));
        return timeBins;
      });
      mRawReaderCRUManager.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
        CRU cruID(cru);
        updateROC(cruID.roc(), rowInSector - (rowInSector > 62) * 63, padInRow, timeBin, adcValue);
        const PadRegionInfo& regionInfo = mMapper.getPadRegionInfo(cruID.region());
        updateCRU(cruID, rowInSector - regionInfo.getGlobalRowOffset(), padInRow, timeBin, adcValue);
        return true;
      });

      for (auto file : *arr) {
        // fix the number of time bins
        auto& reader = mRawReaderCRUManager.createReader(file->GetName(), timeBins);
        reader.setVerbosity(verbosity);
        reader.setDebugLevel(debugLevel);
        reader.setFillADCdataMap(false); // switch off filling and use callback above
        printf("Adding file: %s\n", file->GetName());
        if (arrDataInfo->GetEntriesFast() == 3) {
          const int cru = static_cast<TObjString*>(arrDataInfo->At(2))->String().Atoi();
          reader.forceCRU(cru);
          printf("Forcing CRU %03d\n", cru);
        }
      }
      mRawReaderCRUManager.init();
    } else if (rorcType == "digits") {
      const TString name = arrDataInfo->At(0)->GetName();
      TString cmd = "ls";
      if (name.EndsWith(".list")) {
        cmd = "cat";
      }
      TString files = gSystem->GetFromPipe(TString::Format("%s %s", cmd.Data(), name.Data()));
      // const int timeBins = static_cast<TObjString*>(arrDataInfo->At(1))->String().Atoi();
      std::unique_ptr<TObjArray> arr(files.Tokenize("\n"));
      mDigitTree = std::make_unique<TChain>("o2sim", "Digit chain");
      // mDigitTree = new TChain("Digits","Digit chain");
      for (auto file : *arr) {
        if (TString(file->GetName()).BeginsWith("alien://") && !gGrid) {
          TGrid::Connect("alien");
        }
        mDigitTree->AddFile(file->GetName());
        LOGP(info, "Adding file {}", file->GetName());
      }
    } else if (arrDataInfo->GetEntriesFast() < 3) {
      printf("Error, badly formatte input data string: %s, expected format is <filename:cru:link[:sampaVersion]>\n",
             data.Data());
      delete arrDataInfo;
      continue;
    }

    if (rorcType == "raw") {
      auto rawReader = new RawReader;
      rawReader->addEventSynchronizer(eventSync);
      rawReader->addInputFile(data.Data());

      addRawReader(rawReader);
    } else if ((rorcType != "cru") && (rorcType != "digits")) {
      TString& filename = static_cast<TObjString*>(arrDataInfo->At(0))->String();
      iCRU = static_cast<TObjString*>(arrDataInfo->At(1))->String().Atoi();
      iLink = static_cast<TObjString*>(arrDataInfo->At(2))->String().Atoi();
      if (arrDataInfo->GetEntriesFast() > 3) {
        iSampaVersion = static_cast<TObjString*>(arrDataInfo->At(3))->String().Atoi();
      }

      auto cont = new GBTFrameContainer(iSize, iCRU, iLink, iSampaVersion);

      cont->setEnableAdcClockWarning(false);
      cont->setEnableSyncPatternWarning(false);
      cont->setEnableStoreGBTFrames(false);
      cont->setEnableCompileAdcValues(true);

      std::cout << "Read digits from file " << filename << " with cru " << iCRU << ", link " << iLink << ", rorc type "
                << rorcType << ", SAMPA Version " << iSampaVersion << "...\n";
      cont->addGBTFramesFromBinaryFile(filename.Data(), rorcType.Data(), -1);
      std::cout << " ... done. Read " << cont->getSize() << "\n";

      addGBTFrameContainer(cont);
    }
  }

  delete arrData;
}

void CalibRawBase::rewindEvents()
{
  for (auto& c : mGBTFrameContainers) {
    c.get()->reProcessAllFrames();
  }
}
