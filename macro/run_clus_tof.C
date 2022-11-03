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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>

#include <fairlogger/Logger.h>
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include "TOFBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TOFReconstruction/ClustererTask.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFBase/CalibTOFapi.h"
#include "TOFReconstruction/Clusterer.h"
#include "TOFReconstruction/DataReader.h"
#endif

void run_clus_tof(std::string outputfile = "tofclusters.root", std::string inputfile = "tofdigits.root", bool isMC = true)
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  // logger->SetLogVerbosityLevel("LOW");
  // logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;
  timer.Start();

  std::vector<o2::tof::Digit> mDigits, *mPdigits = &mDigits;
  std::vector<o2::tof::ReadoutWindowData> mRow, *mProw = &mRow;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels, *mPlabels = &mLabels;
  o2::tof::DigitDataReader reader; ///< Digit reader

  // Load digits
  TFile* file = new TFile(inputfile.c_str(), "OLD");
  std::unique_ptr<TTree> treeDig((TTree*)file->Get("o2sim"));
  if (treeDig) {
    treeDig->SetBranchAddress("TOFDigit", &mPdigits);
    treeDig->SetBranchAddress("TOFReadoutWindow", &mProw);

    if (isMC) {
      treeDig->SetBranchAddress("TOFDigitMCTruth", &mPlabels);
    }

    treeDig->GetEntry(0);
  }

  o2::dataformats::CalibLHCphaseTOF lhcPhaseObj;
  o2::dataformats::CalibTimeSlewingParamTOF channelCalibObj;
  // calibration objects set to zero
  lhcPhaseObj.addLHCphase(0, 0);
  lhcPhaseObj.addLHCphase(2000000000, 0);

  for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
    channelCalibObj.addTimeSlewingInfo(ich, 0, 0);
    int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
    int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
    channelCalibObj.setFractionUnderPeak(sector, channelInSector, 1);
  }
  o2::tof::CalibTOFapi calibapi(long(0), &lhcPhaseObj, &channelCalibObj);

  o2::tof::Clusterer clusterer;
  std::vector<o2::tof::Cluster> clustersArray, *pClustersArray = &clustersArray;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> clsLabels, *pClsLabels = &clsLabels;
  clusterer.setCalibApi(&calibapi);
  if (isMC)
    clusterer.setMCTruthContainer(pClsLabels);

  for (int i = 0; i < mRow.size(); i++) {
    printf("# TOF readout window for clusterization = %d/%d (N digits = %d)\n", i, int(mRow.size()), int(mRow.at(i).size()));
    auto digitsRO = mRow.at(i).getBunchChannelData(mDigits);

    reader.setDigitArray(&digitsRO);

    if (isMC) {
      clusterer.process(reader, clustersArray, &(mPlabels->at(i)));
    } else
      clusterer.process(reader, clustersArray, nullptr);
  }

  LOG(info) << "TOF CLUSTERER : TRANSFORMED " << mDigits.size()
            << " DIGITS TO " << clustersArray.size() << " CLUSTERS";

  TFile* fout = new TFile(outputfile.c_str(), "RECREATE");
  TTree* tout = new TTree("o2sim", "o2sim");
  tout->Branch("TOFCluster", &pClustersArray);
  if (isMC)
    tout->Branch("TOFClusterMCTruth", &pClsLabels);
  tout->Fill();
  tout->Write();
  fout->Close();
  timer.Stop();
  timer.Print();
}
