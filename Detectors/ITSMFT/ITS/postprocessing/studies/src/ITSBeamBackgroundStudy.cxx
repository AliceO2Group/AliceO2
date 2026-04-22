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

#include "ITSStudies/ITSBeamBackgroundStudy.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DataFormatsParameters/GRPObject.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

#include <set>
#include <algorithm>
#include <limits>

#include <TTree.h>
#include <TH2.h>

#include "Framework/Task.h"
#include "Framework/Logger.h"

// ZDC
#include "DataFormatsZDC/RecEventFlat.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2::its::study
{
class ITSBeamBackgroundStudy : public Task
{
 public:
  ITSBeamBackgroundStudy(std::shared_ptr<DataRequest> dr,
                         std::shared_ptr<o2::base::GRPGeomRequest> gr,
                         bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC) {}

  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void save_and_reset();

  // Custom
  void process(o2::globaltracking::RecoContainer& recoData);
  void updateTimeDependentParams(ProcessingContext& pc);

 private:
  void getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>&, gsl::span<const unsigned char>&, const o2::itsmft::TopologyDictionary&);
  std::vector<o2::itsmft::ClusterPattern> mPatterns;

  // ITS layout
  int NStaves[7] = {12, 16, 20, 24, 30, 42, 48};
  int N_STAVES_IB = 48;
  int N_CHIP_IB = 432;

  // Utilities
  int ChipToLayer(int chip);
  double ChipToPhi(int chip);
  bool searchBCfromMap(std::map<long, std::set<int>>& BCperorbit, long target_orbit, int target_bc);

  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  int mTFn = 0;
  int mTF_first_after_dump = 1;

  int mStrobeFallBack = 594;
  int mStrobe = mStrobeFallBack;

  // TODO: the following should be make configurable

  std::pair<double, double> TimeWindowZDC = std::make_pair(5., 11.);
  std::pair<double, double> TimeWindowZNAr = std::make_pair(5.64507, 9.64507);
  std::pair<double, double> TimeWindowZNCr = std::make_pair(4.21299, 8.21299);
  std::pair<double, double> TimeWindowZNAl = std::make_pair(-11.3549, -8.35493);
  std::pair<double, double> TimeWindowZNCl = std::make_pair(-12.787, -9.78701);

  int targetClusterMinCol = 128; // definition of anomalous cluster
  int targetClusterMaxRow = 29;  // definition of anomalous cluster

  int mDumpEveryTF = 10; // use -1 to save only at the end
  int mSkimmedOnlyAfterTF = 15000;
  std::string mOutputChip = "chipevents";
  std::string mOutoutChipSkim = "chipeventstarget";

  TH1F* TimeWindowCut;
  TH1F* ZNACall;
  TH1F* ZNCCall;
  TH1F* ZDCAtagBC;
  TH1F* ZDCCtagBC;
  TH1I* Counters;
  TTree* ITSChipEvtTree;
  TTree* ITSChipEvtTargetTree;

  // Tree variables
  int Tbc;
  long Torbit;
  int Tchip;
  double Tphi;
  int TZDCtag;
  int Tnhit, Tnclus, Tnhit_no1pix, Tnclus_no1pix;
  int Tnclus_s20, Tnclus_s100, Tnclus_s150;
  int Tnclus_c20, Tnclus_c100, Tnclus_c128;
  int Tnclus_target;
  double Tnhit1, Tnhit10;
  int Tmissingafter, Tmissingafter2;
  int Tmincol, Tmaxcol;
};

void ITSBeamBackgroundStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  // o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  // static bool initOnceDone = false;
  // if (!initOnceDone) { // this param need to be queried only once
  //   initOnceDone = true;
  //   // mGeom = o2::its::GeometryTGeo::Instance();
  //   // mGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  // }
}

void ITSBeamBackgroundStudy::init(InitContext& ic)
{
  LOGP(info, "Initializing ITSBeamBackgroundStudy");
  LOGP(info, "Fetching ClusterDictionary");
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL("http://alice-ccdb.cern.ch");
  mgr.setTimestamp(o2::ccdb::getCurrentTimestamp());
  mDict = mgr.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");

  LOGP(info, "Setting up trees and histograms. They will be dumped on file every {} TFs", mDumpEveryTF);

  TimeWindowCut = new TH1F("ZDC bkg region", "ZNAr, ZNCr, -ZNAl, -ZNCl", 8, 0, 8);
  TimeWindowCut->SetBinContent(1, TimeWindowZNAr.first);
  TimeWindowCut->SetBinContent(2, TimeWindowZNAr.second);
  TimeWindowCut->SetBinContent(3, TimeWindowZNCr.first);
  TimeWindowCut->SetBinContent(4, TimeWindowZNCr.second);
  TimeWindowCut->SetBinContent(5, -TimeWindowZNAl.first);
  TimeWindowCut->SetBinContent(6, -TimeWindowZNAl.second);
  TimeWindowCut->SetBinContent(7, -TimeWindowZNCl.first);
  TimeWindowCut->SetBinContent(8, -TimeWindowZNCl.second);
  ZNACall = new TH1F("ZNACall", "ZNACall", 40, -20, 20);
  ZNCCall = new TH1F("ZNCCall", "ZNCCall", 40, -20, 20);
  ZDCAtagBC = new TH1F("ZDCA tagged BC", "ZDCA tagged bc", 3564, 0, 3564);
  ZDCCtagBC = new TH1F("ZDCC tagged BC", "ZDCC tagged bc", 3564, 0, 3564);

  Counters = new TH1I("Counters", "Counters", 20, 1, 21);
  Counters->GetXaxis()->SetBinLabel(1, "TF");
  Counters->GetXaxis()->SetBinLabel(2, "ROF");
  Counters->GetXaxis()->SetBinLabel(3, "ZDCA evt");
  Counters->GetXaxis()->SetBinLabel(4, "ROF-ZDCA tagged");
  Counters->GetXaxis()->SetBinLabel(5, "ITStag any");
  Counters->GetXaxis()->SetBinLabel(6, "ITStag TO");
  Counters->GetXaxis()->SetBinLabel(7, "ITStag any + ZDC");
  Counters->GetXaxis()->SetBinLabel(8, "ITStag TO + ZDC");
  Counters->GetXaxis()->SetBinLabel(9, "ZDCC evt");
  Counters->GetXaxis()->SetBinLabel(10, "ROF-ZDCC tagged");

  ITSChipEvtTree = new TTree("chipevt", "chipevt");

  // Chip event branches
  ITSChipEvtTree->Branch("TFprogress", &mTFn, "TFprogress/I");
  ITSChipEvtTree->Branch("orbit", &Torbit, "orbit/L");
  ITSChipEvtTree->Branch("bc", &Tbc, "bc/I");
  ITSChipEvtTree->Branch("chip", &Tchip, "chip/I");
  ITSChipEvtTree->Branch("phi", &Tphi, "phi/D");
  ITSChipEvtTree->Branch("zdctag", &TZDCtag, "zdctag/I");
  ITSChipEvtTree->Branch("nhit", &Tnhit, "nhit/I");
  ITSChipEvtTree->Branch("nhit_no1pix", &Tnhit_no1pix, "nhit_no1pix/I");
  ITSChipEvtTree->Branch("size1", &Tnhit1, "size1/D");
  ITSChipEvtTree->Branch("size10", &Tnhit10, "size10/D");
  ITSChipEvtTree->Branch("nclus", &Tnclus, "nclus/I");
  ITSChipEvtTree->Branch("nclus_no1pix", &Tnclus_no1pix, "nclus_no1pix/I");
  ITSChipEvtTree->Branch("nclus_target", &Tnclus_target, "nclus_target/I");
  ITSChipEvtTree->Branch("missingafter", &Tmissingafter, "missingafter/I");
  ITSChipEvtTree->Branch("missingafter2", &Tmissingafter2, "missingafter2/I");
  ITSChipEvtTree->Branch("mincol", &Tmincol, "mincol/I");
  ITSChipEvtTree->Branch("maxcol", &Tmaxcol, "maxcol/I");

  ITSChipEvtTargetTree = ITSChipEvtTree->CloneTree(0);
  ITSChipEvtTargetTree->SetName("chipevttarget");
  ITSChipEvtTargetTree->SetTitle("chipevttarget");
}

void ITSBeamBackgroundStudy::save_and_reset()
{

  std::string outfile11 = mOutoutChipSkim + "_" + std::to_string(mTF_first_after_dump) + "_" + std::to_string(mTFn) + ".root";
  LOGP(info, "Writing ROOT file {}", outfile11);
  TFile* F11 = TFile::Open(outfile11.c_str(), "recreate");
  TimeWindowCut->Write();
  ZNACall->Write();
  ZNCCall->Write();
  ZDCAtagBC->Write();
  ZDCCtagBC->Write();
  Counters->Write();
  ITSChipEvtTargetTree->Write();
  F11->Close();
  delete F11;

  if (mTFn <= mSkimmedOnlyAfterTF) {
    std::string outfile1 = mOutputChip + "_" + std::to_string(mTF_first_after_dump) + "_" + std::to_string(mTFn) + ".root";
    LOGP(info, "Writing ROOT file {}", outfile1);
    TFile* F1 = TFile::Open(outfile1.c_str(), "recreate");
    TimeWindowCut->Write();
    ZNACall->Write();
    ZNCCall->Write();
    ZDCAtagBC->Write();
    ZDCCtagBC->Write();
    Counters->Write();
    ITSChipEvtTree->Write(); // chip events and the skimmed one
    ITSChipEvtTargetTree->Write();
    F1->Close();
    delete F1;
  }

  LOGP(info, "Resetting historgrams and trees");
  // Delete clears data but keep the branch setup intact
  ITSChipEvtTree->Reset();       // Delete("");
  ITSChipEvtTargetTree->Reset(); // Delete("");
  ZNACall->Reset();
  ZNCCall->Reset();
  ZDCAtagBC->Reset();
  ZDCCtagBC->Reset();
  Counters->Reset();

  mTF_first_after_dump = mTFn + 1;
}

void ITSBeamBackgroundStudy::endOfStream(EndOfStreamContext&)
{
  LOGP(info, "End of stream for ITSBeamBackgroundStudy");

  save_and_reset();

  delete TimeWindowCut;
  delete ZNACall;
  delete ZNCCall;
  delete ZDCAtagBC;
  delete ZDCCtagBC;
  delete Counters;
  delete ITSChipEvtTree;
  delete ITSChipEvtTargetTree;
}

void ITSBeamBackgroundStudy::run(ProcessingContext& pc)
{

  if (mTFn == std::numeric_limits<int>::max()) {
    LOGP(error, "Max {} TFs exceeded. Skipping all next events", mTFn);
    return;
  }

  mTFn++;

  if (mDumpEveryTF > 0 && mTFn > 0 && (mTFn % mDumpEveryTF) == 0) {
    LOGP(info, "Reached TF #{}. Exporting new root files", mTFn);
    save_and_reset();
  }

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  // updateTimeDependentParams(pc);
  LOGP(info, "Calling process() for TF: {}", mTFn);
  process(recoData);
}

void ITSBeamBackgroundStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  return;
}

// Custom area
void ITSBeamBackgroundStudy::process(o2::globaltracking::RecoContainer& recoData)
{

  LOGP(info, "Processing RecoContainer");
  Counters->Fill(1);

  LOGP(info, "Retrieving ZDC data");
  auto RecBC = recoData.getZDCBCRecData();
  auto Energy = recoData.getZDCEnergy();
  auto TDCData = recoData.getZDCTDCData();
  auto Info2 = recoData.getZDCInfo();
  LOGP(info, "sizeof ZDC RC: {}, {}, {}, {}", RecBC.size(), Energy.size(), TDCData.size(), Info2.size());

  LOGP(info, "Retrieving ITS clusters");
  auto rofRecVec = recoData.getITSClustersROFRecords();
  auto clusArr = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  LOGP(info, "sizeof ITS RC: {}, {}, {}", clusArr.size(), clusPatt.size(), rofRecVec.size());

  // TODO: improve this
  if (rofRecVec.size() == 576 || rofRecVec.size() == 192) {
    mStrobe = 3564 / (rofRecVec.size() / 32);
    LOGP(info, "Assuimg TF length = 32 orbits and setting strobe length to {} bc", mStrobe);
  } else {
    mStrobe = mStrobeFallBack;
    LOGP(warning, "Unforeseen number of ROFs in the loop. Using the strobe length fall back value {}", mStrobe);
  }

  std::map<long, std::set<int>> ZNArtag{}; // ZDCAtag[orbit] = <list of bc...>
  std::map<long, std::set<int>> ZNCrtag{};
  std::map<long, std::set<int>> ZNAltag{};
  std::map<long, std::set<int>> ZNCltag{};

  // ________________________________________________________________
  // FILLING ZDC ARRAY
  o2::zdc::RecEventFlat ev;

  ev.init(RecBC, Energy, TDCData, Info2);

  int bkgcounterAr = 0, bkgcounterCr = 0;
  int bkgcounterAl = 0, bkgcounterCl = 0;
  while (ev.next()) {

    int32_t itdcA = o2::zdc::TDCZNAC; // should be == 0
    int32_t itdcC = o2::zdc::TDCZNCC;
    long zdcorbit = (long)ev.ir.orbit;

    // ZDC - A side
    int nhitA = ev.NtdcV(itdcA);
    for (int32_t ipos = 0; ipos < nhitA; ipos++) {

      double mytdc = o2::zdc::FTDCVal * ev.TDCVal[itdcA][ipos];

      ZNACall->Fill(mytdc);

      if (mytdc >= TimeWindowZNAr.first && mytdc <= TimeWindowZNAr.second) {

        // Backgroud event found here!
        bkgcounterAr++;
        Counters->Fill(3);
        ZDCAtagBC->Fill(ev.ir.bc);

        if (ZNArtag.find(zdcorbit) != ZNArtag.end()) {
          bool double_count_bkg = ZNArtag[zdcorbit].insert((int)ev.ir.bc).second;
          if (double_count_bkg) {
            LOGP(warning, "Multiple ZDCAr counts in the same orbit/bc {}/{}", zdcorbit, ev.ir.bc);
          }
        } else {
          std::set<int> zdcbcs{(int)ev.ir.bc};
          ZNArtag[zdcorbit] = zdcbcs;
        }

      } // and of ZNAr time window

      if (mytdc >= TimeWindowZNAl.first && mytdc <= TimeWindowZNAl.second) {

        // Backgroud event found here!
        bkgcounterAl++;
        Counters->Fill(3);
        ZDCAtagBC->Fill(ev.ir.bc);

        if (ZNAltag.find(zdcorbit) != ZNAltag.end()) {
          bool double_count_bkg = ZNAltag[zdcorbit].insert((int)ev.ir.bc).second;
          if (double_count_bkg) {
            LOGP(warning, "Multiple ZDCAl counts in the same orbit/bc {}/{}", zdcorbit, ev.ir.bc);
          }
        } else {
          std::set<int> zdcbcs{(int)ev.ir.bc};
          ZNAltag[zdcorbit] = zdcbcs;
        }

      } // and of ZNAl time window
    }

    // ZDC - C side
    int nhitC = ev.NtdcV(itdcC);
    for (int32_t ipos = 0; ipos < nhitC; ipos++) {

      double mytdc = o2::zdc::FTDCVal * ev.TDCVal[itdcC][ipos];

      ZNCCall->Fill(mytdc);

      if (mytdc >= TimeWindowZNCr.first && mytdc <= TimeWindowZNCr.second) {

        // Backgroud event found here!
        bkgcounterCr++;
        Counters->Fill(9);
        ZDCCtagBC->Fill(ev.ir.bc);

        if (ZNCrtag.find(zdcorbit) != ZNCrtag.end()) {
          bool double_count_bkg = ZNCrtag[zdcorbit].insert((int)ev.ir.bc).second;
          if (double_count_bkg) {
            LOGP(warning, "Multiple ZNCr counts in the same orbit/bc {}/{}", zdcorbit, ev.ir.bc);
          }
        } else {
          std::set<int> zdcbcs{(int)ev.ir.bc};
          ZNCrtag[zdcorbit] = zdcbcs;
        }

      } // end of ZNCr time window

      if (mytdc >= TimeWindowZNCl.first && mytdc <= TimeWindowZNCl.second) {

        // Backgroud event found here!
        bkgcounterCl++;
        Counters->Fill(9);
        ZDCCtagBC->Fill(ev.ir.bc);

        if (ZNCltag.find(zdcorbit) != ZNCltag.end()) {
          bool double_count_bkg = ZNCltag[zdcorbit].insert((int)ev.ir.bc).second;
          if (double_count_bkg) {
            LOGP(warning, "Multiple ZNCl counts in the same orbit/bc {}/{}", zdcorbit, ev.ir.bc);
          }
        } else {
          std::set<int> zdcbcs{(int)ev.ir.bc};
          ZNCltag[zdcorbit] = zdcbcs;
        }

      } // end of ZNCl time window
    }
  } // end of while ev.next()

  LOGP(info, "Found background envents from ZNAright/left {}/{} -- from ZNCright/left {}/{}", bkgcounterAr, bkgcounterAl, bkgcounterCr, bkgcounterCl);
  //__________________________________________________________________

  getClusterPatterns(clusArr, clusPatt, *mDict);

  int inTFROFcounter = -1;

  std::vector<bool> ChipSeenInThisROF(N_CHIP_IB, false);  // ChipSeenInThisROF[chipid] = true/false
  std::vector<bool> ChipSeenInLastROF(N_CHIP_IB, false);  // ChipSeenInLastROF[chipid] = true/false
  std::vector<bool> ChipSeenInLast2ROF(N_CHIP_IB, false); // ChipSeenInLast2ROF[chipid] = true/false

  // Begin loop over ROFs
  for (auto it = rofRecVec.rbegin(); it != rofRecVec.rend(); ++it) {

    auto& rofRec = *it;

    inTFROFcounter++;

    Counters->Fill(2);

    ChipSeenInLast2ROF = ChipSeenInLastROF;
    ChipSeenInLastROF = ChipSeenInThisROF;
    std::fill(ChipSeenInThisROF.begin(), ChipSeenInThisROF.end(), false);

    auto clustersInRof = rofRec.getROFData(clusArr);
    auto patternsInRof = rofRec.getROFData(mPatterns);

    Tbc = (int)rofRec.getBCData().bc;
    Torbit = (long)rofRec.getBCData().orbit;

    if (inTFROFcounter < 1) {
      LOGP(info, "First of TF: ITS orbit/bc {}/{}", Torbit, Tbc);
    }

    // shifting by 60 bc
    int eff_bc = Tbc + 60;
    long eff_orbit = Torbit;
    if (eff_bc > 3563) {
      eff_bc -= 3564;
      eff_orbit += 1;
    }

    // Making a bitmask with ZDC tags for this bc
    bool isZNArtagged = searchBCfromMap(ZNArtag, (long)eff_orbit, eff_bc);
    if (isZNArtagged) {
      Counters->Fill(4);
    }

    bool isZNAltagged = searchBCfromMap(ZNAltag, (long)eff_orbit, eff_bc);
    if (isZNAltagged) {
      Counters->Fill(4);
    }

    bool isZNCrtagged = searchBCfromMap(ZNCrtag, (long)eff_orbit, eff_bc);
    if (isZNCrtagged) {
      Counters->Fill(10);
    }

    bool isZNCltagged = searchBCfromMap(ZNCltag, (long)eff_orbit, eff_bc);
    if (isZNCltagged) {
      Counters->Fill(10);
    }

    TZDCtag = 0;
    TZDCtag |= (isZNArtagged << 0);
    TZDCtag |= (isZNAltagged << 1);
    TZDCtag |= (isZNCrtagged << 2);
    TZDCtag |= (isZNCltagged << 3);

    if (TZDCtag > 0) {
      LOGP(info, "ZDC tag with mask {}: ZNAright = {} - ZNAleft = {} - ZNCright = {} - ZNCleft = {}",
           TZDCtag, (TZDCtag >> 0) & 1, (TZDCtag >> 1) & 1, (TZDCtag >> 2) & 1, (TZDCtag >> 3) & 1);
    }

    // preparing arrays for clusters analysis
    std::set<int> AvailableChips{};
    std::map<int, std::vector<int>> MAPsize{}; // MAP[chip] = {list if sizes}
    std::map<int, std::vector<int>> MAPcols{}; // MAP[chip] = {list of column span}
    std::map<int, int> MAPntarget{};           // MAP[chip] = number of bad clusters in chip
    std::map<int, int> MAPcoo_mincol{};        // MAP[chip] = minimum of column coordinate
    std::map<int, int> MAPcoo_maxcol{};        // MAP[chip] = maximum of (column coordinate + colspan)

    // Finally loop over clusters
    int ntarget_in_rof = 0;
    for (int iclus = 0; iclus < clustersInRof.size(); iclus++) {

      const auto& compClus = clustersInRof[iclus];

      auto chipid = compClus.getSensorID();

      // Analyze only IB
      if (ChipToLayer(chipid) > 2) {
        continue;
      }

      ChipSeenInThisROF[chipid] = true;

      int coo_col = (int)compClus.getCol();
      int coo_row = (int)compClus.getRow();

      auto patti = patternsInRof[iclus];
      int npix = patti.getNPixels();
      int colspan = patti.getColumnSpan();
      int rowspan = patti.getRowSpan();

      bool newchip = AvailableChips.insert(chipid).second;
      if (newchip) {
        MAPsize[chipid] = std::vector<int>{};
        MAPcols[chipid] = std::vector<int>{};
        MAPntarget[chipid] = 0;
        MAPcoo_mincol[chipid] = coo_col;
        MAPcoo_maxcol[chipid] = coo_col + colspan;
      }

      MAPsize[chipid].push_back(npix);
      MAPcols[chipid].push_back(colspan);
      if (colspan >= targetClusterMinCol && rowspan <= targetClusterMaxRow) {
        // Anomalous cluster found
        MAPntarget[chipid] += 1;
        ntarget_in_rof++;
      }
      MAPcoo_mincol[chipid] = TMath::Min(MAPcoo_mincol[chipid], coo_col);
      MAPcoo_maxcol[chipid] = TMath::Max(MAPcoo_maxcol[chipid], coo_col + colspan);

    } // end of loop over clusters in rof

    if (ntarget_in_rof == 0 && mTFn > mSkimmedOnlyAfterTF + 2) { // extra 2 to avoid edge effects?
      // do not need extra computations for this rof since it will not be saved in any case
      continue;
    }

    for (int ic : AvailableChips) {

      Tchip = ic;

      if (inTFROFcounter < 1) {
        Tmissingafter = -1;
      } else if (ChipSeenInLastROF[ic]) {
        Tmissingafter = 0;
      } else {
        Tmissingafter = 1;
      }

      if (inTFROFcounter < 2) {
        Tmissingafter2 = -1;
      } else if (ChipSeenInLast2ROF[ic]) {
        Tmissingafter2 = 0;
      } else {
        Tmissingafter2 = 1;
      }

      Tphi = ChipToPhi(ic);

      Tnclus = MAPsize[ic].size();
      Tmincol = MAPcoo_mincol[ic];
      Tmaxcol = MAPcoo_maxcol[ic];

      std::sort(MAPsize[ic].begin(), MAPsize[ic].end(), std::greater<>());

      Tnhit = Tnclus_s20 = Tnclus_s100 = Tnclus_s150 = 0;
      Tnhit1 = Tnhit10 = 0.;
      Tnclus_c20 = Tnclus_c100 = Tnclus_c128 = 0;
      Tnclus_target = MAPntarget[ic];
      Tnhit_no1pix = 0;
      Tnclus_no1pix = 0;

      int nhit_no1pix = 0;
      int nclus10 = 0, nclus1 = 0;

      for (int nh : MAPsize[ic]) {

        Tnhit += nh;

        if (nh > 1) {
          Tnhit_no1pix += nh;
          Tnclus_no1pix += 1;
        }

        if (nclus10 < 10) {
          nclus10++;
          Tnhit10 += 1. * nh;
        }

        if (nclus1 < 1) {
          nclus1++;
          Tnhit1 += 1. * nh;
        }

        Tnclus_s20 += (nh >= 20);
        Tnclus_s100 += (nh >= 100);
        Tnclus_s150 += (nh >= 150);
      }

      Tnhit10 = (nclus10 == 0) ? 0. : 1. * Tnhit10 / nclus10;

      for (int nc : MAPcols[ic]) {
        Tnclus_c20 += (nc >= 20);
        Tnclus_c100 += (nc >= 100);
        Tnclus_c128 += (nc >= 128);
      }

      ITSChipEvtTree->Fill();
      if (Tnclus_target > 0) {
        ITSChipEvtTargetTree->Fill();
      }

    } // end of loop over available chips
  } // end of loop over ROFs
}

// TODO: To be improved using geometry tools
int ITSBeamBackgroundStudy::ChipToLayer(int chip)
{
  if (chip < 108) {
    return 0;
  }
  if (chip < 252) {
    return 1;
  }
  if (chip < 432) {
    return 2;
  }
  if (chip < 3120) {
    return 3;
  }
  if (chip < 6480) {
    return 4;
  }
  if (chip < 14712) {
    return 5;
  }
  return 6;
}

// TODO: To be improved using geometry tools
double ITSBeamBackgroundStudy::ChipToPhi(int chip)
{
  int staveinlayer = (int)(chip / 9);
  for (int il = 0; il < ChipToLayer(chip); il++) {
    staveinlayer -= NStaves[il];
  }
  return 2. * TMath::Pi() * (0.5 + staveinlayer) / NStaves[ChipToLayer(chip)];
}

bool ITSBeamBackgroundStudy::searchBCfromMap(std::map<long, std::set<int>>& BCperorbit, long its_orbit, int its_bc)
{
  auto it = BCperorbit.find(its_orbit);
  if (it == BCperorbit.end()) {
    return false;
  }

  for (auto bc : it->second) {
    if ((bc / mStrobe) == (its_bc / mStrobe)) {
      return true;
    }
  }
  return false;
}

void ITSBeamBackgroundStudy::getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>& ITSclus, gsl::span<const unsigned char>& ITSpatt, const o2::itsmft::TopologyDictionary& mdict)
{
  mPatterns.clear();
  mPatterns.reserve(ITSclus.size());
  auto pattIt = ITSpatt.begin();

  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];

    auto pattID = clus.getPatternID();
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
      patt.acquirePattern(pattIt);
    } else {
      patt = mdict.getPattern(pattID);
    }

    mPatterns.push_back(patt);
  }
}

// getter
DataProcessorSpec getITSBeamBackgroundStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
{

  // std::cout<<"DEBBUG track and clus masks "<<srcTracksMask<<" "<<srcClustersMask<<" is ZDC in tracks: "<<(srcTracksMask & GTrackID::getSourcesMask("ZDC"))<<" is ITS in clus: "<<(srcClustersMask & GTrackID::getSourcesMask("ITS"))<<std::endl;

  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestClusters(srcClustersMask, useMC);
  // dataRequest->requestTracks(GTrackID::getSourcesMask("ZDC"), useMC);

  dataRequest->requestTracks(srcTracksMask, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-beambkg-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSBeamBackgroundStudy>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace o2::its::study
