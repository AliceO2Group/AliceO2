/// \file CheckITSTracksVsROF.C
/// \brief Simple macro to check ITS tracks in AO2D files as a function of bunch crossing within a ROF

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <CCDB/BasicCCDBManager.h>
#include <CommonConstants/LHCConstants.h>
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsFIT/Triggers.h"
#include <DataFormatsFT0/Digit.h>
#include <Framework/DataTypes.h>
#include <DataFormatsParameters/AggregatedRunInfo.h>
#endif

int32_t nBCsPerOrbit = o2::constants::lhc::LHCMaxBunches;
int32_t nBCsPerITSROF = 198;
int32_t offsetITSROF = 64; // from current analysis of delay of ITS ROF with respect to other detectors
constexpr float minCollTimeFT0A = -5.f;
constexpr float maxCollTimeFT0A = 5.f;
constexpr float minCollTimeFT0C = -5.f;
constexpr float maxCollTimeFT0C = 5.f;
constexpr float minCollTimeFV0A = -5.f;
constexpr float maxCollTimeFV0A = 5.f;
constexpr float maxZv = 10.f;
constexpr float maxTrackEta = 0.8f;

void CheckITSTracksVsROF(int maxFiles = 6, int maxDF = 99999)
{

  printf("nBCsPerOrbit = %d\n", nBCsPerOrbit);
  // define histograms
  TH1F* hTrNoColl = new TH1F("hTrNoColl", "", 2, -0.5, 1.5);
  hTrNoColl->GetXaxis()->SetBinLabel(1, "No collision assoc.");
  hTrNoColl->GetXaxis()->SetBinLabel(2, "With collision assoc.");
  TH1F* hTrTime = new TH1F("hTrTime", ";time;entries", 100, -10000., 10000.);
  TH1F* hCollTime = new TH1F("hCollTime", ";time;entries", 100, -20., 20.);
  TH1F* hVz = new TH1F("hVz", ";z_{v} (cm);entries", 100, -20., 20.);
  TH2F* hTrVsCollTime = new TH2F("hTrVsCollTime", ";coll time;track time", 100, -20., 20., 100, -10000., 10000.);
  int nBinsROF = 395;
  float minROF = -197.5;
  float maxROF = 197.5;
  TH2F* hCluTrackVsBCinROF = new TH2F("hCluTrackVsBC", ";bcInROF;N_{ITSclusters}", nBinsROF, minROF, maxROF, 8, -0.5, 7.5);
  TH1F* hNEventsVsBC = new TH1F("hNEventsVsBC", ";bc;N_{Collisions}", 3601, -0.5, 3600.5);
  TH1F* hNTracksVsBC = new TH1F("hNTracksVsBC", ";bc;N_{Tracks}", 3601, -0.5, 3600.5);
  TH1F* hNTracksPerEventVsBC = new TH1F("hNTracksPerEventVsBC", ";bc;N_{Tracks} per event", 3601, -0.5, 3600.5);
  TH1F* hNEventsVsBCinROF = new TH1F("hNEventsVsBCinROF", ";bcInROF;N_{Collisions}", nBinsROF, minROF, maxROF);
  TH1F* hNTracksVsBCinROF = new TH1F("hNTracksVsBCinROF", ";bcInROF;N_{Tracks}", nBinsROF, minROF, maxROF);
  TH1F* hNTracksPerEventVsBCinROF = new TH1F("hNTracksPerEventVsBCinROF", ";bcInROF;N_{Tracks} per event", nBinsROF, minROF, maxROF);
  TH2F* hNPVContribVsBCinROF = new TH2F("hNPVContribVsBC", ";bcInROF;N_{PVcontrib}", nBinsROF, minROF, maxROF, 151, -0.5, 150.5);

  // process all AO2D files in subdirectories of the current directory
  gSystem->Exec("find ./ -name AO2D.root | sort > filelist.txt");
  int nFiles = 0;
  std::ifstream aodList("filelist.txt");
  std::string line;
  std::vector<std::string> filNames;
  while (std::getline(aodList, line)) {
    if (!line.empty()) {
      filNames.push_back(line);
      ++nFiles;
    }
  }
  if (nFiles > maxFiles)
    nFiles = maxFiles;

  for (int jf = 0; jf < nFiles; ++jf) {
    TFile* fil = TFile::Open(filNames[jf].c_str());
    if (!fil || fil->IsZombie()) {
      printf("Cannot open %s\n", filNames[jf].c_str());
      continue;
    }
    TList* l = (TList*)fil->GetListOfKeys();
    int nKeys = l->GetEntries();
    for (int j = 0; j < nKeys; ++j) {
      if (j >= maxDF)
        break;
      TKey* k = (TKey*)l->At(j);
      TString cname = k->GetClassName();
      if (cname == "TDirectoryFile") {
        TDirectoryFile* d = (TDirectoryFile*)fil->Get(k->GetName());
        printf("%s %s\n", fil->GetName(), d->GetName());

        // Access TTree with BC info
        TTree* tb = (TTree*)d->Get("O2bc_001");
        if (!tb) {
          printf("TTree O2bc_001 missing\n");
          continue;
        }
        printf(" BC Tree entries = %lld\n", tb->GetEntries());
        ULong64_t gloBC;
        int nRun;
        tb->SetBranchAddress("fRunNumber", &nRun);
        tb->SetBranchAddress("fGlobalBC", &gloBC);

        // Access TTree with collision info
        TTree* tc = (TTree*)d->Get("O2collision_001");
        if (!tc) {
          printf("TTree O2collision_001 missing\n");
          continue;
        }
        printf(" Collision Tree entries = %lld\n", tc->GetEntries());
        float xv, yv, zv, ctime;
        int iBC;
        ushort nPVcontrib;
        tc->SetBranchAddress("fPosX", &xv);
        tc->SetBranchAddress("fPosY", &yv);
        tc->SetBranchAddress("fPosZ", &zv);
        tc->SetBranchAddress("fCollisionTime", &ctime);
        tc->SetBranchAddress("fNumContrib", &nPVcontrib);
        tc->SetBranchAddress("fIndexBCs", &iBC);

        // Access TTree with FT0 info
        TTree* tft = (TTree*)d->Get("O2ft0");
        if (!tft) {
          printf("TTree O2ft0 missing\n");
          continue;
        }
        printf(" FT0 Tree entries = %lld\n", tft->GetEntries());
        float timeft0a, timeft0c;
        int iBCft0;
        UChar_t trigMask;
        tft->SetBranchAddress("fIndexBCs", &iBCft0);
        tft->SetBranchAddress("fTimeA", &timeft0a);
        tft->SetBranchAddress("fTimeC", &timeft0c);
        tft->SetBranchAddress("fTriggerMask", &trigMask);

        // Access TTree with FV0 info
        TTree* tfv = (TTree*)d->Get("O2fv0a");
        if (!tfv) {
          printf("TTree O2fv0a missing\n");
          continue;
        }
        printf(" FV0A Tree entries = %lld\n", tfv->GetEntries());
        float timefv0a;
        int iBCfv0;
        tfv->SetBranchAddress("fIndexBCs", &iBCfv0);
        tfv->SetBranchAddress("fTime", &timefv0a);

        // Access TTree with track parameters at their innermost update
        TTree* tt = (TTree*)d->Get("O2track_iu");
        if (!tt) {
          printf("TTree O2track_iu missing\n");
          continue;
        }
        printf(" Track Tree entries = %lld\n", tt->GetEntries());
        // 5 parameters defining the track helix: y, z, sin(phi), tan(lambda), q/pt
        // x is a coordinate along the track
        // alpha: angle between track reference system and global reference system
        float x, alp, y, z, snp, tgl, qpt;
        int iColl;
        tt->SetBranchAddress("fIndexCollisions", &iColl);
        tt->SetBranchAddress("fX", &x);
        tt->SetBranchAddress("fAlpha", &alp);
        tt->SetBranchAddress("fY", &y);
        tt->SetBranchAddress("fZ", &z);
        tt->SetBranchAddress("fSnp", &snp);
        tt->SetBranchAddress("fTgl", &tgl);
        tt->SetBranchAddress("fSigned1Pt", &qpt);

        // Access TTree with track extra information
        TTree* te = (TTree*)d->Get("O2trackextra_002");
        if (!te)
          te = (TTree*)d->Get("O2trackextra_001");
        if (!te) {
          printf("TTree O2trackextra_002 and O2trackextra_001 both missing\n");
          continue;
        }
        uint itsclusiz;
        uint8_t nTPCclu;
        uint trflag;
        float itschi2, tpcchi2, trtime;
        te->SetBranchAddress("fITSClusterSizes", &itsclusiz);
        te->SetBranchAddress("fITSChi2NCl", &itschi2);
        te->SetBranchAddress("fTPCNClsFindable", &nTPCclu);
        te->SetBranchAddress("fTPCChi2NCl", &tpcchi2);
        te->SetBranchAddress("fTrackTime", &trtime);
        te->SetBranchAddress("fFlags", &trflag);

        // get run information
        if (tb->GetEntries() == 0) {
          printf("Empty BC tree\n");
          continue;
        }
        tb->GetEntry(0);
        auto runInfo = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo(o2::ccdb::BasicCCDBManager::instance(), nRun);
        //        uint64_t sorTimestamp = runInfo.sor;
        //        uint64_t eorTimestamp = runInfo.eor;
        int64_t bcSOR = runInfo.orbitSOR * nBCsPerOrbit;
        int64_t nBCsPerTF = runInfo.orbitsPerTF * nBCsPerOrbit;
        auto calcBcInROF = [&](ULong64_t bcGlobal) {
          int bc = (bcGlobal + nBCsPerOrbit) % nBCsPerITSROF;
          bc -= offsetITSROF;
          return bc;
        };

        // fill maps for BC and FT0
        int nFT0 = tft->GetEntries();
        std::unordered_map<int, int> bcToFT0;
        bcToFT0.reserve(nFT0);
        for (int jft = 0; jft < nFT0; ++jft) {
          tft->GetEntry(jft);
          bcToFT0[iBCft0] = jft;
        }
        // fill maps for BC and FV0
        int nFV0 = tfv->GetEntries();
        std::unordered_map<int, int> bcToFV0;
        bcToFV0.reserve(nFV0);
        for (int jfv = 0; jfv < nFV0; ++jfv) {
          tfv->GetEntry(jfv);
          bcToFV0[iBCfv0] = jfv;
        }
        // tag good collisions
        int nColl = tc->GetEntries();
        std::vector<int> ft0indices(nColl, -1);
        std::vector<int> fv0indices(nColl, -1);
        std::vector<bool> isGoodColl(nColl, false);
        for (int i = 0; i < nColl; ++i) {
          ft0indices[i] = -1;
          tc->GetEntry(i);
          if (iBC < 0 || iBC >= tb->GetEntries()) {
            printf("ERROR: iBC out of range\n");
            continue;
          }
          tb->GetEntry(iBC);
          auto it = bcToFT0.find(iBC);
          if (it != bcToFT0.end())
            ft0indices[i] = it->second;
          auto iv = bcToFV0.find(iBC);
          if (iv != bcToFV0.end())
            fv0indices[i] = iv->second;
          if (std::abs(zv) > maxZv)
            continue;
          int jft0 = ft0indices[i];
          int jfv0 = fv0indices[i];
          if (jft0 >= 0 && jfv0 >= 0) {
            tft->GetEntry(jft0);
            tfv->GetEntry(jfv0);
            bool isTVX = TESTBIT(trigMask, o2::ft0::Triggers::bitVertex);
            int64_t bcInTF = (static_cast<int64_t>(gloBC) - bcSOR) % nBCsPerTF;
            bool isTFBorderOK = (bcInTF > 300 && bcInTF < nBCsPerTF - 4000);
            if (isTFBorderOK && isTVX &&
                timeft0a > minCollTimeFT0A && timeft0a < maxCollTimeFT0A &&
                timeft0c > minCollTimeFT0C && timeft0c < maxCollTimeFT0C &&
                timefv0a > minCollTimeFV0A && timefv0a < maxCollTimeFV0A) {
              isGoodColl[i] = true;
              int bcInITSROF = calcBcInROF(gloBC);
              hNEventsVsBC->Fill(gloBC % nBCsPerOrbit);
              hNEventsVsBCinROF->Fill(bcInITSROF);
              hCollTime->Fill(ctime);
              hVz->Fill(zv);
            }
          }
        }

        // loop over tracks
        int nTracks = tt->GetEntries();
        for (int i = 0; i < nTracks; ++i) {
          // O2track_iu and O2trackextra_xxx entries are aligned 1-to-1
          tt->GetEntry(i);
          te->GetEntry(i);
          // compute the ITS cluster map, which can be used in track selecion
          int nITSclu = 0;
          uint itsCluMap = 0;
          for (int jLay = 0; jLay < 7; ++jLay) {
            if ((itsclusiz >> (jLay * 4)) & 0xf) {
              nITSclu++;
              itsCluMap |= (1 << jLay);
            }
          }
          // track selections
          if (!(trflag & o2::aod::track::PVContributor))
            continue;
          if (nITSclu == 0 || itschi2 < 0. || tpcchi2 < 0.)
            continue;
          // compute eta from track dip angle lambda via tgl = tan(lambda)
          float cl = 1. / std::sqrt(1. + tgl * tgl);
          float sl = cl * tgl;
          float eta = 0.5 * std::log((1 + sl) / (1 - sl));
          if (std::abs(eta) > maxTrackEta)
            continue;
          // collision selections
          if (iColl >= 0 && iColl < nColl && isGoodColl[iColl]) {
            tc->GetEntry(iColl);
            hTrTime->Fill(trtime);
            hTrVsCollTime->Fill(ctime, trtime);
            tb->GetEntry(iBC);
            int bcInITSROF = calcBcInROF(gloBC);
            hNTracksVsBC->Fill(gloBC % nBCsPerOrbit);
            hCluTrackVsBCinROF->Fill(bcInITSROF, nITSclu);
            hNTracksVsBCinROF->Fill(bcInITSROF);
            hNPVContribVsBCinROF->Fill(bcInITSROF, nPVcontrib);
            hTrNoColl->Fill(1);
          } else {
            hTrNoColl->Fill(0);
          }
        }
      }
    }
  }

  // plotting
  TProfile* pctr = hNPVContribVsBCinROF->ProfileX();
  TProfile* pclu = hCluTrackVsBCinROF->ProfileX();
  TH1F* hFrac7 = new TH1F("hFrac7", ";bcInROF;Fraction of tracks with 7 clusters", hCluTrackVsBCinROF->GetXaxis()->GetNbins(), hCluTrackVsBCinROF->GetXaxis()->GetXmin(), hCluTrackVsBCinROF->GetXaxis()->GetXmax());
  for (int jx = 1; jx <= hCluTrackVsBCinROF->GetXaxis()->GetNbins(); jx++) {
    double tot = 0.;
    for (int jy = 1; jy <= hCluTrackVsBCinROF->GetYaxis()->GetNbins(); jy++) {
      double c = hCluTrackVsBCinROF->GetBinContent(jx, jy);
      tot += c;
    }
    double seven = hCluTrackVsBCinROF->GetBinContent(jx, 8);
    double f = 0.;
    double ef = 0.;
    if (tot > 0) {
      f = seven / tot;
      ef = std::sqrt(f * (1 - f) / tot);
    }
    hFrac7->SetBinContent(jx, f);
    hFrac7->SetBinError(jx, ef);
  }
  for (int jx = 1; jx <= hNTracksVsBC->GetXaxis()->GetNbins(); jx++) {
    double nt = hNTracksVsBC->GetBinContent(jx);
    double ne = hNEventsVsBC->GetBinContent(jx);
    if (ne > 0)
      hNTracksPerEventVsBC->SetBinContent(jx, nt / ne);
  }
  for (int jx = 1; jx <= hNTracksVsBCinROF->GetXaxis()->GetNbins(); jx++) {
    double nt = hNTracksVsBCinROF->GetBinContent(jx);
    double ne = hNEventsVsBCinROF->GetBinContent(jx);
    if (ne > 0)
      hNTracksPerEventVsBCinROF->SetBinContent(jx, nt / ne);
  }

  hCluTrackVsBCinROF->SetStats(0);
  hFrac7->SetStats(0);
  hNPVContribVsBCinROF->SetStats(0);

  TCanvas* ctr = new TCanvas("ctr", "", 1400, 500);
  ctr->Divide(3, 1);
  ctr->cd(1);
  hTrNoColl->Draw();
  ctr->cd(2);
  hCollTime->Draw();
  ctr->cd(3);
  hVz->Draw();

  TCanvas* cbc = new TCanvas("cbc", "", 1400, 800);
  cbc->Divide(1, 3);
  cbc->cd(1);
  hNEventsVsBC->Draw();
  cbc->cd(2);
  hNTracksVsBC->Draw();
  cbc->cd(3);
  hNTracksPerEventVsBC->Draw();

  TCanvas* crof = new TCanvas("crof", "", 1400, 800);
  crof->Divide(3, 2);
  crof->cd(1);
  hNEventsVsBCinROF->Draw();
  crof->cd(2);
  hNTracksVsBCinROF->Draw();
  crof->cd(3);
  hNTracksPerEventVsBCinROF->Draw();
  crof->cd(4);
  gPad->SetLogz();
  hCluTrackVsBCinROF->Draw("colz");
  pclu->SetLineWidth(2);
  pclu->Draw("same");
  crof->cd(5);
  hFrac7->SetMinimum(0.);
  hFrac7->SetMaximum(1.);
  hFrac7->SetLineWidth(2);
  hFrac7->Draw();
  crof->cd(6);
  gPad->SetLogz();
  hNPVContribVsBCinROF->Draw("colz");
  pctr->SetLineWidth(2);
  pctr->Draw("same");
  crof->SaveAs("ITSTracksVsBCinROF.png");
}
