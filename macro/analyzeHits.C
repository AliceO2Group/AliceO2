#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "ITSMFTSimulation/Hit.h"
#include "TOFSimulation/Detector.h"
#include "EMCALBase/Hit.h"
#include "DataFormatsTRD/Hit.h"
#include "FT0Simulation/Detector.h" // for Fit Hit
#include "DataFormatsFV0/Hit.h"
#include "HMPIDBase/Hit.h"
#include "TPCSimulation/Point.h"
#include "PHOSBase/Hit.h"
#include "DataFormatsFDD/Hit.h"
#include "MCHSimulation/Hit.h"
#include "MIDSimulation/Hit.h"
#include "CPVBase/Hit.h"
#include "DataFormatsZDC/Hit.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#endif

TString gPrefix("");

template <typename Hit, typename Accumulator>
Accumulator analyse(TTree* tr, const char* brname)
{
  Accumulator prop;
  auto br = tr->GetBranch(brname);
  if (!br) {
    return prop;
  }
  auto entries = br->GetEntries();
  std::vector<Hit>* hitvector = nullptr;
  br->SetAddress(&hitvector);

  for (int i = 0; i < entries; ++i) {
    br->GetEntry(i);
    for (auto& hit : *hitvector) {
      prop.addHit(hit);
    }
  }
  prop.normalize();
  return prop;
};

struct HitStatsBase {
  int NHits = 0;
  double XAvg = 0.; // avg 1st moment
  double YAvg = 0.;
  double ZAvg = 0.;
  double X2Avg = 0.; // avg 2nd moment
  double Y2Avg = 0.;
  double Z2Avg = 0.;
  double EAvg = 0.; // average total energy
  double E2Avg = 0.;
  double TAvg = 0.;  // average T
  double T2Avg = 0.; // average T^2

  void print() const
  {
    std::cout << NHits << " "
              << XAvg << " "
              << YAvg << " "
              << ZAvg << " "
              << X2Avg << " "
              << Y2Avg << " "
              << Z2Avg << " "
              << EAvg << " "
              << E2Avg << " "
              << TAvg << " "
              << T2Avg << "\n";
  }

  void normalize()
  {
    XAvg /= NHits;
    YAvg /= NHits;
    ZAvg /= NHits;
    X2Avg /= NHits;
    Y2Avg /= NHits;
    Z2Avg /= NHits;
    EAvg /= NHits;
    E2Avg /= NHits;
    TAvg /= NHits;
    T2Avg /= NHits;
  }
};

template <typename T>
struct HitStats : public HitStatsBase {
  // adds a hit to the statistics
  void addHit(T const& hit)
  {
    NHits++;
    auto x = hit.GetX();
    XAvg += x;
    X2Avg += x * x;
    auto y = hit.GetY();
    YAvg += y;
    Y2Avg += y * y;
    auto z = hit.GetZ();
    ZAvg += z;
    Z2Avg += z * z;
    auto e = hit.GetEnergyLoss();
    EAvg += e;
    E2Avg += e * e;
    auto t = hit.GetTime();
    TAvg += t;
    T2Avg += t * t;
  }
};

struct TRDHitStats : public HitStatsBase {
  // adds a hit to the statistics
  void addHit(o2::trd::Hit const& hit)
  {
    NHits++;
    auto x = hit.GetX();
    XAvg += x;
    X2Avg += x * x;
    auto y = hit.GetY();
    YAvg += y;
    Y2Avg += y * y;
    auto z = hit.GetZ();
    ZAvg += z;
    Z2Avg += z * z;
    auto e = hit.GetCharge();
    EAvg += e;
    E2Avg += e * e;
    auto t = hit.GetTime();
    TAvg += t;
    T2Avg += t * t;
  }
};

struct ITSHitStats : public HitStatsBase {
  // adds a hit to the statistics
  void addHit(o2::itsmft::Hit const& hit)
  {
    NHits++;
    auto x = hit.GetStartX();
    XAvg += x;
    X2Avg += x * x;
    auto y = hit.GetStartY();
    YAvg += y;
    Y2Avg += y * y;
    auto z = hit.GetStartZ();
    ZAvg += z;
    Z2Avg += z * z;
    auto e = hit.GetTotalEnergy();
    EAvg += e;
    E2Avg += e * e;
    auto t = hit.GetTime();
    TAvg += t;
    T2Avg += t * t;
  }
}; // end struct

struct TPCHitStats : public HitStatsBase {
  // adds a hit to the statistics
  void addHit(o2::tpc::HitGroup const& hitgroup)
  {
    for (int i = 0; i < hitgroup.getSize(); ++i) {
      auto hit = hitgroup.getHit(i);
      NHits++;
      auto x = hit.GetX();
      XAvg += x;
      X2Avg += x * x;
      auto y = hit.GetY();
      YAvg += y;
      Y2Avg += y * y;
      auto z = hit.GetZ();
      ZAvg += z;
      Z2Avg += z * z;
      auto e = hit.GetEnergyLoss();
      EAvg += e;
      E2Avg += e * e;
      auto t = hit.GetTime();
      TAvg += t;
      T2Avg += t * t;
    }
  }
}; // end struct

// need a special version for TPC since loop over sectors
TPCHitStats analyseTPC(TTree* tr)
{
  TPCHitStats prop;
  for (int sector = 0; sector < 35; ++sector) {
    std::stringstream brnamestr;
    brnamestr << "TPCHitsShiftedSector" << sector;
    auto br = tr->GetBranch(brnamestr.str().c_str());
    if (!br) {
      return prop;
    }
    auto entries = br->GetEntries();
    std::vector<o2::tpc::HitGroup>* hitvector = nullptr;
    br->SetAddress(&hitvector);

    for (int i = 0; i < entries; ++i) {
      br->GetEntry(i);
      for (auto& hit : *hitvector) {
        prop.addHit(hit);
      }
    }
  }
  prop.normalize();
  return prop;
};

// do comparison for ITS
void analyzeITS(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::itsmft::Hit, ITSHitStats>(reftree, "ITSHit");
  std::cout << gPrefix << " ITS ";
  refresult.print();
}

// do comparison for TOF
void analyzeTOF(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::tof::HitType, HitStats<o2::tof::HitType>>(reftree, "TOFHit");
  std::cout << gPrefix << " TOF ";
  refresult.print();
}

// do comparison for EMC
void analyzeEMC(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::emcal::Hit, HitStats<o2::emcal::Hit>>(reftree, "EMCHit");
  std::cout << gPrefix << " EMC ";
  refresult.print();
}

// do comparison for TRD
// need a different version of TRDHitStats to retrieve the hit value
void analyzeTRD(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::trd::Hit, TRDHitStats>(reftree, "TRDHit");
  std::cout << gPrefix << " TRD ";
  refresult.print();
}

// do comparison for PHS
void analyzePHS(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::phos::Hit, HitStats<o2::phos::Hit>>(reftree, "PHSHit");
  std::cout << gPrefix << " PHS ";
  refresult.print();
}

void analyzeFT0(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::ft0::HitType, HitStats<o2::ft0::HitType>>(reftree, "FT0Hit");
  std::cout << gPrefix << " FT0 ";
  refresult.print();
}

void analyzeHMP(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::hmpid::HitType, HitStats<o2::hmpid::HitType>>(reftree, "HMPHit");
  std::cout << gPrefix << " HMP ";
  refresult.print();
}

void analyzeMFT(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::itsmft::Hit, ITSHitStats>(reftree, "MFTHit");
  std::cout << gPrefix << " MFT ";
  refresult.print();
}

void analyzeFDD(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::fdd::Hit, HitStats<o2::fdd::Hit>>(reftree, "FDDHit");
  std::cout << gPrefix << " FDD ";
  refresult.print();
}

void analyzeFV0(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::fv0::Hit, HitStats<o2::fv0::Hit>>(reftree, "FV0Hit");
  std::cout << gPrefix << " FV0 ";
  refresult.print();
}

void analyzeMCH(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::mch::Hit, HitStats<o2::mch::Hit>>(reftree, "MCHHit");
  std::cout << gPrefix << " MCH ";
  refresult.print();
}

void analyzeMID(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::mid::Hit, HitStats<o2::mid::Hit>>(reftree, "MIDHit");
  std::cout << gPrefix << " MID ";
  refresult.print();
}

void analyzeCPV(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::cpv::Hit, HitStats<o2::cpv::Hit>>(reftree, "CPVHit");
  std::cout << gPrefix << " CPV ";
  refresult.print();
}

void analyzeZDC(TTree* reftree)
{
  if (!reftree)
    return;
  auto refresult = analyse<o2::zdc::Hit, HitStats<o2::zdc::Hit>>(reftree, "ZDCHit");
  std::cout << gPrefix << " ZDC ";
  refresult.print();
}

void analyzeTPC(TTree* reftree)
{
  if (!reftree)
    return;
  // need to loop over sectors
  auto refresult = analyseTPC(reftree);
  std::cout << gPrefix << " TPC ";
  refresult.print();
}

TTree* getHitTree(o2::parameters::GRPObject const* grp, const char* filebase, o2::detectors::DetID detid)
{
  if (!grp->isDetReadOut(detid)) {
    return nullptr;
  }

  std::string filename(o2::base::NameConf::getHitsFileName(detid, filebase).c_str());

  // shamefully leaking memory as the TTree cannot live without the file...
  TFile* file = new TFile(filename.c_str(), "OPEN");
  auto t = (TTree*)file->Get("o2sim");
  return t;
}

// Simple macro to get basic mean properties of simulated hits
// Used for instance to validate MC optimizations or to study
// the effect on the physics at the lowest level.
//
// A prefix (such as a parameter) can be given which will be  prepended before each line of printout.
// This could be useful for plotting.
//
void analyzeHits(const char* filebase = "o2sim", const char* prefix = "")
{
  gPrefix = prefix;

  // READ GRP AND ITERATE OVER DETECTED PARTS
  auto grp = o2::parameters::GRPObject::loadFrom(filebase);

  // should correspond to the same number as defined in DetID
  analyzeITS(getHitTree(grp, filebase, o2::detectors::DetID::ITS));
  analyzeTPC(getHitTree(grp, filebase, o2::detectors::DetID::TPC));
  analyzeMFT(getHitTree(grp, filebase, o2::detectors::DetID::MFT));
  analyzeTOF(getHitTree(grp, filebase, o2::detectors::DetID::TOF));
  analyzeEMC(getHitTree(grp, filebase, o2::detectors::DetID::EMC));
  analyzeTRD(getHitTree(grp, filebase, o2::detectors::DetID::TRD));
  analyzePHS(getHitTree(grp, filebase, o2::detectors::DetID::PHS));
  analyzeCPV(getHitTree(grp, filebase, o2::detectors::DetID::CPV));
  analyzeFT0(getHitTree(grp, filebase, o2::detectors::DetID::FT0));
  analyzeFV0(getHitTree(grp, filebase, o2::detectors::DetID::FV0));
  analyzeFDD(getHitTree(grp, filebase, o2::detectors::DetID::FDD));
  analyzeHMP(getHitTree(grp, filebase, o2::detectors::DetID::HMP));
  analyzeMCH(getHitTree(grp, filebase, o2::detectors::DetID::MCH));
  analyzeMID(getHitTree(grp, filebase, o2::detectors::DetID::MID));
  analyzeZDC(getHitTree(grp, filebase, o2::detectors::DetID::ZDC));
  // analyzeACO(getHitTree(grp, filebase, o2::detectors::DetID::ACO));
}
