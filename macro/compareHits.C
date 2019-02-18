#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "ITSMFTSimulation/Hit.h"
#include "TOFSimulation/Detector.h"
#include "EMCALBase/Hit.h"
#include "TRDSimulation/Detector.h"
#endif

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

template <typename T>
struct HitStats {
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

struct ITSHitStats {
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

  // adds a hit to the statistics
  void addHit(o2::ITSMFT::Hit const& hit)
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
}; // end struct

// do comparison for ITS
void checkITS(TTree* reftree, TTree* testtree)
{
  auto refresult = analyse<o2::ITSMFT::Hit, ITSHitStats>(reftree, "ITSHit");
  refresult.print();
  auto testresult = analyse<o2::ITSMFT::Hit, ITSHitStats>(testtree, "ITSHit");
  testresult.print();
}

// do comparison for TOF
void checkTOF(TTree* reftree, TTree* testtree)
{
  auto refresult = analyse<o2::tof::HitType, HitStats<o2::tof::HitType>>(reftree, "TOFHit");
  refresult.print();
  auto testresult = analyse<o2::tof::HitType, HitStats<o2::tof::HitType>>(testtree, "TOFHit");
  testresult.print();
}

// do comparison for EMC
void checkEMC(TTree* reftree, TTree* testtree)
{
  auto refresult = analyse<o2::EMCAL::Hit, HitStats<o2::EMCAL::Hit>>(reftree, "EMCHit");
  refresult.print();
  auto testresult = analyse<o2::EMCAL::Hit, HitStats<o2::EMCAL::Hit>>(testtree, "EMCHit");
  testresult.print();
}

// do comparison for TRD
void checkTRD(TTree* reftree, TTree* testtree)
{
  auto refresult = analyse<o2::trd::HitType, HitStats<o2::trd::HitType>>(reftree, "TRDHit");
  refresult.print();
  auto testresult = analyse<o2::trd::HitType, HitStats<o2::trd::HitType>>(testtree, "TRDHit");
  testresult.print();
}

// Simple macro to compare properties of simulated hits
// of two runs (for example reference run and test run).
// Used for instance to validate MC optimizations or to study
// the effect on the physics at the lowest level.
void compareHits(const char* reffilename = "o2sim.root", const char* testfilename = "o2sim.root")
{
  TFile rf(reffilename, "OPEN");
  TFile tf(testfilename, "OPEN");
  auto reftree = (TTree*)rf.Get("o2sim");
  auto testtree = (TTree*)tf.Get("o2sim");

  checkITS(reftree, testtree);
  checkTOF(reftree, testtree);
  checkEMC(reftree, testtree);
  checkTRD(reftree, testtree);
}
