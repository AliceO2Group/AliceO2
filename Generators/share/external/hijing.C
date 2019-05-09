// configures a AliGenHijing class from AliRoot
//   usage: o2sim -g extgen --extGenFile hijing.C
// options:                 --extGenFunc hijing(5020., 0., 20.)

/// \author R+Preghenella - October 2018

R__LOAD_LIBRARY(libTHijing)

FairGenerator*
  hijing(double energy = 5020., double bMin = 0., double bMax = 20.)
{
  // instance and configure Hijing
  auto hij = new AliGenHijing(-1);
  hij->SetEnergyCMS(energy);
  hij->SetImpactParameterRange(bMin, bMax);
  hij->SetReferenceFrame("CMS");
  hij->SetProjectile("A", 208, 82);
  hij->SetTarget("A", 208, 82);
  hij->SetSpectators(0);
  hij->KeepFullEvent();
  hij->SetJetQuenching(0);
  hij->SetShadowing(1);
  hij->SetDecaysOff(1);
  hij->SetSelectAll(0);
  hij->SetPtHardMin(2.3);
  hij->Init();

  // instance and configure TGenerator interface
  auto tgen = new o2::eventgen::GeneratorTGenerator();
  tgen->setMomentumUnit(1.);        // [GeV/c]
  tgen->setEnergyUnit(1.);          // [GeV/c]
  tgen->setPositionUnit(0.1);       // [cm]
  tgen->setTimeUnit(3.3356410e-12); // [s]
  tgen->setTGenerator(hij);
  return tgen;
}
