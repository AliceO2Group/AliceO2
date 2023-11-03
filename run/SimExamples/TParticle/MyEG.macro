#include <TGenerator.h>
#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TRandom.h>
#include <TParticle.h>

//--------------------------------------------------------------------
// Our generator class.  Really simple.
class MyGenerator : public TGenerator
{
 public:
  Long_t projectilePDG;
  Long_t targetPDG;
  Double_t sqrts;
  MyGenerator() {}
  void Initialize(Long_t projectile,
                  Long_t target,
                  Double_t sqrts)
  {
    this->projectilePDG = projectile;
    this->targetPDG = target;
    this->sqrts = sqrts;
  }
  void GenerateEvent()
  { /* Do something */
  }
  TObjArray* ImportParticles(Option_t* option = "") { return 0; }
  Int_t ImportParticles(TClonesArray* particles, Option_t* option = "")
  {
    Int_t nParticles = 10;
    Int_t iParticle = 0;
    // Make beam particles
    new ((*particles)[iParticle++]) TParticle(projectilePDG, 4, -1, -1,
                                              2, nParticles - 1,
                                              0, 0, sqrts / 2,
                                              TMath::Sqrt(1 + sqrts * sqrts),
                                              0, 0, 0, 0);
    new ((*particles)[iParticle++]) TParticle(projectilePDG, 4, -1, -1,
                                              2, nParticles - 1,
                                              0, 0, -sqrts / 2,
                                              TMath::Sqrt(1 + sqrts * sqrts),
                                              0, 0, 0, 0);
    for (; iParticle < nParticles; iParticle++)
      new ((*particles)[iParticle])
        TParticle(211, 1, 0, 1,
                  -1, -1,
                  0.1 * iParticle,
                  0.1 * iParticle,
                  0.1 * iParticle,
                  TMath::Sqrt(0.03 * iParticle * iParticle + 0.14 * 0.14),
                  0, 0, 0, 0);

    return nParticles;
  }
};
//--------------------------------------------------------------------
// Our steering class
struct MySteer {
  TGenerator* generator;
  TFile* file;
  TTree* tree;
  TClonesArray* particles;
  Int_t every;
  MySteer(TGenerator* generator, const TString& output, Int_t every)
    : generator(generator),
      file(TFile::Open(output, "RECREATE")),
      tree(new TTree("T", "T")),
      particles(new TClonesArray("TParticle")),
      every(every)
  {
    tree->SetDirectory(file);
    tree->Branch("Particles", &particles);
  }
  ~MySteer()
  {
    close();
  }
  void event()
  {
    particles->Clear();
    generator->GenerateEvent();
    generator->ImportParticles(particles);
    tree->Fill();
  }
  void sync()
  {
    // Important so that GeneratorTParticle picks up the events as
    // they come.
    tree->AutoSave("SaveSelf FlushBaskets Overwrite");
  }
  void run(Int_t nev)
  {
    for (Int_t iev = 0; iev < nev; iev++) {
      event();

      if (every > 0 and (iev % every == 0) and iev != 0)
        sync();
    }
  }
  void close()
  {
    if (not file)
      return;
    file->Write();
    file->Close();
    file = nullptr;
  }
};

//--------------------------------------------------------------------
// Our steering function
void MyEG(Int_t nev, const TString& out, Int_t seed, Int_t every = 1)
{
  gRandom->SetSeed(seed);

  MyGenerator* eg = new MyGenerator();
  eg->Initialize(2212, 2212, 5200);

  MySteer steer(eg, out, every);
  steer.run(nev);
}
// Local Variables:
//  mode: C++
// End:
//
// EOF
//
