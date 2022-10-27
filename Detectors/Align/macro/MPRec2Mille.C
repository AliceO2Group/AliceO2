#if !defined(__CINT__) || defined(__MAKECINT__)
#include "Align/Millepede2Record.h"
#include "Align/Mille.h"
#include <TFile.h>
#include <TChain.h>
#include <TString.h>
#include <TSystem.h>
#include <TClassTable.h>
#include <TMath.h>
#endif

using namespace o2::align;

// convert MPRecord to Mille format
const char* recBranchName = "mprec";
const char* recTreeName = "mpTree";
const char* defOutName = "mpData";
const int defSplit = -200; // default output chunk size in MB

TChain* loadMPrecChain(const char* inpData, const char* chName = recTreeName);
int convertAndStore(Millepede2Record* rec, Mille* mille);
bool processMPRec(Millepede2Record* rec);
std::vector<float> buffLoc;

void MPRec2Mille(const char* inpName,              // name of MPRecord file or list of files
                 const char* outname = defOutName, // out file name
                 int split = defSplit              // 0: no split, >0: on N tracks,<0: on size in MB
)
{
  TChain* mprChain = loadMPrecChain(inpName);
  if (!mprChain)
    return;
  int nEnt = mprChain->GetEntries();
  //
  TBranch* br = mprChain->GetBranch(recBranchName);
  if (!br) {
    printf("provided tree does not contain branch mprec\n");
    return;
  }
  //
  Millepede2Record mp, *mprec = &mp;
  br->SetAddress(&mprec);
  //
  TString mln = outname;
  if (mln.IsNull()) {
    mln = inpName; // use inpname + ".mille"
  }
  if (mln.EndsWith(".mille") > 0) {
    mln.Resize(mln.Last('.'));
  }
  printf(">>%s \n<<%s%s%s\n", inpName, mln.Data(), split ? "_XXX" : "", ".mille");
  if (split) {
    printf("Split on %d %s\n", TMath::Abs(split), split > 0 ? "tracks" : "MB");
  }
  //
  TString milleName;
  std::unique_ptr<Mille> mille;
  int cntTr = 0, cntTot = 0, cntMille = 0;
  double sizeW = 0., sizeWTot = 0.;
  if (split < 0) {
    split *= 1000000;
  }
  for (int i = 0; i < nEnt; i++) {
    mprChain->GetEntry(i);
    //
    if (!processMPRec(mprec))
      continue; // preprocess and skip if needed
    //
    if (!mille || (split > 0 && ++cntTr > split) || (split < 0 && sizeW > -split)) { // start new mille file
      cntTr = sizeW = 0;
      milleName = split ? Form("%s_%03d.%s", mln.Data(), cntMille, "mille") : Form("%s.%s", mln.Data(), "mille");
      cntMille++;
      printf("Opening output file %s\n", milleName.Data());
      mille = std::make_unique<Mille>(milleName.Data());
    }
    cntTot++;
    int nbwr = convertAndStore(mprec, mille.get());
    sizeW += nbwr;
    sizeWTot += nbwr;
  }
  mille.reset();
  br->SetAddress(0);
  delete mprChain;
  //
  printf("converted %d tracks out of %d\n(%ld MB in %d %s%s.mille files written)\n",
         cntTot, nEnt, long(sizeWTot / 1e6), cntMille, mln.Data(), split ? "_XXX" : "");
  //
}

//_________________________________________________________
int convertAndStore(Millepede2Record* rec, Mille* mille)
{
  // convert and store the record
  int nr = rec->getNResid(); // number of residual records
  int nloc = rec->getNVarLoc();
  const float* recDGlo = rec->getArrGlo();
  const float* recDLoc = rec->getArrLoc();
  const short* recLabLoc = rec->getArrLabLoc();
  const int* recLabGlo = rec->getArrLabGlo();
  //
  for (int ir = 0; ir < nr; ir++) {
    buffLoc.clear();
    buffLoc.resize(nloc);
    int ndglo = rec->getNDGlo(ir);
    int ndloc = rec->getNDLoc(ir);
    // fill 0-suppressed array from MPRecord to non-0-suppressed array of Mille
    for (int l = ndloc; l--;) {
      buffLoc[recLabLoc[l]] = recDLoc[l];
    }
    //
    mille->mille(nloc, buffLoc.data(), ndglo, recDGlo, recLabGlo, rec->getResid(ir), rec->getResErr(ir));
    //
    recLabGlo += ndglo; // next record
    recDGlo += ndglo;
    recLabLoc += ndloc;
    recDLoc += ndloc;
  }
  return mille->end(); // bytes written
  //
}

//____________________________________________________________________
TChain* loadMPrecChain(const char* inpData, const char* chName)
{
  TChain* chain = new TChain(chName);
  //
  TString inpDtStr = inpData;
  if (inpDtStr.EndsWith(".root")) {
    chain->AddFile(inpData);
  } else {
    //
    ifstream inpf(inpData);
    if (!inpf.good()) {
      printf("Failed on input filename %s\n", inpData);
      return 0;
    }
    //
    TString flName;
    flName.ReadLine(inpf);
    while (!flName.IsNull()) {
      flName = flName.Strip(TString::kBoth, ' ');
      if (flName.BeginsWith("//") || flName.BeginsWith("#")) {
        flName.ReadLine(inpf);
        continue;
      }
      flName = flName.Strip(TString::kBoth, ',');
      flName = flName.Strip(TString::kBoth, '"');
      printf("Adding %s\n", flName.Data());
      chain->AddFile(flName.Data());
      flName.ReadLine(inpf);
    }
  }
  //
  int n = chain->GetEntries();
  if (n < 1) {
    printf("Obtained chain is empty\n");
    return 0;
  } else
    printf("Opened %s chain with %d entries\n", chName, n);
  return chain;
}

//_________________________________________________________
bool processMPRec(Millepede2Record* rec)
{
  // put here user code
  //
  return true;
}
