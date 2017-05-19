/// \file testContVec.cxx
/// \brief This task tests some features of the ContVec class (vector-like container for copyable objects)
/// \author Ruben Shahoyan, ruben.shahoyan@cern.ch

//#define RUN_CONTVEC_TEST_AS_ROOT_MACRO // uncomment this to run as standalone root macro, e.g. root -b -q testContVec.cxx+

#ifndef RUN_CONTVEC_TEST_AS_ROOT_MACRO
#define BOOST_TEST_MODULE Test ContVec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#endif

#include "DetectorsBase/ContVec.h"
#include "DetectorsBase/Track.h"
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

void  writeToBinFile(const char* ptr, int nbytes, const char* fname="containerTest.bin");
char* readFromBinFile(int& nread, const char* fname="containerTest.bin");

using namespace o2::Base::Track;
using namespace o2::Base;
using namespace std;

using myCont_t = ContVec<o2::Base::Track::TrackPar,int>;

bool compareTracks(const TrackPar* tr0, const TrackPar* tr1);
bool compareContainers(const myCont_t& cont0, const myCont_t& cont1, string msg);
bool testContVec(bool cleanTmp=true);

#ifndef RUN_CONTVEC_TEST_AS_ROOT_MACRO
BOOST_AUTO_TEST_CASE(ContVec_test)
{
  BOOST_CHECK(testContVec());
}
#endif

bool testContVec(bool cleanTmp)
{
  array<float,5> arp={0.,0.,0.1,0.1,1};

  // container for objects of type TrackPar
  // and user info object type int
  myCont_t cnt;
  bool res = false;
  
  // set user info (data identifier etc.)
  cnt.setUserInfo(0xdeadbeaf);
  
  float b = 5.0f; // B-field

  // here we create some track in the container
  for (auto i=0;i<10;i++) {
    //
    TrackPar tr0(0.f,0.f,arp);
    tr0.PropagateParamTo(float(1+(i*10)%100),b);
    cnt.push_back(tr0);                           // track copied into container
    //
    TrackPar* tr1 = cnt.emplace_back(0.,1., arp); // or created in the array directly
    tr1->PropagateParamTo(float(5+(i*10)%100),b); 
    //
  }

  // store container in a root file as single object
  string outRoot={"contobj.root"};

  TFile* flout = TFile::Open(outRoot.data(),"recreate");
  cnt.Write("cntw");
  flout->Close();
  delete flout;
  //
  // read back
  TFile* flin = TFile::Open(outRoot.data());
  auto cntroot = reinterpret_cast<myCont_t*>(flin->Get("cntw"));
  flin->Close();
  delete flin;
  //
  //***********************//
  //***********************//
  //***********************//
  res = compareContainers(cnt, *cntroot,"copy read from root file"); // compare with original
  if (cleanTmp) gSystem->Unlink(outRoot.data());
  if (!res) return res;
  
  //======================================================================
  // Test transfer of raw pointers, makes sense only for plain data objects
  //
  int nb = 0;
  char* pntr = 0;
  string outBin={"cont.bin"};
  // Write to bin file the raw pointer preserving the original object
  writeToBinFile(cnt.getPtr(),cnt.sizeInBytes(),outBin.data());
  //
  // Read: recreate containers from raw pointer
  pntr = readFromBinFile(nb,outBin.data());
  //  
  // "false" indicates that the pntr is currently not managed, so that the new container does not need
  // to create new buffer but just take ownership of the buffer pointed by pntr
  // 
  // nb is passed just for consistency check
  myCont_t cntb0(pntr,false,nb); 

  //***********************//
  //***********************//
  //***********************//
  res = compareContainers(cnt, cntb0,"using external raw pointer");  // compare with original
  if (cleanTmp) gSystem->Unlink(outBin.data());
  if (!res) return res;

  
  // Write to bin file the raw pointer resetting the original object
  nb = cntroot->sizeInBytes();
  unique_ptr<char[]> tmpPtr( cntroot->release() ); // this will reset cntroot
  writeToBinFile(tmpPtr.get(),nb, outBin.data());
  delete[] tmpPtr.release(); // just to dispose unnecessary buffer
  //
  // Read: recreate containers from raw pointer
  tmpPtr.reset( readFromBinFile(nb,outBin.data()) );
  //  
  // "true" indicates that the raw pointer is currently managed, so that the new container must 
  // create new buffer and copy the content
  // -1 (default) indicates no request for buffer size consistency check
  myCont_t cntb1(tmpPtr.get(),true,-1);
  delete[] tmpPtr.release(); // just to dispose unnecessary buffer

  //***********************//
  //***********************//
  //***********************//
  res = compareContainers(cnt, cntb1,"copy from raw pointer");  // compare with original
  if (cleanTmp) gSystem->Unlink(outBin.data());
  if (!res) return res;
  
  
  //======================================================================
  // Test output to TTree
  //
  // For debug purposes (also until the messaging is fully operational) we need
  // to be able to store objects as vectors in the root tree 
  //
  string outTree={"contTree.root"};
  
  TFile* fltree = TFile::Open(outTree.data(),"recreate");
  TTree* tree = new TTree("tstTree","testTree");
  for (auto j=0;j<cnt.size();j++) {
    auto trc = cnt[j];
    trc->PropagateParamTo( trc->GetX()+j%5 ,b);
  }
  for (auto i=0;i<5;i++) {
    // modifiy slightly the tracks to simulate new event
    cnt.AddToTree(tree,"Tracks");
    tree->Fill();
    //
  }
  tree->Write();
  delete tree;
  fltree->Close();
  delete fltree;
  //
  /// read back the tree
  //
  fltree = TFile::Open(outTree.data());
  tree = reinterpret_cast<TTree*>( fltree->Get("tstTree") );
  int nent = tree->GetEntries();
  vector<o2::Base::Track::TrackPar> *vecIn = 0;
  tree->SetBranchAddress("Tracks",&vecIn);
  tree->GetEntry(nent-1);
  cntb1.clear();
  for (size_t i=0;i<vecIn->size();i++) {
    cntb1.push_back( (*vecIn)[i] );
  }
  delete tree;
  fltree->Close();
  delete fltree;
  //
  //***********************//
  //***********************//
  //***********************//
  res = compareContainers(cnt, cntb1,"recreated from container->tree->container");  // compare with original
  if (cleanTmp) gSystem->Unlink(outTree.data());

  return res;  
}


//____________________________________________
void  writeToBinFile(const char* ptr, int nbytes, const char* fname)
{
  // write char buffer to binary file
  ofstream file(fname, ios::out|ios::binary|ios::trunc);
  file.write(ptr,nbytes);
  file.close();
}

//____________________________________________
char* readFromBinFile(int &nread, const char* fname)
{
  // read char buffer from binary file
  nread = 0;
  ifstream file(fname, ios::in|ios::binary|ios::ate);
  nread = (int)file.tellg();
  unique_ptr<char[]> buff(new char[nread]);
  file.seekg (0, ios::beg);
  file.read(buff.get(),nread);
  file.close();
  return buff.release();
}

//____________________________________________
bool compareContainers(const myCont_t& cont0, const myCont_t& cont1, string msg)
{
  // compare the 2 containers
  if (cont0.size()!=cont1.size()) {
    cerr << "Size difference: original container vs " << msg << endl;
    return false;
  }
  for (auto i=0;i<cont0.size();i++) {
    if (!compareTracks(cont0[i],cont1[i])) {
      cerr << "Track " << i << " difference: original container vs " << msg << endl;
      cont0[i]->Print();
      cont1[i]->Print();      
      return false;     
    }
  }
  return true;
}

//____________________________________________
bool compareTracks(const TrackPar* tr0, const TrackPar* tr1)
{
  // compare parameters of 2 tracks
  if ( tr0->GetX()     != tr1->GetX() ||
       tr0->GetAlpha() != tr1->GetAlpha() ||
       tr0->GetY()     != tr1->GetY() ||
       tr0->GetZ()     != tr1->GetZ() )
    {
      cout << "Sizes of original container and its copy from root file are different" << endl;
      return false;
    }
  return true;
}
