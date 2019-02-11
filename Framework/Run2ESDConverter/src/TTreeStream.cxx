/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

//
//  marian.ivanov@cern.ch
//
//  ------------------------------------------------------------------------------------------------
//  TTreeStream
//  Standard stream (cout) like input for the tree
//  Run and see TTreeStreamer::Test() - to see TTreeStreamer functionality
//  ------------------------------------------------------------------------------------------------  
//
//  -------------------------------------------------------------------------------------------------
//  TTreeSRedirector
//  Redirect file to  different TTreeStreams  
//  Run and see   TTreeSRedirector::Test() as an example of TTreeSRedirector functionality
// 

#include <TClass.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TObjArray.h>
#include <TTree.h>
#include "TTreeStream.h"
// includes for test procedures
#include "TVectorD.h"
#include "TRandom.h"
#include "TLeaf.h"

ClassImp(TTreeDataElement)
ClassImp(TTreeStream)
ClassImp(TTreeSRedirector)



void TTreeStream::Test()
{
  //
  // 
  TFile *ftest = new TFile("teststreamer.root","recreate");
  if (!ftest) ftest = new TFile("teststreamer.root","new");
  //
  //create to streems Tree1 and Tree2
  TTreeStream stream1("Tree1");
  TTreeStream stream2("Tree2");
  //
  Char_t ch='s';
  Float_t f=3.;
  Float_t f2=1;
  TObject *po  = new TObject;
  TObject *po2 = new TObject;
  for (Int_t i=0;i<100000;i++) {
    f=i*100;
    po->SetUniqueID(i);
    po2->SetUniqueID(i*100);
    ch=i%120;
    //
    //    Stream the data
    //    The data layout of stream is defined during first invocation of streamer.
    //    Endl is the trigger which define the end of structure.
    // 
    //    The name of branch can be specified using strings with = at the the end
    //    if string is not specified automatic convention is u (sed B0, B1, ...Bn)
    stream1<<"i="<<i<<"ch="<<ch<<"f="<<f<<"po="<<po<<"\n";
    f  = 1./(100.1+i);
    f2 = -f;     
    //3.) just another example - we can fill the same tree with different objects
    //
    stream2<<f<<po<<"\n";
    stream2<<f2<<po2<<"\n";
  }
  //
  //4.) Close the streeamers (Write the streamed tree's to the file) and close the corresponding file.
  //
  stream1.Close();
  stream2.Close();
  ftest->Close();
  delete ftest;
  //
  //5.) and now see results  in file tteststreamer.root
}

void TTreeSRedirector::Test2()
{
  //
  //Example test function to show functionality of TTreeSRedirector
  //
  //
  //1.)create the  redirector associated with file (testredirector.root)
  //
  //
  TFile* file = new TFile("test.root","recreate");
  TTreeSRedirector *pmistream= new TTreeSRedirector();
  TTreeSRedirector &mistream = *pmistream;
  Char_t ch='s';
  Float_t f=3.;
  Float_t f2=1;
  TObject *po  = new TObject;
  TObject *po2 = new TObject;
  for (Int_t i=0;i<100000;i++) {
    f=i*100;
    po->SetUniqueID(i);
    po2->SetUniqueID(i*100);
    ch=i%120;
    //
    //2.) create the tree with identifier specified by first argument
    //                                layout specified by sequence of arguments
    //                                Tree identifier has to be specified as first argument !!! 
    //    if the tree and layout was already defined the consistency if layout is checked
    //                                if the data are consisten fill given tree 
    //    the name of branch can be specified using strings with = at the the end
    //    if string is not specified use automatic convention  B0, B1, ...Bn
    mistream<<"TreeIdentifier"<<"i="<<i<<"ch="<<ch<<"f="<<f<<"po="<<po<<"\n";
    f  = 1./(100.1+i);
    f2 = -f; 
    
    //3.) just another example - we can fill the same tree with different objects
    //
    mistream<<"TreeK"<<f<<po<<"\n";
    mistream<<"TreeK"<<f2<<po2<<"\n";
  }
  //
  //4.) write the streamed tree's to the file and close the corresponding file in destructor
  //
  delete pmistream;
  delete file;
  //
  //5.) and now see results in file testredirector.root 
}

void TTreeSRedirector::Test()
{
  //
  //Example test function to show functionality of TTreeSRedirector
  //
  //
  //1.)create the  redirector associated with file (testredirector.root)
  //
  //
  TTreeSRedirector *pmistream= new TTreeSRedirector("testredirector.root");
  TTreeSRedirector &mistream = *pmistream;
  Char_t ch='s';
  Float_t f=3.;
  Float_t f2=1;
  TObject *po  = new TObject;
  TObject *po2 = new TObject;
  for (Int_t i=0;i<100000;i++) {
    f=i*100;
    po->SetUniqueID(i);
    po2->SetUniqueID(i*100);
    ch=i%120;
    //
    //2.) create the tree with identifier specified by first argument
    //                                layout specified by sequence of arguments
    //                                Tree identifier has to be specified as first argument !!! 
    //    if the tree and layout was already defined the consistency if layout is checked
    //                                if the data are consisten fill given tree 
    //    the name of branch can be specified using strings with = at the the end
    //    if string is not specified use automatic convention  B0, B1, ...Bn
    mistream<<"TreeIdentifier"<<"i="<<i<<"ch="<<ch<<"f="<<f<<"po="<<po<<"\n";
    f  = 1./(100.1+i);
    f2 = -f; 
    
    //3.) just another example - we can fill the same tree with different objects
    //
    mistream<<"TreeK"<<f<<po<<"\n";
    mistream<<"TreeK"<<f2<<po2<<"\n";
  }
  //
  //4.) write the streamed tree's to the file and close the corresponding file in destructor
  //
  delete pmistream;
  //
  //5.) and now see results in file testredirector.root 
}

void TTreeSRedirector::UnitTest(Int_t testEntries){
  //
  //
  //
  UnitTestSparse(0.5,testEntries);
  UnitTestSparse(0.1,testEntries);
  UnitTestSparse(0.01,testEntries);
}

void TTreeSRedirector::UnitTestSparse(Double_t scale, Int_t testEntries){
  //
  // Unit test for the TTreeSRedirector
  // 1.) Test TTreeRedirector 
  //      a.) Fill tree with random vectors
  //      b.) Fill downscaled version of vectors
  //      c.) The same skipping first entry
  // 2.) Check results wtitten to terminale
  //     a.) Disk consumption 
  //             skip data should be scale time smaller than full
  //             zerro replaced  ata should be compresed time smaller than full
  //     b.) Test invariants
  // Input parameter scale => downscaling of sprse element 
  //            
  if (scale<=0) scale=1;
  if (scale>1) scale=1;
  TTreeSRedirector *pcstream = new TTreeSRedirector("testpcstreamSparse.root","recreate");
  for (Int_t ientry=0; ientry<testEntries; ientry++){
    TVectorD vecRandom(200);
    TVectorD vecZerro(200);   // zerro vector
    for (Int_t j=0; j<200; j++) vecRandom[j]=j+ientry+0.1*gRandom->Rndm();
    Bool_t isSelected= (gRandom->Rndm()<scale);
    TVectorD *pvecFull   = &vecRandom;
    TVectorD *pvecSparse = isSelected ? &vecRandom:0;
    TVectorD *pvecSparse0 = isSelected ? &vecRandom:0;
    TVectorD *pvecSparse1 = isSelected ? &vecRandom:&vecZerro;

    if (ientry==0) {
      pvecSparse0=0;
      pvecSparse=&vecRandom;
    }
    (*pcstream)<<"Full"<<                  // stored all vectors
      "ientry="<<ientry<<
      "vec.="<<pvecFull<<                  
      "\n";
    (*pcstream)<<"SparseSkip"<<                // fraction of vectors stored
      "ientry="<<ientry<<
      "vec.="<<pvecSparse<<                
      "\n";
    (*pcstream)<<"SparseSkip0"<<               // fraction with -pointer
      "ientry="<<ientry<<
      "vec.="<<pvecSparse0<<
      "\n";
    (*pcstream)<<"SparseZerro"<<               // all vectors filled, franction filled with 0
      "ientry="<<ientry<<
      "vec.="<<pvecSparse1<<
      "\n";
  }
  delete pcstream;
  //
  // 2.) check results
  //

  TFile* f = TFile::Open("testpcstreamSparse.root");
  if (!f){
    printf("FAILED: file: %p, TTreeSRedirector::IsDisabled()=%i\n",f,TTreeSRedirector::IsDisabled()?1:0);
    return;
  }
  TTree * treeFull = (TTree*)f->Get("Full");
  TTree * treeSparseSkip = (TTree*)f->Get("SparseSkip");
  TTree * treeSparseSkip0 = (TTree*)f->Get("SparseSkip0");
  TTree * treeSparseZerro = (TTree*)f->Get("SparseZerro");
  //    a.) data volume
  //
  Double_t ratio=(1./scale)*treeSparseSkip->GetZipBytes()/Double_t(treeFull->GetZipBytes());
  Double_t ratio0=(1./scale)*treeSparseSkip0->GetZipBytes()/Double_t(treeFull->GetZipBytes());
  Double_t ratio1=(1./scale)*treeSparseZerro->GetZipBytes()/Double_t(treeFull->GetZipBytes());
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tRatioSkip\t%f\n",scale,ratio);
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tRatioSkip0\t%f\n",scale,ratio0);
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tRatioZerro\t%f\n",scale,ratio1);
  //    b.) Integrity 
  Int_t outlyersSparseSkip=treeSparseSkip->Draw("1","(vec.fElements-ientry-Iteration$-0.5)>0.5","goff");
  Int_t outlyersSparseSkip0=treeSparseSkip0->Draw("1","(vec.fElements-ientry-Iteration$-0.5)>0.5","goff");
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tOutlyersSkip\t%d\n",scale,outlyersSparseSkip!=0);
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tOutlyersSkip0\t%d\n",scale,outlyersSparseSkip0!=0);
  //    c.) Number of entries
  //
  Int_t entries=treeFull->GetEntries();
  Int_t entries0=treeSparseSkip0->GetEntries();
  Bool_t  isOKStat =(entries==entries0);
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tEntries\t%d\n",scale,isOKStat);
  //
  //   d.)Reading test
  TVectorD *pvecRead   = 0;
  treeSparseSkip0->SetBranchAddress("vec.",&pvecRead);
  Bool_t readOK=kTRUE;
  for (Int_t ientry=0; ientry<testEntries; ientry++){
    if (!pvecRead) continue;
    if (pvecRead->GetNrows()==0) continue;
    if (TMath::Abs((*pvecRead)[0]-ientry)>0.5) readOK=kFALSE;
  }
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tReadOK\t%d\n",scale,readOK);
  //
  //   e.)Global test
  Bool_t isOK=(outlyersSparseSkip0==0)&&isOKStat&&readOK;
  printf("#UnitTest:\tTTreeSRedirector::TestSparse(%f)\tisOk\t%d\n",scale,isOK);  

}

Bool_t TTreeSRedirector::fgDisabled=kFALSE;
TTreeSRedirector::TTreeSRedirector(const char *fname,const char * option) :
  fDirectory(NULL),
  fDirectoryOwner(kTRUE),
  fDataLayouts(NULL)
{
  //
  // Constructor
  //
  if (fgDisabled) {fDirectory=gDirectory;fDirectoryOwner=kFALSE;return;}

  TString name(fname);
  if (!name.IsNull()){
    fDirectory = new TFile(fname,option);
  }
  else
  {
    fDirectory = gDirectory;
    fDirectoryOwner = kFALSE;
  }
}

TTreeSRedirector::~TTreeSRedirector()
{
  //
  // Destructor
  //
  Close();       //write the tree to the selected file
  if (fDirectoryOwner)
  {
    fDirectory->Close();
    delete fDirectory;
  }
}
void TTreeSRedirector::StoreObject(TObject* object){
  //
  //
  //
  if (fgDisabled) return;
  TDirectory * backup = gDirectory;
  fDirectory->cd();
  object->Write();
  if (backup) backup->cd();
}

void  TTreeSRedirector::SetDirectory(TDirectory *sfile){
  //
  // Set the external file 
  // In case other file already attached old file is closed before
  // Redirector will be the owner of file ?
  if (fDirectory && fDirectoryOwner) {
    fDirectory->Close();
    delete fDirectory;
  }
  fDirectory=sfile;
}

TTreeStream  & TTreeSRedirector::operator<<(Int_t id)
{
  //
  // return reference to the data layout with given identifier
  // if not existing - creates new
  if (!fDataLayouts) fDataLayouts = new TObjArray(fgDisabled?1:10000);
  TTreeStream *clayout=0;
  Int_t entries = fDataLayouts->GetEntriesFast();
  for (Int_t i=0;i<entries;i++){
    TTreeStream * layout = (TTreeStream*)fDataLayouts->At(i);
    if (!layout) continue;
    if (fgDisabled?kTRUE:layout->fId==id) {
      clayout = layout;
      break;
    }
  }
  if (!clayout){
    TDirectory * backup = gDirectory;
    fDirectory->cd();
    char chname[100];
    snprintf(chname,100,"Tree%d",id);
    clayout = new TTreeStream(chname);
    clayout->fId=id;
    fDataLayouts->AddAt(clayout,entries);
    if (backup) backup->cd();
  }
  return *clayout;
}

void TTreeSRedirector::SetExternalTree(const char* name, TTree* externalTree)
{
  TTreeStream *clayout=(TTreeStream*)fDataLayouts->FindObject(name);

  if (!clayout){
    TDirectory * backup = gDirectory;
    fDirectory->cd();
    clayout = new TTreeStream(name,externalTree);
    clayout->fId=-1;
    clayout->SetName(name);
    Int_t entries = fDataLayouts->GetEntriesFast();
    fDataLayouts->AddAt(clayout,entries);
    if (backup) backup->cd();
  }
  //else
  //  AliError(Form("identifier %s already associated",name));
}


TTreeStream  & TTreeSRedirector::operator<<(const char* name)
{
  //
  // return reference to the data layout with given identifier
  // if not existing - creates new
  if (!fDataLayouts) fDataLayouts = new TObjArray(10000);
  TTreeStream *clayout=(TTreeStream*)fDataLayouts->FindObject(name);
  Int_t entries = fDataLayouts->GetEntriesFast();

  if (!clayout){
    TDirectory * backup = gDirectory;
    fDirectory->cd();
    clayout = new TTreeStream(name);
    clayout->fId=-1;
    clayout->SetName(name);
    fDataLayouts->AddAt(clayout,entries);    
    if (backup) backup->cd();
  }
  return *clayout;
}




void TTreeSRedirector::Close(){
  //
  //
  TDirectory * backup = gDirectory;
  fDirectory->cd();
  if (fDataLayouts){
    Int_t entries = fDataLayouts->GetEntriesFast();
    for (Int_t i=0;i<entries;i++){
      TTreeStream * layout = (TTreeStream*)fDataLayouts->At(i);
      if (layout && !fgDisabled){
	if (layout->fTree) layout->fTree->Write(layout->GetName());
      }
    }
    delete fDataLayouts;
    fDataLayouts=0;
  }
  if (backup) backup->cd();
}

//-------------------------------------------------------------
TTreeDataElement:: TTreeDataElement(Char_t type) :
  TNamed(),
  fType(type),
  fDType(0),
  fClass(0),
  fPointer(0)
{
  //
  //
  //
}

TTreeDataElement:: TTreeDataElement(TDataType* type) :
  TNamed(),
  fType(0),
  fDType(type),
  fClass(0),
  fPointer(0)
{
  //
  //
  //
}

TTreeDataElement:: TTreeDataElement(TClass* cl) :
  TNamed(),
  fType(0),
  fDType(0),
  fClass(cl),
  fPointer(0)
{
  //
  //
  //
}

//-------------------------------------------------------------------
TTreeStream::TTreeStream(const char *treename, TTree* externalTree):
  TNamed(treename,treename),
  fElements(0),
  fBranches(0),
  fTree(externalTree),
  fCurrentIndex(0),
  fId(0),
  fNextName(),
  fNextNameCounter(),
  fStatus(0)
{
  //
  // Standard ctor
  //
  if (!fTree) fTree = new TTree(treename, treename);
}

TTreeStream::~TTreeStream()
{
  //
  // Class dtor
  //
  fElements->Delete();
  fBranches->Clear();
  delete fElements;
  delete fBranches;
}

void TTreeStream::Close()
{
  //
  // Flush data to disk and close
  //
  if (TTreeSRedirector::IsDisabled()) return;
  fTree->Write();
}

Int_t TTreeStream::CheckIn(Char_t type, void *pointer)
{
  //
  // Insert object of given type
  //
  if (TTreeSRedirector::IsDisabled()) return 0;
  if (!fElements) fElements = new TObjArray(10000);
  if (fElements->GetSize()<=fCurrentIndex) fElements->Expand(fCurrentIndex*2);
  TTreeDataElement* element = (TTreeDataElement*)fElements->At(fCurrentIndex);
  if (!element) {
    element = new TTreeDataElement(type);
    //
    char name[1000];
    if (fNextName.Length()>0){
      if (fNextNameCounter==0){
	snprintf(name,1000,"%s",(const char*)fNextName);
      }
      if (fNextNameCounter>0){
	snprintf(name,1000,"%s%d",(const char*)fNextName,fNextNameCounter);
      }      
    }
    else{
      snprintf(name,1000,"B%d.",fCurrentIndex);
    }
    element->SetName(name);
    //
    element->SetPointer(pointer);
    fElements->AddAt(element,fCurrentIndex);
    fCurrentIndex++;
    return 0; //new element added
  }
  if (element->GetType()!=type){
    fStatus++;
    return 1; //mismatched data element
  }
  element->SetPointer(pointer);
  fCurrentIndex++;
  return 0;
}

Int_t TTreeStream::CheckIn(TObject *pObject){
  //
  // Insert TObject
  //
  if (TTreeSRedirector::IsDisabled()) return 0;
  TClass *pClass = 0;
  if (pObject) pClass=pObject->IsA();
  if (!fElements) fElements = new TObjArray(1000);
  TTreeDataElement* element = (TTreeDataElement*)fElements->At(fCurrentIndex);
  if (!element) {
    element = new TTreeDataElement(pClass);
    //
    char name[1000];
    if (fNextName.Length()>0){
      if (fNextNameCounter==0){
	snprintf(name,1000,"%s",(const char*)fNextName);
      }
      if (fNextNameCounter>0){
	snprintf(name,1000,"%s%d",(const char*)fNextName,fNextNameCounter);
      }      
    }
    else{
      snprintf(name,1000,"B%d",fCurrentIndex);
    }
    element->SetName(name);
    
    element->SetPointer(pObject);
    fElements->AddAt(element,fCurrentIndex);
    fCurrentIndex++;
    return 0; //new element added
  }
  if (element->fClass==0) {
    element->fClass=pClass;
  }else{
    if (element->fClass!=pClass && pClass!=0){
      fStatus++;
      return 1; //mismatched data element
    }
  }
  element->SetPointer(pObject);
  fCurrentIndex++;
  return 0;  
}

void TTreeStream::BuildTree(){
  //
  // Build the Tree
  //
  //if (fTree && fTree->GetEntries()>0) return;
  if (TTreeSRedirector::IsDisabled()) return;
  Int_t entriesFilled=0;
  if (!fTree)  {
    fTree = new TTree(GetName(),GetName());
  }else{
    entriesFilled=fTree->GetEntries();
  }
  Int_t entries = fElements->GetEntriesFast();  
  if (!fBranches) fBranches = new TObjArray(entries);
  
  for (Int_t i=0;i<entries;i++){
    //
    TTreeDataElement* element = (TTreeDataElement*)fElements->At(i);
    if (fBranches->At(i)) continue;
    char bname1[1000];
    if (element->GetName()[0]==0){
      snprintf(bname1,1000,"B%d",i);
    }
    else{
      snprintf(bname1,1000,"%s",element->GetName());
    }
    if (element->fClass){
      if (element->fClass->GetBaseClass("TClonesArray")){
	TBranch * br = fTree->Branch(bname1,element->fClass->GetName(),&(element->fPointer));
	if (entriesFilled!=0) {
	  br->SetAddress(0);
	  for (Int_t ientry=0; ientry<entriesFilled;ientry++) br->Fill();
	  br->SetAddress(&(element->fPointer));
	}
	fBranches->AddAt(br,i);
      }else
	{
	  TBranch * br = fTree->Branch(bname1,element->fClass->GetName(),&(element->fPointer));
	  if (entriesFilled!=0) {
	    br->SetAddress(0);
	    for (Int_t ientry=0; ientry<entriesFilled;ientry++) br->Fill();
	    br->SetAddress(&(element->fPointer));
	  }
	  fBranches->AddAt(br,i);
	}
    }
    if (element->GetType()>0){
      char bname2[1000];
      snprintf(bname2,1000,"%s/%c",bname1,element->GetType());
      TBranch * br = fTree->Branch(bname1,element->fPointer,bname2);
      if (entriesFilled!=0) {
	br->SetAddress(0);
	for (Int_t ientry=0; ientry<entriesFilled;ientry++) br->Fill();
	br->SetAddress(element->fPointer);
      }

      fBranches->AddAt(br,i);
    }
  }
}

void TTreeStream::Fill(){
  //
  // Fill the tree
  //
  if (TTreeSRedirector::IsDisabled()) return;
  if (fTree) { 
    Int_t entries=fElements->GetEntriesFast();
    if (entries>fTree->GetNbranches()) BuildTree();
    for (Int_t i=0;i<entries;i++){    
      TTreeDataElement* el  = (TTreeDataElement*)fElements->At(i);
      if (!el) continue;
      if (!el->GetType()) continue;
      TBranch      * br  = (TBranch*)fBranches->At(i);
      if (br &&el){
	if (el->GetType())  br->SetAddress(el->fPointer);
      }
    }
    if (fStatus==0) fTree->Fill(); //fill only in case of non conflicts
    fStatus=0;
  }
}

TTreeStream & TTreeStream::Endl()
{
  //
  // Perform pseudo endl operation
  //
  if (TTreeSRedirector::IsDisabled()) return *this;
  if (fTree->GetNbranches()==0) BuildTree();
  Fill();
  fStatus =0;
  fCurrentIndex=0;
  return *this;
}


TTreeStream  &TTreeStream::operator<<(const Char_t *name)
{
  //
  // Endl 
  //
  if (name[0]=='\n'){
    return Endl();
  }
  //
  //if tree was already defined ignore
  if (fTree->GetEntries()>0) return *this;
  //check branch name if tree was not 
  //
  Int_t last=0;
  for (last=0;;last++){
    if (name[last]==0) break;    
  }
  
  if (last>0&&name[last-1]=='='){
    fNextName = name;
    fNextName[last-1]=0;
    fNextNameCounter=0;
  }
  return *this;
}


void TTreeSRedirector::FixLeafNameBug(TTree* tree){
  // On the fly BUG FIX for name and titles of branches and Leave:
  //     renaming Leaves and changing title of branches to be the same as Branch name
  // Explanation of FIX:
  //    In the  friend tree Join logic it is assumed leaf names are used to find appropraiat primary/secondary keys
  //    For the standard queries hovwer the branch names are used to identify data
  //    Hovewer in the Branch constructor it is not checked
  // As a consequence  - in case the name of the leave and and the name of branch is not the same  + freind trees are sparse
  //    wrong joins ( unrelated pair of information) are used
  // FIX:
  //   To be able to use friend trees with proper indexing (in case of sarse trees) branches and leaves has to be named consistently
  //   In this routine bnrach name is taken as a reference and branch title and leave name titles are renamed
  //   After the fix unit test code with pairs of sprse friend trees worked properly
  // Side effects of fix:
  //
  if (tree==NULL) return;
  TObjArray *brArray = tree->GetListOfBranches();
  TObjArray *lArray = tree->GetListOfLeaves();
  for (Int_t i = 0; i < brArray->GetLast(); i++) {
    TBranch *br = (TBranch *) brArray->At(i);
    if (TString(br->GetTitle()).Contains(br->GetName()) == 0) {
      TString brTitle(br->GetTitle());
      Int_t pos = brTitle.First("/");
      TString leafName = "";
      if (pos < brTitle.Length()) {
        brTitle[pos] = 0;
        leafName = TString::Format("%s", brTitle.Data()).Data();
        TLeaf * leaf = (TLeaf*)lArray->FindObject(leafName);
        if (leaf) {
          leaf->SetName(br->GetName());
          leaf->SetTitle(br->GetName());
          br->SetTitle(TString::Format("%s/%s",br->GetName(),&(brTitle.Data()[pos+1])).Data());
        }
      }
    }
  }
}
