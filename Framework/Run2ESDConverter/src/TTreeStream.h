#ifndef TTREESTREAM_H
#define TTREESTREAM_H
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TTreeSRedirector                                                              //                      

#include "TObject.h"
#include "TString.h"
#include "TTree.h"
#include "TDirectory.h"
#include "TFile.h"
class TObjArray;
class TTree;
class TDataType;

class TTreeDataElement: public TNamed {
  friend class TTreeStream;
 public:
  TTreeDataElement(Char_t type);
  TTreeDataElement(TDataType* type);
  TTreeDataElement(TClass* cl);
  void   SetPointer(void* pointer) {fPointer=pointer;} 
  Char_t GetType() const {return fType;}
 protected:

  TTreeDataElement(const TTreeDataElement & tde);
  TTreeDataElement & operator=(const TTreeDataElement & tde);

  Char_t  fType;     // type of data element
  TDataType *fDType; //data type pointer 
  TClass    *fClass; //data type pointer
  void * fPointer;  // pointer to element
  ClassDef(TTreeDataElement,2)
};

class TTreeStream: public TNamed {
  friend class TTreeSRedirector;
public:
  TTreeStream(const char *treename, TTree* externalTree=NULL);
  ~TTreeStream();
  void Close();
  static void Test();
  Int_t CheckIn(Char_t type, void *pointer);  
  //Int_t CheckIn(const char *type, void *pointer);
  Int_t CheckIn(TObject *o);
  void BuildTree();
  void Fill();
  Double_t GetSize(){ return fTree->GetZipBytes();}
  TTreeStream& Endl();
  //
  TTreeStream  &operator<<(Bool_t   &b){CheckIn('B',&b);return *this;}
  TTreeStream  &operator<<(Char_t   &c){CheckIn('B',&c);return *this;}
  TTreeStream  &operator<<(UChar_t  &c){CheckIn('b',&c);return *this;}
  TTreeStream  &operator<<(Short_t  &h){CheckIn('S',&h);return *this;}
  TTreeStream  &operator<<(UShort_t &h){CheckIn('s',&h);return *this;}
  TTreeStream  &operator<<(Int_t    &i){CheckIn('I',&i);return *this;}
  TTreeStream  &operator<<(UInt_t   &i){CheckIn('i',&i);return *this;}
  TTreeStream  &operator<<(Long_t   &l){CheckIn('L',&l);return *this;}
  TTreeStream  &operator<<(ULong_t  &l){CheckIn('l',&l);return *this;}
  TTreeStream  &operator<<(Long64_t &l){CheckIn('L',&l);return *this;}
  TTreeStream  &operator<<(ULong64_t &l){CheckIn('l',&l);return *this;}
  TTreeStream  &operator<<(Float_t   &f){CheckIn('F',&f);return *this;}
  TTreeStream  &operator<<(Double_t  &d){CheckIn('D',&d);return *this;}
  TTreeStream  &operator<<(TObject*o){CheckIn(o);return *this;} 
  TTreeStream  &operator<<(const Char_t *name);
  TTree * GetTree() const { return fTree;}
 protected:
  //

  TTreeStream(const TTreeStream & ts);
  TTreeStream & operator=(const TTreeStream & ts);

  TObjArray *fElements; //array of elements
  TObjArray *fBranches; //pointers to branches
  TTree *fTree;         //data storage
  Int_t fCurrentIndex;  //index of current element
  Int_t fId;            //identifier of layout
  TString fNextName;    //name for next entry
  Int_t   fNextNameCounter; //next name counter
  Int_t   fStatus;      //status of the layout
  ClassDef(TTreeStream,1)
};


class TTreeSRedirector: public TObject { 
public:
  TTreeSRedirector(const char *fname="", const char * option="update");
  virtual ~TTreeSRedirector();
  void Close();
  static void Test();
  static void Test2();
  static void UnitTestSparse(Double_t scale, Int_t testEntries);
  static void UnitTest(Int_t testEntries=5000);
  void StoreObject(TObject* object);
  TFile * GetFile() {return fDirectory->GetFile();}
  TDirectory * GetDirectory() {return fDirectory;}
  virtual   TTreeStream  &operator<<(Int_t id);
  virtual   TTreeStream  &operator<<(const char *name);
  void      SetDirectory(TDirectory *sfile); 
  void      SetFile(TFile *sfile) {SetDirectory(sfile);} 
  void SetExternalTree(const char* name, TTree* externalTree);
  static void SetDisabled(Bool_t b=kTRUE) {fgDisabled=b;}
  static Bool_t IsDisabled()        {return fgDisabled;}
    static void FixLeafNameBug(TTree* tree);
private:

  TTreeSRedirector(const TTreeSRedirector & tsr);
  TTreeSRedirector & operator=(const TTreeSRedirector & tsr);

  TDirectory* fDirectory;        //file
  Bool_t      fDirectoryOwner;   //do we own the directory?
  TObjArray *fDataLayouts;   //array of data layouts
  static Bool_t fgDisabled;  //disable - do not open any files
  ClassDef(TTreeSRedirector,2) 
};




#endif
