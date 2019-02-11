#ifndef AliOADBContainer_H
#define AliOADBContainer_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//     Offline Analysis Database Container and Service Class 
//     Author: Andreas Morsch, CERN
//-------------------------------------------------------------------------

#include <TNamed.h>
#include <TList.h>
#include <TArrayI.h>
#include <TObjArray.h>
#include <TString.h>
class TFile;

class AliOADBObjCache : public TObject {
 public :
  AliOADBObjCache() : fRun1(0), fRun2(0), fObj(0) {;}
  AliOADBObjCache(TObject *obj, Int_t r1, Int_t r2) : fRun1(r1), fRun2(r2), fObj(obj) {;}
  ~AliOADBObjCache() {delete fObj;}
  TObject *GetObject(Int_t run)    const {if ((run>=fRun1) && (run<=fRun2)) return fObj; else return 0;}
  const char *GetName()            const {if (fObj) return fObj->GetName(); else return TObject::GetName();}
  const char *GetTitle()           const {if (fObj) return fObj->GetTitle(); else return TObject::GetTitle();}
  void 	Print(Option_t *option="") const {printf("%s range %d to %d\n",GetName(),fRun1,fRun2); if (strlen(option)>0) fObj->Print();}
 private :
  Int_t fRun1;   ///< lowest run number for which the object is valid
  Int_t fRun2;   ///< highest run number for which the object is valid
  TObject *fObj; ///< the cached object
  AliOADBObjCache(const AliOADBObjCache& cont); 
  AliOADBObjCache& operator=(const AliOADBObjCache & cont);
  ClassDef(AliOADBObjCache, 1); // AliOADBObjCache
};

class AliOADBContainer : public TNamed {

 public :
  AliOADBContainer(Bool_t b=0);
  AliOADBContainer(const char* name, Bool_t b=0);
  virtual ~AliOADBContainer();
  AliOADBContainer(const AliOADBContainer& cont); 
  AliOADBContainer& operator=(const AliOADBContainer& cont);
// Object adding and removal
/** 
 * @brief Setting new default object
 * AddDefaultObject must get objects which are unique, and are not handled as run-dependent objects, otherwise using SetOwner will end up in double-deletes.
*/
  void   AppendObject(TObject* obj, Int_t lower, Int_t upper, TString passName="");
  void   UpdateObject(Int_t index, TObject* obj, Int_t lower, Int_t upper, TString passName="");
  void   RemoveObject(Int_t index);
  void   AddDefaultObject(TObject* obj);
  void   CleanDefaultList();
  void   CleanLists();
  TList* GetDefaultList() const {return fDefaultList;}
// I/O  
  void  WriteToFile(const char* fname)  const;
  Int_t InitFromFile(const char* fname, const char* key);
  void  SetOwner(Bool_t flag);
// Getters
  Int_t GetNumberOfEntries()    const {return fEntries;}
  Int_t LowerLimit(Int_t idx)   const {return fLowerLimits[idx];}
  Int_t UpperLimit(Int_t idx)   const {return fUpperLimits[idx];}
  TObjArray* GetObjArray() {return fArray;}
  void SetToZeroObjArray() {fArray=0;}
  AliOADBObjCache* GetObjectCache(Int_t run, const char* def="", TString passName="") const;
  TObject* GetObject(Int_t run, const char* def = "", TString passName="") const;
  TObject* GetObjectFromFile(TFile* file, Int_t run, const char* def = "", TString passName="") const;
  TObject* GetObjectByIndex(Int_t run) const;
  TObject* GetPassNameByIndex(Int_t idx) const;
  TObject* GetDefaultObject(const char* key) 
           {return(fDefaultList->FindObject(key));}
// Debugging  
  void List();
// Browsable
  virtual Bool_t	IsFolder() const { return kTRUE; }
  void Browse(TBrowser *b);
  Int_t GetIndexForRun(Int_t run, TString passName="") const;
//
  static const char*   GetOADBPath();
 private:
  Int_t HasOverlap(Int_t lower, Int_t upper, TString passName) const;
 private :
  TObjArray*               fArray;         ///< Array with objects corresponding to run ranges
  TList*                   fDefaultList;   ///< List with default arrays (not in run ranges!)
  TObjArray*               fPassNames;     ///< Pass names
  TArrayI                  fLowerLimits;   ///< lower limit of run range
  TArrayI                  fUpperLimits;   ///< upper limit of run range
  Int_t                    fEntries;       ///< Number of entries
  Bool_t                   fDefOwn;        ///< Default ownership (off by default)
  ClassDef(AliOADBContainer, 3);
};

#endif
