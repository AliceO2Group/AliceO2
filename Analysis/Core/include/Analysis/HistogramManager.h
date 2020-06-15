#ifndef HISTOGRAMMANAGER_H
#define HISTOGRAMMANAGER_H

#include <TString.h>
#include <TObject.h>
#include <THn.h>
#include <TList.h>
#include <THashList.h>

#include "Analysis/VarManager.h"

class TAxis;
class TArrayD;
class TObjArray;
//class TDirectoryFile;
class TFile;


class HistogramManager : public TObject {

 public:
  HistogramManager();
  HistogramManager(const char* name);
  virtual ~HistogramManager();
  
  void SetName(const Char_t* name) {fName = name;}
  
  void AddHistClass(const Char_t* histClass);
  void AddHistogram(const Char_t* histClass, const Char_t* name, const Char_t* title, Bool_t isProfile,
                    Int_t nXbins, Double_t xmin, Double_t xmax, Int_t varX,
                    Int_t nYbins=0, Double_t ymin=0, Double_t ymax=0, Int_t varY=-1,
                    Int_t nZbins=0, Double_t zmin=0, Double_t zmax=0, Int_t varZ=-1,
                    const Char_t* xLabels="", const Char_t* yLabels="", const Char_t* zLabels="",
                    Int_t varT=-1, Int_t varW=-1);
  void AddHistogram(const Char_t* histClass, const Char_t* name, const Char_t* title, Bool_t isProfile,
                    Int_t nXbins, Double_t* xbins, Int_t varX,
                    Int_t nYbins=0, Double_t* ybins=0x0, Int_t varY=-1,
                    Int_t nZbins=0, Double_t* zbins=0x0, Int_t varZ=-1,
                    const Char_t* xLabels="", const Char_t* yLabels="", const Char_t* zLabels="",
                    Int_t varT=-1, Int_t varW=-1);
  void AddHistogram(const Char_t* histClass, const Char_t* name, const Char_t* title,
                    Int_t nDimensions, Int_t* vars, Int_t* nBins, Double_t* xmin, Double_t* xmax,
                    TString* axLabels=0x0, Int_t varW=-1, Bool_t useSparse=kFALSE);
  void AddHistogram(const Char_t* histClass, const Char_t* name, const Char_t* title,
                    Int_t nDimensions, Int_t* vars, TArrayD* binLimits,
                    TString* axLabels=0x0, Int_t varW=-1, Bool_t useSparse=kFALSE);
  
  void FillHistClass(const Char_t* className, Float_t* values);
  
  void SetUseDefaultVariableNames(Bool_t flag) {fUseDefaultVariableNames = flag;};
  void SetDefaultVarNames(TString* vars, TString* units);
  void WriteOutput(TFile* saveFile);
  void InitFile(const Char_t* filename, const Char_t* mainListName="");    // open an output file for reading
  void AddToOutputList(TList* list) {fOutputList.Add(list);}
  void CloseFile();
  const THashList* GetMainHistogramList() const {return &fMainList;}    // get a histogram list
  const THashList* GetMainDirectory() const {return fMainDirectory;}    // get the main histogram list from the loaded file
  THashList* AddHistogramsToOutputList(); // get all histograms on a THashList
  THashList* GetHistogramOutputList() {return &fOutputList;} 
  THashList* GetHistogramList(const Char_t* listname) const;    // get a histogram list
  TObject* GetHistogram(const Char_t* listname, const Char_t* hname) const;  // get a histogram from an old output
  const Bool_t* GetUsedVars() const {return fUsedVars;}
    
  ULong_t GetAllocatedBins() const {return fBinsAllocated;}  
  void Print(Option_t*) const;
  
 private: 
   
  THashList fMainList;           // master histogram list
  TString fName;                 // master histogram list name
  THashList* fMainDirectory;     //! main directory with analysis output (this is used for loading output files and retrieving histograms offline)
  TFile* fHistFile;              //! pointer to a TFile opened for reading 
  THashList fOutputList;         // THashList for output histograms
   
  // Array of bool flags toggled when a variable is used (filled in a histogram)
  Bool_t fUsedVars[VarManager::kNVars];           // map of used variables
  
  Bool_t fUseDefaultVariableNames;       // toggle the usage of default variable names and units
  ULong_t fBinsAllocated;                // number of allocated bins
  TString fVariableNames[VarManager::kNVars];               //! variable names
  TString fVariableUnits[VarManager::kNVars];               //! variable units
    
  void MakeAxisLabels(TAxis* ax, const Char_t* labels);
  
  HistogramManager& operator= (const HistogramManager &c);
  HistogramManager(const HistogramManager &c);
  
  ClassDef(HistogramManager, 1)
};

#endif
