#ifndef HistogramManager_H
#define HistogramManager_H

#include <TString.h>
#include <TObject.h>
#include <TList.h>
#include <THashList.h>

class TAxis;
class TArrayD;
class TObjArray;
class TFile;


class HistogramManager : public TObject {

 public:
  HistogramManager(const char* name, const int maxNVars);
  virtual ~HistogramManager();
  
  enum Constants {
    kNothing = -1
  };
  
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
  const Bool_t* GetUsedVars() const {return fUsedVars;}
  
  void WriteOutput(TFile* saveFile);
  void InitFile(const Char_t* filename, const Char_t* mainListName="");    // open an output file for reading
  void AddToOutputList(TList* list) {fOutputList.Add(list);}
  void CloseFile();
  
  const THashList* GetMainHistogramList() const {return &fMainList;}    // get a histogram list
  const THashList* GetMainDirectory() const {return fMainDirectory;}    // get the main histogram list from the loaded file
  
  TList* AddHistogramsToOutputList(); // get all histograms on a TList              // NEWNEW
  
  TList* GetHistogramOutputList() {return &fOutputList;}        // NEWNEW
  TList* GetHistogramList(const Char_t* listname) const;    // get a histogram list      NEWNEW
  TObject* GetHistogram(const Char_t* listname, const Char_t* hname) const;  // get a histogram from an old output
  
    
  ULong_t GetAllocatedBins() const {return fBinsAllocated;}  
  void Print(Option_t*) const;
  
 private: 
   
  THashList fMainList;           // master histogram list
  int fNVars;
  THashList* fMainDirectory;     //! main directory with analysis output (this is used for loading output files and retrieving histograms offline)
  TFile* fHistFile;              //! pointer to a TFile opened for reading 
  TList fOutputList;         // TList for output histograms
   
  // Array of bool flags toggled when a variable is used (filled in a histogram)
  bool* fUsedVars;           // map of used variables
  
  Bool_t fUseDefaultVariableNames;       // toggle the usage of default variable names and units
  ULong_t fBinsAllocated;                // number of allocated bins
  TString* fVariableNames;               //! variable names
  TString* fVariableUnits;               //! variable units
    
  void MakeAxisLabels(TAxis* ax, const Char_t* labels);
  
  HistogramManager& operator= (const HistogramManager &c);
  HistogramManager(const HistogramManager &c);
  
  ClassDef(HistogramManager, 1)
};

#endif
