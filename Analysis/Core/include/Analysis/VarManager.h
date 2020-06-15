
#ifndef VARMANAGER_H
#define VARMANAGER_H

#include <TObject.h>
#include <TString.h>

//_________________________________________________________________________
class VarManager : public TObject {

public:  
  enum Variables {
    kNothing = -1,
    // Run wise variables
    kRunNo = 0,
    kRunTimeStart,
    kRunTimeStop,
    kLHCFillNumber,
    kDipolePolarity,
    kL3Polarity,
    kNRunWiseVariables,
    
    // Event wise variables
    kTimeStamp,
    kBC,
    kInstLumi,
    kEventType,
    kIsPhysicsSelection,
    kVtxX,
    kVtxY,
    kVtxZ,
    kVtxNcontrib,
    kVtxCovXX,
    kVtxCovXY,
    kVtxCovXZ,
    kVtxCovYY,
    kVtxCovYZ,
    kVtxCovZZ,
    kVtxChi2,
    kCentVZERO,
    kNEventWiseVariables,
    
    // Basic track(pair) wise variables
    kPt,
    kEta,
    kPhi,
    kRap,
    kMass,
    kCharge,
    kNBasicTrackVariables,
    
    // Barrel track variables
    kPin,
    kITSncls,
    kITSchi2,
    kITSlayerHit,
    kTPCncls,
    kTPCchi2,
    kTPCsignal,
    kTRDsignal,
    kTOFsignal,
    kTrackLength,
    kNBarrelTrackVariables,
    
    // Muon track variables
    kNMuonTrackVariables,
    
    // Pair variables
    kCandidateId,
    kPairType,
    kPairLxy,
    kNPairVariables,
    
    // Candidate-track correlation variables
    kDeltaEta,
    kDeltaPhi,
    kNCorrelationVariables,
    
    kNVars
  };           // end of Variables enumeration
  
  static TString fgVariableNames[kNVars];         // variable names
  static TString fgVariableUnits[kNVars];         // variable units  
  static void SetDefaultVarNames();  
    
  static void SetUseVariable(Variables var) {fgUsedVars[var] = kTRUE; SetVariableDependencies();}
  static void SetUseVars(bool* usedVars) {
    for(int i=0;i<kNVars;++i) {
      if(usedVars[i]) fgUsedVars[i]=kTRUE;    // overwrite only the variables that are being used since there are more channels to modify the used variables array, independently
    }
    SetVariableDependencies();
  }
  static bool GetUsedVar(Variables var) {return fgUsedVars[var];}
  
public:   
  VarManager();
  virtual ~VarManager();
  
private:
   
  static Bool_t fgUsedVars[kNVars];            // holds flags for when the corresponding variable is needed (e.g., in the histogram manager, in cuts, mixing handler, etc.) 
  static void SetVariableDependencies();       // toggle those variables on which other used variables might depend 
  
  VarManager& operator= (const VarManager &c);
  VarManager(const VarManager &c);
  
  ClassDef(VarManager, 1)
};
#endif
