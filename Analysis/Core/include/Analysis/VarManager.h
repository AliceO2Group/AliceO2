
#ifndef VARMANAGER_H
#define VARMANAGER_H

#include <TObject.h>
#include <TString.h>

//#include "Framework/runDataProcessing.h"
//#include "Framework/ASoA.h"
//#include "Framework/ASoAHelpers.h"
//#include "Framework/DataTypes.h"
//#include "Framework/AnalysisTask.h"
//#include "Framework/AnalysisDataModel.h"
#include "Analysis/ReducedInfoTables.h"


//using namespace o2;
//using namespace o2::framework;
//using namespace o2::framework::expressions;
//using namespace o2::aod;

//using Event = soa::Join<o2::aod::ReducedEvents, o2::aod::ReducedEventsExtended, o2::aod::ReducedEventsVtxCov>::iterator;
//using Track = soa::Join<o2::aod::ReducedTracks, o2::aod::ReducedTracksBarrel>::iterator;

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
  
  /*template <typename T> 
  void Fill(T const& object, float* values) {
    if constexpr (std::is_same_v<T, Event>)
      FillEvent(T,values);
    if constexpr (std::is_same_v<T, Track>)
      FillTrack(T,values);
  };*/
  
  static void FillEvent(o2::aod::ReducedEvent event, float* values);
  static void FillTrack(o2::aod::ReducedTrack track, float* values);
    
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
