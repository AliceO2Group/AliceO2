// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//

#ifndef VarManager_H
#define VarManager_H

//#include "Analysis/ReducedInfoTables.h"
#include <TObject.h>
#include <TString.h>

#include <vector>
#include <map>
//using namespace o2::aod;
using std::vector;

//_________________________________________________________________________
class VarManager : public TObject
{

 public:
  enum Variables {
    kNothing = -1,
    // Run wise variables
    kRunNo = 0,
    kRunId,
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
  }; // end of Variables enumeration

  static TString fgVariableNames[kNVars]; // variable names
  static TString fgVariableUnits[kNVars]; // variable units
  static void SetDefaultVarNames();

  static void SetUseVariable(Variables var)
  {
    fgUsedVars[var] = kTRUE;
    SetVariableDependencies();
  }
  static void SetUseVars(const bool* usedVars)
  {
    for (int i = 0; i < kNVars; ++i) {
      if (usedVars[i])
        fgUsedVars[i] = kTRUE; // overwrite only the variables that are being used since there are more channels to modify the used variables array, independently
    }
    SetVariableDependencies();
  }
  static bool GetUsedVar(Variables var) { return fgUsedVars[var]; }

  static void SetRunNumbers(int n, int* runs);

  static void FillEvent(vector<float> event, float* values = nullptr);
  static void FillTrack(vector<float> track, float* values = nullptr);

 public:
  VarManager();
  ~VarManager() override;

  static float fgValues[kNVars]; // array holding all variables computed during analysis
  static void ResetValues(int startValue = 0, int endValue = kNVars);

 private:
  static bool fgUsedVars[kNVars];        // holds flags for when the corresponding variable is needed (e.g., in the histogram manager, in cuts, mixing handler, etc.)
  static void SetVariableDependencies(); // toggle those variables on which other used variables might depend

  static std::map<int, int> fgRunMap; // map of runs to be used in histogram axes

  VarManager& operator=(const VarManager& c);
  VarManager(const VarManager& c);

  ClassDef(VarManager, 1)
};
#endif
