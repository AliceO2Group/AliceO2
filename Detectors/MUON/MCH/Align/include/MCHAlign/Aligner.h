// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file CathodeSegmentation.h
 * C++ Alignmnet .
 * @author  Javier Castillo Castellanos
 */

#ifndef ALICEO2_MCH_ALIGNER
#define ALICEO2_MCH_ALIGNER

#include <string>
#include <vector>

#include "DataFormatsMCH/Cluster.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "ForwardAlign/MillePede2.h"
#include "ForwardAlign/MillePedeRecord.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "MCHTracking/Track.h"

#include <TFile.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TObject.h>
#include <TString.h>
#include <TTree.h>

namespace o2
{

namespace mch
{

/// local track parameters, for refit
class LocalTrackParam
{
 public:
  //* construction
  LocalTrackParam() = default;
  ~LocalTrackParam() = default;

  // private:
  //* y and z
  double fTrackX = 0.0;
  double fTrackY = 0.0;
  double fTrackZ = 0.0;
  double fTrackSlopeX = 0.0;
  double fTrackSlopeY = 0.0;
}; // class LocalTrackParam

/// local track residual, for tempoarary eval
class LocalTrackClusterResidual
{
 public:
  //* construction
  LocalTrackClusterResidual() = default;
  ~LocalTrackClusterResidual() = default;

  // private:
  //* y and z
  int fClDetElem = 0.0;
  int fClDetElemNumber = 0.0;

  float fClusterX = 0.0;
  float fClusterY = 0.0;
  float fClusterZ = 0.0;
  float fClusterXloc = 0.0;
  float fClusterYloc = 0.0;

  float fTrackX = 0.0;
  float fTrackY = 0.0;
  float fTrackZ = 0.0;
  float fTrackXloc = 0.0;
  float fTrackYloc = 0.0;

  float fTrackSlopeX = 0.0;
  float fTrackSlopeY = 0.0;

  float fBendingMomentum = 0.0;

  float fResiduXGlobal = 0.0;
  float fResiduYGlobal = 0.0;

  float fResiduXLocal = 0.0;
  float fResiduYLocal = 0.0;

  float fCharge = 0.0;

  float fBx = 0.0;
  float fBy = 0.0;
  float fBz = 0.0;

}; // class LocalTrackClusterResidual

class Aligner : public TObject
{

 public:
  Aligner();

  ~Aligner();

  // initialize
  void init(TString DataRecFName = "millerecords.root", TString ConsRecFName = "milleconstraints.root");

  // terminate
  void terminate();

  // array dimendions
  enum {
    /// Number tracking stations
    fgNSt = 5,

    /// Number tracking chambers
    fgNCh = 10,

    /// Number of tracking modules
    fgNTrkMod = 16,

    /// Number of half chambers
    fgNHalfCh = 20,

    /// max number of detector elements per half chamber
    fgNDetHalfChMax = 13,

    /// Total number of detection elements
    /// (4*2 + 4*2 + 18*2 + 26*2 + 26*2)
    fgNDetElem = 156,

    /// Number of local parameters
    fNLocal = 4, // t_x, t_y, x0, y0

    /// Number of degrees of freedom per chamber
    fgNParCh = 4, // x,y,z,phi

    /// Number of global parameters
    fNGlobal = fgNParCh * fgNDetElem
  };

  /// Number of detection elements per chamber
  static const int fgNDetElemCh[fgNCh];

  /// Sum of detection elements up to this chamber
  static const int fgSNDetElemCh[fgNCh + 1];

  /// Number of detection element per tracking module
  static const int fgNDetElemHalfCh[fgNHalfCh];

  /// list of detection elements per tracking module
  static const int fgDetElemHalfCh[fgNHalfCh][fgNDetHalfChMax];

  /// global parameter bit set, used for masks
  enum ParameterMask {
    ParX = 1 << 0,
    ParY = 1 << 1,
    ParZ = 1 << 2,
    ParTZ = 1 << 3,

    ParAllTranslations = ParX | ParY | ParZ,
    ParAllRotations = ParTZ,
    ParAll = ParAllTranslations | ParAllRotations

  };

  /// detector sides bit set, used for selecting sides in constrains
  enum SidesMask {
    SideTop = 1 << 0,
    SideLeft = 1 << 1,
    SideBottom = 1 << 2,
    SideRight = 1 << 3,
    SideTopLeft = SideTop | SideLeft,
    SideTopRight = SideTop | SideRight,
    SideBottomLeft = SideBottom | SideLeft,
    SideBottomRight = SideBottom | SideRight,
    AllSides = SideTop | SideBottom | SideLeft | SideRight
  };

  void ProcessTrack(Track& track, const o2::mch::geo::TransformationCreator& transformation, Bool_t doAlignment, Double_t weight = 1);

  //@name modifiers
  //@{

  /// run number
  void SetRunNumber(int id)
  {
    fRunNumber = id;
  }

  /// Set flag for Magnetic field On/Off
  void SetBFieldOn(bool value)
  {
    fBFieldOn = value;
  }

  /// set to true to do refit evaluation
  void SetDoEvaluation(bool value)
  {
    fDoEvaluation = value;
  }

  /// set to true to refit tracks
  void SetRefitStraightTracks(bool value)
  {
    fRefitStraightTracks = value;
  }

  void SetAllowedVariation(int iPar, double value);

  void SetSigmaXY(double sigmaX, double sigmaY);

  void FixAll(unsigned int parameterMask = ParAll);

  void FixChamber(int iCh, unsigned int parameterMask = ParAll);

  void FixDetElem(int iDetElemId, unsigned int parameterMask = ParAll);

  void FixHalfSpectrometer(const bool* bChOnOff, unsigned int sidesMask = AllSides, unsigned int parameterMask = ParAll);

  void FixParameter(int iPar);

  void FixParameter(int iDetElem, int iPar)
  {
    FixParameter(iDetElem * fgNParCh + iPar);
  }

  //@}

  //@name releasing detectors
  //@{

  void ReleaseChamber(int iCh, unsigned int parameterMask = ParAll);

  void ReleaseDetElem(int iDetElemId, unsigned int parameterMask = ParAll);

  void ReleaseParameter(int iPar);

  void ReleaseParameter(int iDetElem, int iPar)
  {
    ReleaseParameter(iDetElem * fgNParCh + iPar);
  }

  //@}

  //@name grouping detectors
  //@{

  void GroupChamber(int iCh, unsigned int parameterMask = ParAll);

  void GroupHalfChamber(int iCh, int iHalf, unsigned int parameterMask = ParAll);

  void GroupDetElems(int detElemMin, int detElemMax, unsigned int parameterMask = ParAll);

  void GroupDetElems(const int* detElemList, int nDetElem, unsigned int parameterMask = ParAll);

  //@}

  //@name define non linearity
  //@{

  void SetChamberNonLinear(int iCh, unsigned int parameterMask);

  void SetDetElemNonLinear(int iSt, unsigned int parameterMask);

  void SetParameterNonLinear(int iPar);

  void SetParameterNonLinear(int iDetElem, int iPar)
  {
    SetParameterNonLinear(iDetElem * fgNParCh + iPar);
  }

  //@}

  //@name constraints
  //@{

  void AddConstraints(const bool* bChOnOff, unsigned int parameterMask);

  void AddConstraints(const bool* bChOnOff, const bool* lVarXYT, unsigned int sidesMask = AllSides);

  //@}

  /// initialize global parameters to a give set of values
  void InitGlobalParameters(double* par);

  /// perform global fit
  void GlobalFit(std::vector<double>& params, std::vector<double>& errors, std::vector<double>& pulls);

  /// print global parameters
  void PrintGlobalParameters(void) const;

  /// get error on a given parameter
  double GetParError(int iPar) const;

  o2::fwdalign::MillePedeRecord& GetRecord() { return fTrackRecord; }

  void ReAlign(std::vector<o2::detectors::AlignParam>& params, std::vector<double>& misAlignments);

  void SetAlignmentResolution(const TClonesArray* misAlignArray, int chId, double chResX, double chResY, double deResX, double deResY);

  TTree* GetResTree()
  {
    return fTTree;
  }

  void SetReadOnly()
  {
    mRead = true;
  }

  void DisableRecordWriter()
  {
    fDisableRecordWriter = true;
  }

 private:
  /// Not implemented
  Aligner(const Aligner& right);

  /// Not implemented
  Aligner& operator=(const Aligner& right);

  /// Set array of local derivatives
  void SetLocalDerivative(int index, double value)
  {
    fLocalDerivatives[index] = value;
  }

  /// Set array of global derivatives
  void SetGlobalDerivative(int index, double value)
  {
    fGlobalDerivatives[index] = value;
  }

  /// refit track using straight track model
  LocalTrackParam RefitStraightTrack(Track&, double) const;

  void FillDetElemData(const Cluster*);

  void FillRecPointData(const Cluster*);

  void FillTrackParamData(const TrackParam*);

  void LocalEquationX(const double* r);

  void LocalEquationY(const double* r);

  TGeoCombiTrans DeltaTransform(const double* detElemMisAlignment) const;

  bool isMatrixConvertedToAngles(const double* rot, double& psi, double& theta, double& phi) const;

  ///@name utilities
  //@{

  void AddConstraint(double* parameters, double value);

  int GetChamberId(int iDetElemNumber) const;

  bool DetElemIsValid(int iDetElemId) const;

  int GetDetElemNumber(int iDetElemId) const;

  TString GetParameterMaskString(unsigned int parameterMask) const;

  TString GetSidesMaskString(unsigned int sidesMask) const;

  //@}

  /// true when initialized
  bool fInitialized;

  /// current run id
  int fRunNumber;

  /// Flag for Magnetic filed On/Off
  bool fBFieldOn;

  /// true if straight track refit is to be performed
  bool fRefitStraightTracks;

  /// "Encouraged" variation for degrees of freedom
  double fAllowVar[fgNParCh];

  /// Initial value for chi2 cut
  double fStartFac;

  /// Cut on residual for first iteration
  double fResCutInitial;

  /// Cut on residual for other iterations
  double fResCut;

  /// Detector independent alignment class
  o2::fwdalign::MillePede2* fMillepede; // AliMillePede2 implementation

  /// Number of standard deviations for chi2 cut
  int fNStdDev;

  /// Cluster (global) position
  double fClustPos[3];

  /// Track slope at reference point
  double fTrackSlope0[2];

  /// Track slope at current point
  double fTrackSlope[2];

  /// Track intersection at reference point
  double fTrackPos0[3];

  /// Track intersection at current point
  double fTrackPos[3];

  /// Current measurement (depend on B field On/Off)
  double fMeas[2];

  /// Estimated resolution on measurement
  double fSigma[2];

  /// degrees of freedom
  enum {
    kFixedParId = -1,
    kFreeParId = kFixedParId - 1,
    kGroupBaseId = -10
  };

  /// Array of effective degrees of freedom
  /// it is used to fix detectors, parameters, etc.
  std::vector<int> fGlobalParameterStatus;

  /// Array of global derivatives
  std::vector<double> fGlobalDerivatives;

  /// Array of local derivatives
  std::vector<double> fLocalDerivatives;

  /// current detection element number
  int fDetElemNumber;

  /// running Track record
  o2::fwdalign::MillePedeRecord fTrackRecord;

  /// Geometry transformation
  o2::mch::geo::TransformationCreator fTransformCreator;

  /// preform evaluation
  bool fDoEvaluation;

  /// disable record saving
  bool fDisableRecordWriter;

  LocalTrackClusterResidual* fTrkClRes;

  /// output TFile
  TFile* fTFile;

  /// output TTree
  TTree* fTTree;

  /// Option switch for read/write mode
  bool mRead;

  long mNEntriesAutoSave = 10000; ///< number of entries needed to call AutoSave for the output TTrees

  o2::fwdalign::MilleRecordWriter* mRecordWriter;         ///< utility that handles the writing of the data records to a ROOT file
  bool mWithConstraintsRecWriter;                         ///< boolean to be set to true if one wants to also write constaints records
  o2::fwdalign::MilleRecordWriter* mConstraintsRecWriter; ///< utility that handles the writing of the constraints records

  o2::fwdalign::MilleRecordReader* mRecordReader;         ///< utility that handles the reading of the data records from a ROOT file
  bool mWithConstraintsRecReader = false;                 ///< boolean to be set to true if one wants to also read constaints records
  o2::fwdalign::MilleRecordReader* mConstraintsRecReader; ///< utility that handles the reading of the constraints records

}; // class Alignment

} // namespace mch
} // namespace o2
#endif // ALICEO2_MCH_ALIGNER_H_
