/// \file Constants.h
/// \brief Constants for the MFT; distance unit is cm
/// \author Antonio Uras <antonio.uras@cern.ch>

#ifndef ALICEO2_MFT_CONSTANTS_H_
#define ALICEO2_MFT_CONSTANTS_H_

#include "TObject.h"

namespace o2 {
namespace MFT {

class Constants : public TObject {

public:

  // Geometry
  static const Int_t sNDisks = 5;             ///< \brief Number of Disk

  /// Return Disk thickness in X0
  static Double_t diskThicknessInX0(Int_t Id) { return (Id >= 0 && Id < sNDisks) ? sDiskThicknessInX0[Id] : 0.; }
  
  /// Return Plane Z position
  static Double_t defaultPlaneZ(Int_t Id) { return (Id >= 0 && Id < sNDisks*2) ? sPlaneZPos[Id]+(-(Id%2)*2-1)*0.0025 : 0.; } // Move to the middle of the CMOS sensor in Z direction

  static const Int_t sNMaxPlanes = 20;

  static const Int_t sNMaxDigitsPerCluster = 50;  ///< max number of digits per cluster
  static const Double_t sCutForAvailableDigits;   ///<
  static const Double_t sCutForAttachingDigits;   ///<

  static const Int_t sNMaxMCTracksPerCluster = 10;   ///< max number of MC tracks sharing the same MFT cluster
  static const Int_t sNMaxMCTracksPerDigit = 3;      ///< max number of MC tracks sharing the same MFT digit

  static const Double_t sElossPerElectron;

  // superposition between the active elements tasselling the MFT planes, for having a full acceptance coverage even in case of 10 degrees inclined tracks
  static const Double_t sActiveSuperposition;  ///<
                                                
  static const Double_t sHeightActive;   ///< height of the active elements
  static const Double_t sHeightReadout;  ///< height of the readout elements attached to the active ones
	 
  // minimum border size between the end of the support plane and the sensors: fHeightReadout + 0.3
  static const Double_t sSupportExtMargin;  ///<

  static const Int_t sNMaxDetElemPerPlane = 1000;  ///<

  static const Double_t sRadLengthSi;    ///< expressed in cm

  static const Double_t sWidthChip;      ///< expressed in cm

  static const Double_t sPrecisionPointOfClosestApproach;  ///< precision (along z) for the research of the point of closest approach for a dimuon

  static const Double_t sZEvalKinem;     // z coordinate at which the kinematics is evaluated for the ESD and AOD tracks

  static const Double_t sXVertexTolerance;   // tolerance on the vertex for the first extrapolation of MUON tracks to I.P.
  static const Double_t sYVertexTolerance;   // tolerance on the vertex for the first extrapolation of MUON tracks to I.P.

  static const Double_t sPrimaryVertexResX;   // expected res. in Pb-Pb for the prim vtx from ITS+MFT combined vertexing (should not be used, depends on contributors)
  static const Double_t sPrimaryVertexResY;   // expected res. in Pb-Pb for the prim vtx from ITS+MFT combined vertexing (should not be used, depends on contributors)
  static const Double_t sPrimaryVertexResZ;   // expected res. in Pb-Pb for the prim vtx from ITS+MFT combined vertexing (should not be used, depends on contributors)

  static const Double_t sMisalignmentMagnitude;   // Expected misalignment magnitude (for MC, waiting for OCDB)

  static const Int_t sNMaxMuonsForPCA = 10;
  static const Int_t sNMaxPileUpEvents = 5;         // Max events to consider for pile-up studies
  static const Int_t sLabelOffsetMC = 10000000;     // Offset to be added to MC labels of tracks from merged underlying and pile-up events
  static const Int_t sNMaxLaddersPerPlane = 20;
  static const Int_t sNMaxChipsPerLadder = 5;
  static const Int_t sNMFTHalves = 2;
  static const Double_t sChipWidth;
  static const Double_t sMinDistanceLadderFromSupportRMin;
  static const Double_t sChipThickness;
  static const Double_t sChipInterspace; // Offset between two adjacent chip on a ladder
  static const Double_t sChipSideOffset; // Side Offset between the ladder edge and the chip edge
  static const Double_t sChipTopOffset; // Top Offset between the ladder edge and the chip edge

protected:

  Constants() : TObject() {}
  ~Constants() override = default;

  static Double_t  sDiskThicknessInX0[sNDisks]; ///< default disk thickness in X0 for reconstruction
  static Double_t  sPlaneZPos[2*sNDisks]; ///< default Plane Z position for reconstruction

  ClassDefOverride(Constants, 1);

};

}
}

#endif
