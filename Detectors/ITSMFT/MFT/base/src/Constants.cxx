/// \file Constants.cxx
/// \brief Constants for the Muon Forward Tracker
/// \author antonio.uras@cern.ch

#include "MFTBase/Constants.h"

#include <TString.h>

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Constants)
/// \endcond

// Geometry Related Constants

const Double_t Constants::kSensorLength=3.; //[cm]
const Double_t Constants::kSensorHeight=1.5; //[cm]
const Double_t Constants::kXPixelPitch=29.250e-4; // 29.15 micron // TODO : Check that
const Double_t Constants::kYPixelPitch=26.880e-4; // 26.88 micron // TODO : Check that
const Double_t Constants::kSensorMargin=29.120e-4; // 29.12 micron // TODO : Check that

const Double_t Constants::kSensorActiveWidth  = kNPixelX * kXPixelPitch; //[cm]
const Double_t Constants::kSensorActiveHeight = kNPixelY * kYPixelPitch; //[cm]

const Double_t Constants::kSensorInterspace = 0.01; //[cm]  Offset between two adjacent chip on a ladder
const Double_t Constants::kSensorSideOffset=0.04; // [cm] Side Offset between the ladder edge and the chip edge
const Double_t Constants::kSensorTopOffset=0.04; // [cm] Top Offset between the ladder edge and the chip edge
const Double_t Constants::kLadderOffsetToEnd=4.7; // [cm] Offset between the last Chip of the ladder and the end of the ladder toward the DAQ connector
const Double_t Constants::kSensorThickness=50.e-4; // 50 micron

// Defaults parameters for track reconstruction
Double_t Constants::fgDiskThicknessInX0[kNDisks] = {0.008, 0.008, 0.008, 0.008, 0.008};
Double_t Constants::fgPlaneZPos[2*kNDisks] = {-45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -68.0, -69.4, -76.1, -77.5};

const Double_t Constants::fCutForAvailableDigits = 5.;
const Double_t Constants::fCutForAttachingDigits = 1.;

const Double_t Constants::fElossPerElectron = 3.62e-09;

const Double_t Constants::fActiveSuperposition = 0.05;
                                 
const Double_t Constants::fHeightActive = 1.3;
const Double_t Constants::fHeightReadout = 0.2;

const Double_t Constants::fSupportExtMargin = fHeightReadout + 0.3;

const Double_t Constants::fRadLengthSi = 9.37;

const Double_t Constants::fWidthChip = 1.0;

const Double_t Constants::fPrecisionPointOfClosestApproach = 10.e-4;  // 10 micron

const Double_t Constants::fZEvalKinem = 0.;

const Double_t Constants::fXVertexTolerance = 500.e-4;    // 500 micron
const Double_t Constants::fYVertexTolerance = 500.e-4;    // 500 micron

const Double_t Constants::fPrimaryVertexResX = 5.e-4;   // 5 micron
const Double_t Constants::fPrimaryVertexResY = 5.e-4;   // 5 micron
const Double_t Constants::fPrimaryVertexResZ = 5.e-4;   // 5 micron

const Double_t Constants::fMisalignmentMagnitude = 15.e-4;    // 15 micron

const Double_t Constants::fChipWidth = 3.; // 3 cm ???
const Double_t Constants::fChipThickness=500.e-4; // 50 micron
const Double_t Constants::fMinDistanceLadderFromSupportRMin = 0.1; // 1mm ???

const Double_t Constants::fChipInterspace=500.e-4; // 50um // Offset between two adjacent chip on a ladder
const Double_t Constants::fChipSideOffset=500.e-4; // Side Offset between the ladder edge and the chip edge
const Double_t Constants::fChipTopOffset=500.e-4; // Top Offset between the ladder edge and the chip edge

// Allmost everything you wanted to know about the FPC
const Double_t Constants::kLineWidth= 100.e-4;         // line width, 100 microns
const Double_t Constants::kVarnishThickness= 20.e-4;   // 20 micron
const Double_t Constants::kAluThickness = 25.e-4;      // 25 microns
const Double_t Constants::kKaptonThickness = 75.e-4;   // 75 microns
const Double_t Constants::kFlexThickness = kKaptonThickness + 2*kAluThickness + 2*kVarnishThickness; // total thickness of a FPC
const Double_t Constants::kFlexHeight = 1.68;
const Double_t Constants::kClearance=300.e-4;      // 300 microns clearance without any conducting metal all around the FPC
const Double_t Constants::kRadiusHole1=0.125;      // diameter of the FPC crew, closest to the FPC electric connector
const Double_t Constants::kRadiusHole2=0.1;        // diameter of the FPC pin locator, after the previous hole crew
const Double_t Constants::kHoleShift1=2.8;        // shift of the FPC crew
const Double_t Constants::kHoleShift2=3.6;        // shift of the FPC pin locator
const Double_t Constants::kConnectorOffset=0.4;    // distance between the connector and the start of the FPC
const Double_t Constants::kCapacitorDx=0.05;
const Double_t Constants::kCapacitorDy=0.1;
const Double_t Constants::kCapacitorDz=0.05;
const Double_t Constants::kConnectorLength=0.1; 
const Double_t Constants::kConnectorWidth=0.025;
const Double_t Constants::kConnectorHeight=0.1;
const Double_t Constants::kConnectorThickness=0.01;
const Double_t Constants::kShiftDDGNDline=0.4; // positionning of the line to separate AVDD/DVDD et AGND/DGND on the FPC
const Double_t Constants::kShiftline=0.025; // positionning of the line along the FPC side
const Double_t Constants::kEpsilon=0.0001; // to see the removed volumes produced by TGeoSubtraction

const Double_t Constants::kGlueThickness=50.e-4; // 50 microns
const Double_t Constants::kGlueEdge=300.e-4; // in case the glue is not spreaded on the whole surface of the sensor

