/// \file Constants.cxx
/// \brief Constants for the Muon Forward Tracker
/// \author antonio.uras@cern.ch

#include "MFTBase/Constants.h"

#include <TString.h>

using namespace o2::MFT;

ClassImp(o2::MFT::Constants)

// Defaults parameters for track reconstruction
Double_t Constants::sDiskThicknessInX0[Constants::sNDisks] = {0.008, 0.008, 0.008, 0.008, 0.008};
Double_t Constants::sPlaneZPos[2*Constants::sNDisks] = {-45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -68.0, -69.4, -76.1, -77.5};


const Double_t Constants::sCutForAvailableDigits = 5.;
const Double_t Constants::sCutForAttachingDigits = 1.;

const Double_t Constants::sElossPerElectron = 3.62e-09;

const Double_t Constants::sActiveSuperposition = 0.05;
                                 
const Double_t Constants::sHeightActive = 1.3;
const Double_t Constants::sHeightReadout = 0.2;

const Double_t Constants::sSupportExtMargin = sHeightReadout + 0.3;

const Double_t Constants::sRadLengthSi = 9.37;

const Double_t Constants::sWidthChip = 1.0;

const Double_t Constants::sPrecisionPointOfClosestApproach = 10.e-4;  // 10 micron

const Double_t Constants::sZEvalKinem = 0.;

const Double_t Constants::sXVertexTolerance = 500.e-4;    // 500 micron
const Double_t Constants::sYVertexTolerance = 500.e-4;    // 500 micron

const Double_t Constants::sPrimaryVertexResX = 5.e-4;   // 5 micron
const Double_t Constants::sPrimaryVertexResY = 5.e-4;   // 5 micron
const Double_t Constants::sPrimaryVertexResZ = 5.e-4;   // 5 micron

const Double_t Constants::sMisalignmentMagnitude = 15.e-4;    // 15 micron

const Double_t Constants::sChipWidth = 3.; // 3 cm ???
const Double_t Constants::sChipThickness=500.e-4; // 50 micron
const Double_t Constants::sMinDistanceLadderFromSupportRMin = 0.1; // 1mm ???

const Double_t Constants::sChipInterspace=500.e-4; // 50um // Offset between two adjacent chip on a ladder
const Double_t Constants::sChipSideOffset=500.e-4; // Side Offset between the ladder edge and the chip edge
const Double_t Constants::sChipTopOffset=500.e-4; // Top Offset between the ladder edge and the chip edge
