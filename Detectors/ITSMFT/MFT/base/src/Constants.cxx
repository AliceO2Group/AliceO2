/// \file Constants.cxx
/// \brief Constants for the Muon Forward Tracker
/// \author antonio.uras@cern.ch

#include "MFTBase/Constants.h"

#include <TString.h>

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::Constants)

// Defaults parameters for track reconstruction
Double_t Constants::fgDiskThicknessInX0[Constants::kNDisks] = {0.008, 0.008, 0.008, 0.008, 0.008};
Double_t Constants::fgPlaneZPos[2*Constants::kNDisks] = {-45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -68.0, -69.4, -76.1, -77.5};


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
