// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <FairLogger.h>

#include "TRDBase/TRDGeometryBase.h"
#include "TRDBase/TRDPadPlane.h"

using namespace o2::trd;

//_____________________________________________________________________________

//
// Dimensions of the detector
//

// Total length of the TRD mother volume
const float TRDGeometryBase::fgkTlength = 751.0;

// Parameter of the super module mother volumes
const float TRDGeometryBase::fgkSheight = 77.9;
const float TRDGeometryBase::fgkSwidth1 = 94.881;
const float TRDGeometryBase::fgkSwidth2 = 122.353;
const float TRDGeometryBase::fgkSlength = 702.0;

// Length of the additional space in front of the supermodule
// used for services
const float TRDGeometryBase::fgkFlength = (TRDGeometryBase::fgkTlength - TRDGeometryBase::fgkSlength) / 2.0;

// The super module side plates
const float TRDGeometryBase::fgkSMpltT = 0.2;

// Vertical spacing of the chambers
const float TRDGeometryBase::fgkVspace = 1.784;
// Horizontal spacing of the chambers
const float TRDGeometryBase::fgkHspace = 2.0;
// Radial distance of the first ROC to the outer plates of the SM
const float TRDGeometryBase::fgkVrocsm = 1.2;

// Height of different chamber parts
// Radiator
const float TRDGeometryBase::fgkCraH = 4.8;
// Drift region
const float TRDGeometryBase::fgkCdrH = 3.0;
// Amplification region
const float TRDGeometryBase::fgkCamH = 0.7;
// Readout
const float TRDGeometryBase::fgkCroH = 2.316;
// Additional width of the readout chamber frames
const float TRDGeometryBase::fgkCroW = 0.9;
// Services on top of ROC
const float TRDGeometryBase::fgkCsvH = TRDGeometryBase::fgkVspace - 0.742;
// Total height (w/o services)
const float TRDGeometryBase::fgkCH =
  TRDGeometryBase::fgkCraH + TRDGeometryBase::fgkCdrH + TRDGeometryBase::fgkCamH + TRDGeometryBase::fgkCroH;
// Total height (with services)

const float TRDGeometryBase::fgkCHsv = TRDGeometryBase::fgkCH + TRDGeometryBase::fgkCsvH;

// Distance of anode wire plane relative to middle of alignable volume
const float TRDGeometryBase::fgkAnodePos =
  TRDGeometryBase::fgkCraH + TRDGeometryBase::fgkCdrH + TRDGeometryBase::fgkCamH / 2.0 - TRDGeometryBase::fgkCHsv / 2.0;

// Thicknesses of different parts of the chamber frame
// Lower aluminum frame
const float TRDGeometryBase::fgkCalT = 0.4;
// Lower Wacosit frame sides
const float TRDGeometryBase::fgkCclsT = 0.21;
// Lower Wacosit frame front
const float TRDGeometryBase::fgkCclfT = 1.0;
// Thickness of glue around radiator
const float TRDGeometryBase::fgkCglT = 0.25;
// Upper Wacosit frame around amplification region
const float TRDGeometryBase::fgkCcuTa = 1.0;
const float TRDGeometryBase::fgkCcuTb = 0.8;
// Al frame of back panel
const float TRDGeometryBase::fgkCauT = 1.5;
// Additional Al ledge at the lower chamber frame
// Actually the dimensions are not realistic, but
// modified in order to allow to mis-alignment.
// The amount of material is, however, correct
const float TRDGeometryBase::fgkCalW = 2.5;
const float TRDGeometryBase::fgkCalH = 0.4;
const float TRDGeometryBase::fgkCalWmod = 0.4;
const float TRDGeometryBase::fgkCalHmod = 2.5;
// Additional Wacosit ledge at the lower chamber frame
const float TRDGeometryBase::fgkCwsW = 1.2;
const float TRDGeometryBase::fgkCwsH = 0.3;

// Difference of outer chamber width and pad plane width
const float TRDGeometryBase::fgkCpadW = 0.0;
const float TRDGeometryBase::fgkRpadW = 1.0;

//
// Thickness of the the material layers
//
const float TRDGeometryBase::fgkDrThick = TRDGeometryBase::fgkCdrH;
const float TRDGeometryBase::fgkAmThick = TRDGeometryBase::fgkCamH;
const float TRDGeometryBase::fgkXeThick = TRDGeometryBase::fgkDrThick + TRDGeometryBase::fgkAmThick;
const float TRDGeometryBase::fgkWrThick = 0.00011;

const float TRDGeometryBase::fgkRMyThick = 0.0015;
const float TRDGeometryBase::fgkRCbThick = 0.0055;
const float TRDGeometryBase::fgkRGlThick = 0.0065;
const float TRDGeometryBase::fgkRRhThick = 0.8;
const float TRDGeometryBase::fgkRFbThick = fgkCraH - 2.0 * (fgkRMyThick + fgkRCbThick + fgkRRhThick);

const float TRDGeometryBase::fgkPPdThick = 0.0025;
const float TRDGeometryBase::fgkPPpThick = 0.0356;
const float TRDGeometryBase::fgkPGlThick = 0.1428;
const float TRDGeometryBase::fgkPCbThick = 0.019;
const float TRDGeometryBase::fgkPPcThick = 0.0486;
const float TRDGeometryBase::fgkPRbThick = 0.0057;
const float TRDGeometryBase::fgkPElThick = 0.0029;
const float TRDGeometryBase::fgkPHcThick =
  fgkCroH - fgkPPdThick - fgkPPpThick - fgkPGlThick - fgkPCbThick * 2.0 - fgkPPcThick - fgkPRbThick - fgkPElThick;

//
// Position of the material layers
//
const float TRDGeometryBase::fgkDrZpos = 2.4;
const float TRDGeometryBase::fgkAmZpos = 0.0;
const float TRDGeometryBase::fgkWrZposA = 0.0;
const float TRDGeometryBase::fgkWrZposB = -fgkAmThick / 2.0 + 0.001;
const float TRDGeometryBase::fgkCalZpos = 0.3;

const int TRDGeometryBase::fgkMCMmax = 16;
const int TRDGeometryBase::fgkMCMrow = 4;
const int TRDGeometryBase::fgkROBmaxC0 = 6;
const int TRDGeometryBase::fgkROBmaxC1 = 8;
const int TRDGeometryBase::fgkADCmax = 21;
const int TRDGeometryBase::fgkTBmax = 60;
const int TRDGeometryBase::fgkPadmax = 18;
const int TRDGeometryBase::fgkColmax = 144;
const int TRDGeometryBase::fgkRowmaxC0 = 12;
const int TRDGeometryBase::fgkRowmaxC1 = 16;

const double TRDGeometryBase::fgkTime0Base = 300.65;
const float TRDGeometryBase::fgkTime0[6] = { static_cast<float>(fgkTime0Base + 0 * (Cheight() + Cspace())),
                                         static_cast<float>(fgkTime0Base + 1 * (Cheight() + Cspace())),
                                         static_cast<float>(fgkTime0Base + 2 * (Cheight() + Cspace())),
                                         static_cast<float>(fgkTime0Base + 3 * (Cheight() + Cspace())),
                                         static_cast<float>(fgkTime0Base + 4 * (Cheight() + Cspace())),
                                         static_cast<float>(fgkTime0Base + 5 * (Cheight() + Cspace())) };

const double TRDGeometryBase::fgkXtrdBeg = 288.43; // Values depend on position of TRD
const double TRDGeometryBase::fgkXtrdEnd = 366.33; // mother volume inside space frame !!!

// The outer width of the chambers
const float TRDGeometryBase::fgkCwidth[kNlayer] = { 90.4, 94.8, 99.3, 103.7, 108.1, 112.6 };

// The outer lengths of the chambers
// Includes the spacings between the chambers!
const float TRDGeometryBase::fgkClength[kNlayer][kNstack] = {
  { 124.0, 124.0, 110.0, 124.0, 124.0 }, { 124.0, 124.0, 110.0, 124.0, 124.0 }, { 131.0, 131.0, 110.0, 131.0, 131.0 },
  { 138.0, 138.0, 110.0, 138.0, 138.0 }, { 145.0, 145.0, 110.0, 145.0, 145.0 }, { 147.0, 147.0, 110.0, 147.0, 147.0 }
};

char TRDGeometryBase::fgSMstatus[kNsector] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

//_____________________________________________________________________________
TRDGeometryBase::TRDGeometryBase()
{
  //
  // TRDGeometry default constructor
  //
}

//_____________________________________________________________________________
TRDGeometryBase::~TRDGeometryBase()
{
  //
  // TRDGeometry destructor
  //
}

//_____________________________________________________________________________
int TRDGeometryBase::GetDetectorSec(int layer, int stack)
{
  //
  // Convert plane / stack into detector number for one single sector
  //

  return (layer + stack * kNlayer);
}

//_____________________________________________________________________________
int TRDGeometryBase::GetDetector(int layer, int stack, int sector)
{
  //
  // Convert layer / stack / sector into detector number
  //

  return (layer + stack * kNlayer + sector * kNlayer * kNstack);
}

//_____________________________________________________________________________
int TRDGeometryBase::GetLayer(int det)
{
  //
  // Reconstruct the layer number from the detector number
  //

  return ((int)(det % kNlayer));
}

//_____________________________________________________________________________
int TRDGeometryBase::GetStack(int det) const
{
  //
  // Reconstruct the stack number from the detector number
  //

  return ((int)(det % (kNlayer * kNstack)) / kNlayer);
}

//_____________________________________________________________________________
int TRDGeometryBase::GetStack(double z, int layer) const
{
  //
  // Reconstruct the chamber number from the z position and layer number
  //
  // The return function has to be protected for positiveness !!
  //

  if ((layer < 0) || (layer >= kNlayer))
    return -1;

  int istck = kNstack;
  double zmin = 0.0;
  double zmax = 0.0;

  do {
    istck--;
    if (istck < 0)
      break;
    TRDPadPlane* pp = GetPadPlane(layer, istck);
    zmax = pp->GetRow0();
    int nrows = pp->GetNrows();
    zmin = zmax - 2 * pp->GetLengthOPad() - (nrows - 2) * pp->GetLengthIPad() - (nrows - 1) * pp->GetRowSpacing();
  } while ((z < zmin) || (z > zmax));

  return istck;
}

//_____________________________________________________________________________
int TRDGeometryBase::GetSector(int det)
{
  //
  // Reconstruct the sector number from the detector number
  //

  return ((int)(det / (fgkNlayer * fgkNstack)));
}

//_____________________________________________________________________________
TRDPadPlane* TRDGeometryBase::GetPadPlane(int layer, int stack) const
{
  //
  // Returns the pad plane for a given plane <pl> and stack <st> number
  //

  int ipp = GetDetectorSec(layer, stack);
  return &mPadPlaneArray[ipp];
}

//_____________________________________________________________________________
int TRDGeometryBase::GetRowMax(int layer, int stack, int /*sector*/) const
{
  //
  // Returns the number of rows on the pad plane
  //

  return GetPadPlane(layer, stack)->GetNrows();
}

//_____________________________________________________________________________
int TRDGeometryBase::GetColMax(int layer) const
{
  //
  // Returns the number of rows on the pad plane
  //

  return GetPadPlane(layer, 0)->GetNcols();
}

//_____________________________________________________________________________
double TRDGeometryBase::GetRow0(int layer, int stack, int /*sector*/) const
{
  //
  // Returns the position of the border of the first pad in a row
  //

  return GetPadPlane(layer, stack)->GetRow0();
}

//_____________________________________________________________________________
double TRDGeometryBase::GetCol0(int layer) const
{
  //
  // Returns the position of the border of the first pad in a column
  //

  return GetPadPlane(layer, 0)->GetCol0();
}

//_____________________________________________________________________________
bool TRDGeometryBase::IsHole(int /*la*/, int st, int se) const
{
  //
  // Checks for holes in front of PHOS
  //

  if (((se == 13) || (se == 14) || (se == 15)) && (st == 2)) {
    return true;
  }

  return false;
}

//_____________________________________________________________________________
bool TRDGeometryBase::IsOnBoundary(int det, float y, float z, float eps) const
{
  //
  // Checks whether position is at the boundary of the sensitive volume
  //

  int ly = GetLayer(det);
  if ((ly < 0) || (ly >= kNlayer))
    return true;

  int stk = GetStack(det);
  if ((stk < 0) || (stk >= kNstack))
    return true;

  TRDPadPlane* pp = &mPadPlaneArray[GetDetectorSec(ly, stk)];
  if (!pp)
    return true;

  double max = pp->GetRow0();
  int n = pp->GetNrows();
  double min = max - 2 * pp->GetLengthOPad() - (n - 2) * pp->GetLengthIPad() - (n - 1) * pp->GetRowSpacing();
  if (z < min + eps || z > max - eps) {
    // printf("z : min[%7.2f (%7.2f)] %7.2f max[(%7.2f) %7.2f]\n", min, min+eps, z, max-eps, max);
    return true;
  }
  min = pp->GetCol0();
  n = pp->GetNcols();
  max = min + 2 * pp->GetWidthOPad() + (n - 2) * pp->GetWidthIPad() + (n - 1) * pp->GetColSpacing();
  if (y < min + eps || y > max - eps) {
    // printf("y : min[%7.2f (%7.2f)] %7.2f max[(%7.2f) %7.2f]\n", min, min+eps, y, max-eps, max);
    return true;
  }

  return false;
}

ClassImp(TRDGeometryBase)
