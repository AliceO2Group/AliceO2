// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TMath.h>
#include <TVirtualMC.h>
#include <FairLogger.h>

#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Utils.h" // utils::bit2Mask
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"

using namespace o2::trd;

//_____________________________________________________________________________

const o2::detectors::DetID TRDGeometry::sDetID(o2::detectors::DetID::TRD);

//_____________________________________________________________________________
TRDGeometry::TRDGeometry() : TRDGeometryBase(), o2::detectors::DetMatrixCacheIndirect(sDetID)
{
  createPadPlaneArray();
}

//_____________________________________________________________________________
bool TRDGeometry::rotateBack(int det, const float* const loc, float* glb) const
{
  //
  // Rotates a chambers to transform the corresponding local frame
  // coordinates <loc> into the coordinates of the ALICE restframe <glb>.
  //

  int sector = getSector(det);
  float phi = 2.0 * TMath::Pi() / (float)kNsector * ((float)sector + 0.5);

  glb[0] = loc[0] * TMath::Cos(phi) - loc[1] * TMath::Sin(phi);
  glb[1] = loc[0] * TMath::Sin(phi) + loc[1] * TMath::Cos(phi);
  glb[2] = loc[2];

  return true;
}

//_____________________________________________________________________________
void TRDGeometry::createPadPlaneArray()
{
  //
  // Creates the array of TRDPadPlane objects
  //

  for (int ilayer = 0; ilayer < kNlayer; ilayer++) {
    for (int istack = 0; istack < kNstack; istack++) {
      createPadPlane(ilayer, istack);
    }
  }
}

//_____________________________________________________________________________
void TRDGeometry::createPadPlane(int ilayer, int istack)
{
  //
  // Creates an TRDPadPlane object
  //
  int ipp = getDetectorSec(ilayer, istack);
  auto& padPlane = mPadPlanes[ipp];

  padPlane.setLayer(ilayer);
  padPlane.setStack(istack);

  padPlane.setRowSpacing(0.0);
  padPlane.setColSpacing(0.0);

  padPlane.setLengthRim(1.0);
  padPlane.setWidthRim(0.5);

  padPlane.setNcols(144);

  padPlane.setAnodeWireOffset(0.25);

  //
  // The pad plane parameter
  //
  const float kTiltAngle = 2.0;
  switch (ilayer) {
    case 0:
      if (istack == 2) {
        // L0C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
        padPlane.setLengthIPad(9.0);
      } else {
        // L0C1 type
        padPlane.setNrows(16);
        padPlane.setLength(122.0);
        padPlane.setLengthOPad(7.5);
        padPlane.setLengthIPad(7.5);
      }
      padPlane.setWidth(92.2);
      padPlane.setWidthOPad(0.515);
      padPlane.setWidthIPad(0.635);
      padPlane.setTiltingAngle(-kTiltAngle);
      break;
    case 1:
      if (istack == 2) {
        // L1C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
        padPlane.setLengthIPad(9.0);
      } else {
        // L1C1 type
        padPlane.setNrows(16);
        padPlane.setLength(122.0);
        padPlane.setLengthOPad(7.5);
        padPlane.setLengthIPad(7.5);
      }
      padPlane.setWidth(96.6);
      padPlane.setWidthOPad(0.585);
      padPlane.setWidthIPad(0.665);
      padPlane.setTiltingAngle(kTiltAngle);
      break;
    case 2:
      if (istack == 2) {
        // L2C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
        padPlane.setLengthIPad(9.0);
      } else {
        // L2C1 type
        padPlane.setNrows(16);
        padPlane.setLength(129.0);
        padPlane.setLengthOPad(7.5);
        padPlane.setLengthIPad(8.0);
      }
      padPlane.setWidth(101.1);
      padPlane.setWidthOPad(0.705);
      padPlane.setWidthIPad(0.695);
      padPlane.setTiltingAngle(-kTiltAngle);
      break;
    case 3:
      if (istack == 2) {
        // L3C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
        padPlane.setLengthIPad(9.0);
      } else {
        // L3C1 type
        padPlane.setNrows(16);
        padPlane.setLength(136.0);
        padPlane.setLengthOPad(7.5);
        padPlane.setLengthIPad(8.5);
      }
      padPlane.setWidth(105.5);
      padPlane.setWidthOPad(0.775);
      padPlane.setWidthIPad(0.725);
      padPlane.setTiltingAngle(kTiltAngle);
      break;
    case 4:
      if (istack == 2) {
        // L4C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
      } else {
        // L4C1 type
        padPlane.setNrows(16);
        padPlane.setLength(143.0);
        padPlane.setLengthOPad(7.5);
      }
      padPlane.setWidth(109.9);
      padPlane.setWidthOPad(0.845);
      padPlane.setLengthIPad(9.0);
      padPlane.setWidthIPad(0.755);
      padPlane.setTiltingAngle(-kTiltAngle);
      break;
    case 5:
      if (istack == 2) {
        // L5C0 type
        padPlane.setNrows(12);
        padPlane.setLength(108.0);
        padPlane.setLengthOPad(8.0);
      } else {
        // L5C1 type
        padPlane.setNrows(16);
        padPlane.setLength(145.0);
        padPlane.setLengthOPad(8.5);
      }
      padPlane.setWidth(114.4);
      padPlane.setWidthOPad(0.965);
      padPlane.setLengthIPad(9.0);
      padPlane.setWidthIPad(0.785);
      padPlane.setTiltingAngle(kTiltAngle);
      break;
  };

  //
  // The positions of the borders of the pads
  //
  // Row direction
  //
  float row = CLENGTH[ilayer][istack] / 2.0 - RPADW - padPlane.getLengthRim();
  for (int ir = 0; ir < padPlane.getNrows(); ir++) {
    padPlane.setPadRow(ir, row);
    row -= padPlane.getRowSpacing();
    if (ir == 0) {
      row -= padPlane.getLengthOPad();
    } else {
      row -= padPlane.getLengthIPad();
    }
  }
  //
  // Column direction
  //
  float col = -CWIDTH[ilayer] / 2.0 - CROW + padPlane.getWidthRim();
  for (int ic = 0; ic < padPlane.getNcols(); ic++) {
    padPlane.setPadCol(ic, col);
    col += padPlane.getColSpacing();
    if (ic == 0) {
      col += padPlane.getWidthOPad();
    } else {
      col += padPlane.getWidthIPad();
    }
  }
  // Calculate the offset to translate from the local ROC system into
  // the local supermodule system, which is used for clusters
  float rowTmp = CLENGTH[ilayer][0] + CLENGTH[ilayer][1] + CLENGTH[ilayer][2] / 2.0;
  for (int jstack = 0; jstack < istack; jstack++) {
    rowTmp -= CLENGTH[ilayer][jstack];
  }
  padPlane.setPadRowSMOffset(rowTmp - CLENGTH[ilayer][istack] / 2.0);
}

void TRDGeometry::createVolume(const char* name, const char* shape, int nmed, float* upar, int np)
{
  TVirtualMC::GetMC()->Gsvolu(name, shape, nmed, upar, np);

  // add to sensitive volumes for TRD if matching criterion
  // these are coded with J+K in the second character (according to AliTRDv1.cxx of AliROOT)
  if (name[1] == 'J' || name[1] == 'K') {
    mSensitiveVolumeNames.emplace_back(name);
  }
}

//_____________________________________________________________________________
void TRDGeometry::createGeometry(std::vector<int> const& idtmed)
{
  //
  // Create the TRD geometry

  if (!gGeoManager) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(FATAL) << "Geometry is not loaded";
  }

  createVolumes(idtmed);
}

void TRDGeometry::createVolumes(std::vector<int> const& idtmed)
{
  //
  // Create the TRD geometry volumes
  //
  //
  // Names of the TRD volumina (xx = detector number):
  //
  //   Volume (Air) wrapping the readout chamber components
  //     UTxx    includes: UAxx, UDxx, UFxx, UUxx
  //
  //   Lower part of the readout chambers (drift volume + radiator)
  //     UAxx    Aluminum frames                (Al)
  //
  //   Upper part of the readout chambers (readout plane + fee)
  //     UDxx    Wacosit frames of amp. region  (Wacosit)
  //     UFxx    Aluminum frame of back panel   (Al)
  //
  //   Services on chambers (cooling, cables, MCMs, DCS boards, ...)
  //     UUxx    Volume containing the services (Air)
  //
  //   Material layers inside sensitive area:
  //     Name    Description                     Mat.      Thick.   Dens.    Radl.    X/X_0
  //
  //     URMYxx  Mylar layers (x2)               Mylar     0.0015   1.39     28.5464  0.005%
  //     URCBxx  Carbon layer (x2)               Carbon    0.0055   1.75     24.2824  0.023%
  //     URGLxx  Glue on the carbon layers (x2)  Araldite  0.0065   1.12     37.0664  0.018%
  //     URRHxx  Rohacell layer (x2)             Rohacell  0.8      0.075    536.005  0.149%
  //     URFBxx  Fiber mat layer                 PP        3.186    0.068    649.727  0.490%
  //
  //     UJxx    Drift region                    Xe/CO2    3.0      0.00495  1792.37  0.167%
  //     UKxx    Amplification region            Xe/CO2    0.7      0.00495  1792.37  0.039%
  //     UWxx    Wire planes (x2)                Copper    0.00011  8.96     1.43503  0.008%
  //
  //     UPPDxx  Copper of pad plane             Copper    0.0025   8.96     1.43503  0.174%
  //     UPPPxx  PCB of pad plane                G10       0.0356   2.0      14.9013  0.239%
  //     UPGLxx  Glue on pad planes              Araldite  0.0923   1.12     37.0664  0.249%
  //             + add. glue (ca. 600g)          Araldite  0.0505   1.12     37.0663  0.107%
  //     UPCBxx  Carbon fiber mats (x2)          Carbon    0.019    1.75     24.2824  0.078%
  //     UPHCxx  Honeycomb structure             Aramide   2.0299   0.032    1198.84  0.169%
  //     UPPCxx  PCB of readout board            G10       0.0486   2.0      14.9013  0.326%
  //     UPRDxx  Copper of readout board         Copper    0.0057   8.96     1.43503  0.404%
  //     UPELxx  Electronics + cables            Copper    0.0029   8.96     1.43503  0.202%
  //

  const int kNparTrd = 4;
  const int kNparCha = 3;

  float xpos;
  float ypos;
  float zpos;

  float parTrd[kNparTrd];
  float parCha[kNparCha];

  const int kTag = 100;
  char cTagV[kTag];
  char cTagM[kTag];

  // There are three TRD volumes for the supermodules in order to accomodate
  // the different arrangements in front of PHOS
  // UTR1: Default supermodule
  // UTR2: Supermodule in front of PHOS with double carbon cover
  // UTR3: As UTR2, but w/o middle stack
  // UTR4: Sector 17 with missing chamber L4S4
  //
  // The mother volume for one sector (Air), full length in z-direction
  // Provides material for side plates of super module
  parTrd[0] = SWIDTH1 / 2.0;
  parTrd[1] = SWIDTH2 / 2.0;
  parTrd[2] = SLENGTH / 2.0;
  parTrd[3] = SHEIGHT / 2.0;
  createVolume("UTR1", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTR2", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTR3", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTR4", "TRD1", idtmed[2], parTrd, kNparTrd);
  // The outer aluminum plates of the super module (Al)
  parTrd[0] = SWIDTH1 / 2.0;
  parTrd[1] = SWIDTH2 / 2.0;
  parTrd[2] = SLENGTH / 2.0;
  parTrd[3] = SHEIGHT / 2.0;
  createVolume("UTS1", "TRD1", idtmed[1], parTrd, kNparTrd);
  createVolume("UTS2", "TRD1", idtmed[1], parTrd, kNparTrd);
  createVolume("UTS3", "TRD1", idtmed[1], parTrd, kNparTrd);
  createVolume("UTS4", "TRD1", idtmed[1], parTrd, kNparTrd);
  // The inner part of the TRD mother volume for one sector (Air),
  // full length in z-direction
  parTrd[0] = SWIDTH1 / 2.0 - SMPLTT;
  parTrd[1] = SWIDTH2 / 2.0 - SMPLTT;
  parTrd[2] = SLENGTH / 2.0;
  parTrd[3] = SHEIGHT / 2.0 - SMPLTT;
  createVolume("UTI1", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTI2", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTI3", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTI4", "TRD1", idtmed[2], parTrd, kNparTrd);

  // The inner part of the TRD mother volume for services in front
  // of the supermodules  (Air),
  parTrd[0] = SWIDTH1 / 2.0;
  parTrd[1] = SWIDTH2 / 2.0;
  parTrd[2] = FLENGTH / 2.0;
  parTrd[3] = SHEIGHT / 2.0;
  createVolume("UTF1", "TRD1", idtmed[2], parTrd, kNparTrd);
  createVolume("UTF2", "TRD1", idtmed[2], parTrd, kNparTrd);

  for (int istack = 0; istack < kNstack; istack++) {
    for (int ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);

      // The lower part of the readout chambers (drift volume + radiator)
      // The aluminum frames
      snprintf(cTagV, kTag, "UA%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parCha[2] = CRAH / 2.0 + CDRH / 2.0;
      createVolume(cTagV, "BOX ", idtmed[1], parCha, kNparCha);
      // The additional aluminum on the frames
      // This part has not the correct shape but is just supposed to
      // represent the missing material. The correct form of the L-shaped
      // profile would not fit into the alignable volume.
      snprintf(cTagV, kTag, "UZ%02d", iDet);
      parCha[0] = CALWMOD / 2.0;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parCha[2] = CALHMOD / 2.0;
      createVolume(cTagV, "BOX ", idtmed[1], parCha, kNparCha);
      // The additional Wacosit on the frames
      snprintf(cTagV, kTag, "UP%02d", iDet);
      parCha[0] = CWSW / 2.0;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parCha[2] = CWSH / 2.0;
      createVolume(cTagV, "BOX ", idtmed[7], parCha, kNparCha);
      // The Wacosit frames
      snprintf(cTagV, kTag, "UB%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 - CALT;
      parCha[1] = -1.0;
      parCha[2] = -1.0;
      createVolume(cTagV, "BOX ", idtmed[7], parCha, kNparCha);
      // The glue around the radiator
      snprintf(cTagV, kTag, "UX%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 - CALT - CCLST;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0 - CCLFT;
      parCha[2] = CRAH / 2.0;
      createVolume(cTagV, "BOX ", idtmed[11], parCha, kNparCha);
      // The inner part of radiator (air)
      snprintf(cTagV, kTag, "UC%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 - CALT - CCLST - CGLT;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0 - CCLFT - CGLT;
      parCha[2] = -1.0;
      createVolume(cTagV, "BOX ", idtmed[2], parCha, kNparCha);

      // The upper part of the readout chambers (amplification volume)
      // The Wacosit frames
      snprintf(cTagV, kTag, "UD%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 + CROW;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parCha[2] = CAMH / 2.0;
      createVolume(cTagV, "BOX ", idtmed[7], parCha, kNparCha);
      // The inner part of the Wacosit frame (air)
      snprintf(cTagV, kTag, "UE%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 + CROW - CCUTB;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0 - CCUTA;
      parCha[2] = -1.;
      createVolume(cTagV, "BOX ", idtmed[2], parCha, kNparCha);

      // The back panel, including pad plane and readout boards
      // The aluminum frames
      snprintf(cTagV, kTag, "UF%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 + CROW;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parCha[2] = CROH / 2.0;
      createVolume(cTagV, "BOX ", idtmed[1], parCha, kNparCha);
      // The inner part of the aluminum frames
      snprintf(cTagV, kTag, "UG%02d", iDet);
      parCha[0] = CWIDTH[ilayer] / 2.0 + CROW - CAUT;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0 - CAUT;
      parCha[2] = -1.0;
      createVolume(cTagV, "BOX ", idtmed[2], parCha, kNparCha);

      //
      // The material layers inside the chambers
      //

      // Mylar layer (radiator)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = RMYTHICK / 2.0;
      snprintf(cTagV, kTag, "URMY%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[27], parCha, kNparCha);
      // Carbon layer (radiator)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = RCBTHICK / 2.0;
      snprintf(cTagV, kTag, "URCB%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[26], parCha, kNparCha);
      // Araldite layer (radiator)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = RGLTHICK / 2.0;
      snprintf(cTagV, kTag, "URGL%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[11], parCha, kNparCha);
      // Rohacell layer (radiator)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = RRHTHICK / 2.0;
      snprintf(cTagV, kTag, "URRH%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[15], parCha, kNparCha);
      // Fiber layer (radiator)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = RFBTHICK / 2.0;
      snprintf(cTagV, kTag, "URFB%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[28], parCha, kNparCha);

      // Xe/Isobutane layer (drift volume)
      parCha[0] = CWIDTH[ilayer] / 2.0 - CALT - CCLST;
      parCha[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0 - CCLFT;
      parCha[2] = DRTHICK / 2.0;
      snprintf(cTagV, kTag, "UJ%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[9], parCha, kNparCha);

      // Xe/Isobutane layer (amplification volume)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = AMTHICK / 2.0;
      snprintf(cTagV, kTag, "UK%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[9], parCha, kNparCha);
      // Cu layer (wire plane)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = WRTHICK / 2.0;
      snprintf(cTagV, kTag, "UW%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[3], parCha, kNparCha);

      // Cu layer (pad plane)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PPDTHICK / 2.0;
      snprintf(cTagV, kTag, "UPPD%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[5], parCha, kNparCha);
      // G10 layer (pad plane)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PPPTHICK / 2.0;
      snprintf(cTagV, kTag, "UPPP%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[13], parCha, kNparCha);
      // Araldite layer (glue)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PGLTHICK / 2.0;
      snprintf(cTagV, kTag, "UPGL%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[11], parCha, kNparCha);
      // Carbon layer (carbon fiber mats)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PCBTHICK / 2.0;
      snprintf(cTagV, kTag, "UPCB%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[26], parCha, kNparCha);
      // Aramide layer (honeycomb)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PHCTHICK / 2.0;
      snprintf(cTagV, kTag, "UPHC%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[10], parCha, kNparCha);
      // G10 layer (PCB readout board)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PPCTHICK / 2;
      snprintf(cTagV, kTag, "UPPC%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[13], parCha, kNparCha);
      // Cu layer (traces in readout board)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PRBTHICK / 2.0;
      snprintf(cTagV, kTag, "UPRB%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[6], parCha, kNparCha);
      // Cu layer (other material on in readout board, incl. screws)
      parCha[0] = -1.0;
      parCha[1] = -1.0;
      parCha[2] = PELTHICK / 2.0;
      snprintf(cTagV, kTag, "UPEL%02d", iDet);
      createVolume(cTagV, "BOX ", idtmed[4], parCha, kNparCha);

      //
      // Position the layers in the chambers
      //
      xpos = 0.0;
      ypos = 0.0;

      // Lower part
      // Mylar layers (radiator)
      zpos = RMYTHICK / 2.0 - CRAH / 2.0;
      snprintf(cTagV, kTag, "URMY%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = -RMYTHICK / 2.0 + CRAH / 2.0;
      snprintf(cTagV, kTag, "URMY%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Carbon layers (radiator)
      zpos = RCBTHICK / 2.0 + RMYTHICK - CRAH / 2.0;
      snprintf(cTagV, kTag, "URCB%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = -RCBTHICK / 2.0 - RMYTHICK + CRAH / 2.0;
      snprintf(cTagV, kTag, "URCB%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Carbon layers (radiator)
      zpos = RGLTHICK / 2.0 + RCBTHICK + RMYTHICK - CRAH / 2.0;
      snprintf(cTagV, kTag, "URGL%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = -RGLTHICK / 2.0 - RCBTHICK - RMYTHICK + CRAH / 2.0;
      snprintf(cTagV, kTag, "URGL%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Rohacell layers (radiator)
      zpos = RRHTHICK / 2.0 + RGLTHICK + RCBTHICK + RMYTHICK - CRAH / 2.0;
      snprintf(cTagV, kTag, "URRH%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = -RRHTHICK / 2.0 - RGLTHICK - RCBTHICK - RMYTHICK + CRAH / 2.0;
      snprintf(cTagV, kTag, "URRH%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Fiber layers (radiator)
      zpos = 0.0;
      snprintf(cTagV, kTag, "URFB%02d", iDet);
      snprintf(cTagM, kTag, "UC%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // Xe/Isobutane layer (drift volume)
      zpos = DRZPOS;
      snprintf(cTagV, kTag, "UJ%02d", iDet);
      snprintf(cTagM, kTag, "UB%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // Upper part
      // Xe/Isobutane layer (amplification volume)
      zpos = AMZPOS;
      snprintf(cTagV, kTag, "UK%02d", iDet);
      snprintf(cTagM, kTag, "UE%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Cu layer (wire planes inside amplification volume)
      zpos = WRZPOSA;
      snprintf(cTagV, kTag, "UW%02d", iDet);
      snprintf(cTagM, kTag, "UK%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = WRZPOSB;
      snprintf(cTagV, kTag, "UW%02d", iDet);
      snprintf(cTagM, kTag, "UK%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // Back panel + pad plane + readout part
      // Cu layer (pad plane)
      zpos = PPDTHICK / 2.0 - CROH / 2.0;
      snprintf(cTagV, kTag, "UPPD%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // G10  layer (pad plane)
      zpos = PPPTHICK / 2.0 + PPDTHICK - CROH / 2.0;
      snprintf(cTagV, kTag, "UPPP%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Araldite layer (glue)
      zpos = PGLTHICK / 2.0 + PPPTHICK + PPDTHICK - CROH / 2.0;
      snprintf(cTagV, kTag, "UPGL%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Carbon layers (carbon fiber mats)
      zpos = PCBTHICK / 2.0 + PGLTHICK + PPPTHICK + PPDTHICK - CROH / 2.0;
      snprintf(cTagV, kTag, "UPCB%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      zpos = -PCBTHICK / 2.0 - PPCTHICK - PRBTHICK - PELTHICK + CROH / 2.0;
      snprintf(cTagV, kTag, "UPCB%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Aramide layer (honeycomb)
      zpos = PHCTHICK / 2.0 + PCBTHICK + PGLTHICK + PPPTHICK + PPDTHICK - CROH / 2.0;
      snprintf(cTagV, kTag, "UPHC%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // G10 layer (PCB readout board)
      zpos = -PPCTHICK / 2.0 - PRBTHICK - PELTHICK + CROH / 2.0;
      snprintf(cTagV, kTag, "UPPC%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Cu layer (traces in readout board)
      zpos = -PRBTHICK / 2.0 - PELTHICK + CROH / 2.0;
      snprintf(cTagV, kTag, "UPRB%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // Cu layer (other materials on readout board, incl. screws)
      zpos = -PELTHICK / 2.0 + CROH / 2.0;
      snprintf(cTagV, kTag, "UPEL%02d", iDet);
      snprintf(cTagM, kTag, "UG%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // Position the inner volumes of the chambers in the frames
      xpos = 0.0;
      ypos = 0.0;

      // The inner part of the radiator (air)
      zpos = 0.0;
      snprintf(cTagV, kTag, "UC%02d", iDet);
      snprintf(cTagM, kTag, "UX%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // The glue around the radiator
      zpos = CRAH / 2.0 - CDRH / 2.0 - CRAH / 2.0;
      snprintf(cTagV, kTag, "UX%02d", iDet);
      snprintf(cTagM, kTag, "UB%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
      // The lower Wacosit frame inside the aluminum frame
      zpos = 0.0;
      snprintf(cTagV, kTag, "UB%02d", iDet);
      snprintf(cTagM, kTag, "UA%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // The inside of the upper Wacosit frame
      zpos = 0.0;
      snprintf(cTagV, kTag, "UE%02d", iDet);
      snprintf(cTagM, kTag, "UD%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");

      // The inside of the upper aluminum frame
      zpos = 0.0;
      snprintf(cTagV, kTag, "UG%02d", iDet);
      snprintf(cTagM, kTag, "UF%02d", iDet);
      TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
    }
  }

  // Create the volumes of the super module frame
  createFrame(idtmed);

  // Create the volumes of the services
  createServices(idtmed);

  for (int istack = 0; istack < kNstack; istack++) {
    for (int ilayer = 0; ilayer < kNlayer; ilayer++) {
      assembleChamber(ilayer, istack);
    }
  }

  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTI1", 1, "UTS1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTI2", 1, "UTS2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTI3", 1, "UTS3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTI4", 1, "UTS4", xpos, ypos, zpos, 0, "ONLY");

  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTS1", 1, "UTR1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTS2", 1, "UTR2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTS3", 1, "UTR3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTS4", 1, "UTR4", xpos, ypos, zpos, 0, "ONLY");

  // Put the TRD volumes into the space frame mother volumes
  // if enabled via status flag
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  for (int isector = 0; isector < kNsector; isector++) {
    if (getSMstatus(isector)) {
      snprintf(cTagV, kTag, "BTRD%d", isector);
      switch (isector) {
        case 17:
          // Missing L4S4 chamber
          TVirtualMC::GetMC()->Gspos("UTR4", 1, cTagV, xpos, ypos, zpos, 0, "ONLY");
          break;
        case 13:
        case 14:
        case 15:
          // Double carbon, w/o middle stack
          TVirtualMC::GetMC()->Gspos("UTR3", 1, cTagV, xpos, ypos, zpos, 0, "ONLY");
          break;
        case 11:
        case 12:
          // Double carbon, all stacks
          TVirtualMC::GetMC()->Gspos("UTR2", 1, cTagV, xpos, ypos, zpos, 0, "ONLY");
          break;
        default:
          // Standard supermodule
          TVirtualMC::GetMC()->Gspos("UTR1", 1, cTagV, xpos, ypos, zpos, 0, "ONLY");
      };
    }
  }

  // Put the TRD volumes into the space frame mother volumes
  // if enabled via status flag
  xpos = 0.0;
  ypos = 0.5 * SLENGTH + 0.5 * FLENGTH;
  zpos = 0.0;
  for (int isector = 0; isector < kNsector; isector++) {
    if (getSMstatus(isector)) {
      snprintf(cTagV, kTag, "BTRD%d", isector);
      TVirtualMC::GetMC()->Gspos("UTF1", 1, cTagV, xpos, ypos, zpos, 0, "ONLY");
      TVirtualMC::GetMC()->Gspos("UTF2", 1, cTagV, xpos, -ypos, zpos, 0, "ONLY");
    }
  }

  // Resolve runtime shapes (which is done as part of TGeoManager::CheckGeometry) NOW.
  // This is otherwise done when saying gGeoManager->CloseGeometry().
  // However, we need to make sure all the TGeoVolumes are correctly available even before this
  // stage because FairMCApplication initializes the sensisitive volumes before closing the geometry.
  // The true origin of the "problem" comes from the fact, that the TRD construction above uses
  // Geant3-like construction routines that allow giving negative parameters, indicating dimensions to be
  // fixed later. This prevents immediate construction of the TGeoVolume.
  gGeoManager->CheckGeometry();
}

//_____________________________________________________________________________
void TRDGeometry::createFrame(std::vector<int> const& idtmed)
{
  //
  // Create the geometry of the frame of the supermodule
  //
  // Names of the TRD services volumina
  //
  //        USRL    Support rails for the chambers (Al)
  //        USxx    Support cross bars between the chambers (Al)
  //        USHx    Horizontal connection between the cross bars (Al)
  //        USLx    Long corner ledges (Al)
  //

  int ilayer = 0;

  float xpos = 0.0;
  float ypos = 0.0;
  float zpos = 0.0;

  const int kTag = 100;
  char cTagV[kTag];
  char cTagM[kTag];

  const int kNparTRD = 4;
  float parTRD[kNparTRD];
  const int kNparBOX = 3;
  float parBOX[kNparBOX];
  const int kNparTRP = 11;
  float parTRP[kNparTRP];

  // The rotation matrices
  const int kNmatrix = 7;
  int matrix[kNmatrix];
  TVirtualMC::GetMC()->Matrix(matrix[0], 100.0, 0.0, 90.0, 90.0, 10.0, 0.0);
  TVirtualMC::GetMC()->Matrix(matrix[1], 80.0, 0.0, 90.0, 90.0, 10.0, 180.0);
  TVirtualMC::GetMC()->Matrix(matrix[2], 90.0, 0.0, 0.0, 0.0, 90.0, 90.0);
  TVirtualMC::GetMC()->Matrix(matrix[3], 90.0, 180.0, 0.0, 180.0, 90.0, 90.0);
  TVirtualMC::GetMC()->Matrix(matrix[4], 170.0, 0.0, 80.0, 0.0, 90.0, 90.0);
  TVirtualMC::GetMC()->Matrix(matrix[5], 170.0, 180.0, 80.0, 180.0, 90.0, 90.0);
  TVirtualMC::GetMC()->Matrix(matrix[6], 180.0, 180.0, 90.0, 180.0, 90.0, 90.0);

  //
  // The carbon inserts in the top/bottom aluminum plates
  //

  const int kNparCrb = 3;
  float parCrb[kNparCrb];
  parCrb[0] = 0.0;
  parCrb[1] = 0.0;
  parCrb[2] = 0.0;
  createVolume("USCR", "BOX ", idtmed[26], parCrb, 0);
  // Bottom 1 (all sectors)
  parCrb[0] = 77.49 / 2.0;
  parCrb[1] = 104.60 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 0.0;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 1, "UTS1", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 2, "UTS2", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 3, "UTS3", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 4, "UTS4", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  // Bottom 2 (all sectors)
  parCrb[0] = 77.49 / 2.0;
  parCrb[1] = 55.80 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 85.6;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 5, "UTS1", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 6, "UTS2", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 7, "UTS3", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 8, "UTS4", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 9, "UTS1", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 10, "UTS2", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 11, "UTS3", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 12, "UTS4", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  // Bottom 3 (all sectors)
  parCrb[0] = 77.49 / 2.0;
  parCrb[1] = 56.00 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 148.5;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 13, "UTS1", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 14, "UTS2", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 15, "UTS3", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 16, "UTS4", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 17, "UTS1", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 18, "UTS2", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 19, "UTS3", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 20, "UTS4", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  // Bottom 4 (all sectors)
  parCrb[0] = 77.49 / 2.0;
  parCrb[1] = 118.00 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 240.5;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 21, "UTS1", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 22, "UTS2", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 23, "UTS3", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 24, "UTS4", xpos, ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 25, "UTS1", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 26, "UTS2", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 27, "UTS3", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 28, "UTS4", xpos, -ypos, zpos, 0, "ONLY", parCrb, kNparCrb);
  // Top 1 (only in front of PHOS)
  parCrb[0] = 111.48 / 2.0;
  parCrb[1] = 105.00 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 0.0;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 29, "UTS2", xpos, ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 30, "UTS3", xpos, ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);
  // Top 2 (only in front of PHOS)
  parCrb[0] = 111.48 / 2.0;
  parCrb[1] = 56.00 / 2.0;
  parCrb[2] = SMPLTT / 2.0;
  xpos = 0.0;
  ypos = 85.5;
  zpos = SMPLTT / 2.0 - SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("USCR", 31, "UTS2", xpos, ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 32, "UTS3", xpos, ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 33, "UTS2", xpos, -ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);
  TVirtualMC::GetMC()->Gsposp("USCR", 34, "UTS3", xpos, -ypos, -zpos, 0, "ONLY", parCrb, kNparCrb);

  //
  // The chamber support rails
  //

  const float kSRLhgt = 2.00;
  const float kSRLwidA = 2.3;
  const float kSRLwidB = 1.947;
  const float kSRLdst = 1.135;
  const int kNparSRL = 11;
  float parSRL[kNparSRL];
  // Trapezoidal shape
  parSRL[0] = SLENGTH / 2.0;
  parSRL[1] = 0.0;
  parSRL[2] = 0.0;
  parSRL[3] = kSRLhgt / 2.0;
  parSRL[4] = kSRLwidB / 2.0;
  parSRL[5] = kSRLwidA / 2.0;
  parSRL[6] = 5.0;
  parSRL[7] = kSRLhgt / 2.0;
  parSRL[8] = kSRLwidB / 2.0;
  parSRL[9] = kSRLwidA / 2.0;
  parSRL[10] = 5.0;
  createVolume("USRL", "TRAP", idtmed[1], parSRL, kNparSRL);

  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    xpos = CWIDTH[ilayer] / 2.0 + kSRLwidA / 2.0 + kSRLdst;
    ypos = 0.0;
    zpos = VROCSM + SMPLTT - CALZPOS - SHEIGHT / 2.0 + CRAH + CDRH - CALH - kSRLhgt / 2.0 +
           ilayer * (CH + VSPACE);
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1, "UTI1", xpos, ypos, zpos, matrix[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + kNlayer, "UTI1", -xpos, ypos, zpos, matrix[3], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 2 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 3 * kNlayer, "UTI2", -xpos, ypos, zpos, matrix[3], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 4 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 5 * kNlayer, "UTI3", -xpos, ypos, zpos, matrix[3], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 6 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("USRL", ilayer + 1 + 7 * kNlayer, "UTI4", -xpos, ypos, zpos, matrix[3], "ONLY");
  }

  //
  // The cross bars between the chambers
  //

  const float kSCBwid = 1.0;
  const float kSCBthk = 2.0;
  const float kSCHhgt = 0.3;

  const int kNparSCB = 3;
  float parSCB[kNparSCB];
  parSCB[1] = kSCBwid / 2.0;
  parSCB[2] = CH / 2.0 + VSPACE / 2.0 - kSCHhgt;

  const int kNparSCI = 3;
  float parSCI[kNparSCI];
  parSCI[1] = -1;

  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  for (ilayer = 0; ilayer < kNlayer; ilayer++) {
    // The aluminum of the cross bars
    parSCB[0] = CWIDTH[ilayer] / 2.0 + kSRLdst / 2.0;
    snprintf(cTagV, kTag, "USF%01d", ilayer);
    createVolume(cTagV, "BOX ", idtmed[1], parSCB, kNparSCB);

    // The empty regions in the cross bars
    float thkSCB = kSCBthk;
    if (ilayer < 2) {
      thkSCB *= 1.5;
    }
    parSCI[2] = parSCB[2] - thkSCB;
    parSCI[0] = parSCB[0] / 4.0 - kSCBthk;
    snprintf(cTagV, kTag, "USI%01d", ilayer);
    createVolume(cTagV, "BOX ", idtmed[2], parSCI, kNparSCI);

    snprintf(cTagV, kTag, "USI%01d", ilayer);
    snprintf(cTagM, kTag, "USF%01d", ilayer);
    ypos = 0.0;
    zpos = 0.0;
    xpos = parSCI[0] + thkSCB / 2.0;
    TVirtualMC::GetMC()->Gspos(cTagV, 1, cTagM, xpos, ypos, zpos, 0, "ONLY");
    xpos = -parSCI[0] - thkSCB / 2.0;
    TVirtualMC::GetMC()->Gspos(cTagV, 2, cTagM, xpos, ypos, zpos, 0, "ONLY");
    xpos = 3.0 * parSCI[0] + 1.5 * thkSCB;
    TVirtualMC::GetMC()->Gspos(cTagV, 3, cTagM, xpos, ypos, zpos, 0, "ONLY");
    xpos = -3.0 * parSCI[0] - 1.5 * thkSCB;
    TVirtualMC::GetMC()->Gspos(cTagV, 4, cTagM, xpos, ypos, zpos, 0, "ONLY");

    snprintf(cTagV, kTag, "USF%01d", ilayer);
    xpos = 0.0;
    zpos = VROCSM + SMPLTT + parSCB[2] - SHEIGHT / 2.0 + ilayer * (CH + VSPACE);

    ypos = CLENGTH[ilayer][2] / 2.0 + CLENGTH[ilayer][1];
    TVirtualMC::GetMC()->Gspos(cTagV, 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");

    ypos = -CLENGTH[ilayer][2] / 2.0 - CLENGTH[ilayer][1];
    TVirtualMC::GetMC()->Gspos(cTagV, 2, "UTI1", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 4, "UTI2", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 6, "UTI3", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 8, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  }

  //
  // The horizontal connections between the cross bars
  //

  const int kNparSCH = 3;
  float parSCH[kNparSCH];

  for (ilayer = 1; ilayer < kNlayer - 1; ilayer++) {
    parSCH[0] = CWIDTH[ilayer] / 2.0;
    parSCH[1] = (CLENGTH[ilayer + 1][2] / 2.0 + CLENGTH[ilayer + 1][1] - CLENGTH[ilayer][2] / 2.0 -
                 CLENGTH[ilayer][1]) /
                2.0;
    parSCH[2] = kSCHhgt / 2.0;

    snprintf(cTagV, kTag, "USH%01d", ilayer);
    createVolume(cTagV, "BOX ", idtmed[1], parSCH, kNparSCH);
    xpos = 0.0;
    ypos = CLENGTH[ilayer][2] / 2.0 + CLENGTH[ilayer][1] + parSCH[1];
    zpos = VROCSM + SMPLTT - kSCHhgt / 2.0 - SHEIGHT / 2.0 + (ilayer + 1) * (CH + VSPACE);
    TVirtualMC::GetMC()->Gspos(cTagV, 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
    ypos = -ypos;
    TVirtualMC::GetMC()->Gspos(cTagV, 2, "UTI1", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 4, "UTI2", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 6, "UTI3", xpos, ypos, zpos, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos(cTagV, 8, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  }

  //
  // The asymmetric flat frame in the middle
  //

  // The envelope volume (aluminum)
  parTRD[0] = 87.60 / 2.0;
  parTRD[1] = 114.00 / 2.0;
  parTRD[2] = 1.20 / 2.0;
  parTRD[3] = 71.30 / 2.0;
  createVolume("USDB", "TRD1", idtmed[1], parTRD, kNparTRD);
  // Empty spaces (air)
  parTRP[0] = 1.20 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 27.00 / 2.0;
  parTRP[4] = 50.60 / 2.0;
  parTRP[5] = 5.00 / 2.0;
  parTRP[6] = 3.5;
  parTRP[7] = 27.00 / 2.0;
  parTRP[8] = 50.60 / 2.0;
  parTRP[9] = 5.00 / 2.0;
  parTRP[10] = 3.5;
  createVolume("USD1", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = 18.0;
  ypos = 0.0;
  zpos = 27.00 / 2.0 - 71.3 / 2.0;
  TVirtualMC::GetMC()->Gspos("USD1", 1, "USDB", xpos, ypos, zpos, matrix[2], "ONLY");
  // Empty spaces (air)
  parTRP[0] = 1.20 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 33.00 / 2.0;
  parTRP[4] = 5.00 / 2.0;
  parTRP[5] = 62.10 / 2.0;
  parTRP[6] = 3.5;
  parTRP[7] = 33.00 / 2.0;
  parTRP[8] = 5.00 / 2.0;
  parTRP[9] = 62.10 / 2.0;
  parTRP[10] = 3.5;
  createVolume("USD2", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = 21.0;
  ypos = 0.0;
  zpos = 71.3 / 2.0 - 33.0 / 2.0;
  TVirtualMC::GetMC()->Gspos("USD2", 1, "USDB", xpos, ypos, zpos, matrix[2], "ONLY");
  // Empty spaces (air)
  parBOX[0] = 22.50 / 2.0;
  parBOX[1] = 1.20 / 2.0;
  parBOX[2] = 70.50 / 2.0;
  createVolume("USD3", "BOX ", idtmed[2], parBOX, kNparBOX);
  xpos = -25.75;
  ypos = 0.0;
  zpos = 0.4;
  TVirtualMC::GetMC()->Gspos("USD3", 1, "USDB", xpos, ypos, zpos, 0, "ONLY");
  // Empty spaces (air)
  parTRP[0] = 1.20 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 25.50 / 2.0;
  parTRP[4] = 5.00 / 2.0;
  parTRP[5] = 65.00 / 2.0;
  parTRP[6] = -1.0;
  parTRP[7] = 25.50 / 2.0;
  parTRP[8] = 5.00 / 2.0;
  parTRP[9] = 65.00 / 2.0;
  parTRP[10] = -1.0;
  createVolume("USD4", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = 2.0;
  ypos = 0.0;
  zpos = -1.6;
  TVirtualMC::GetMC()->Gspos("USD4", 1, "USDB", xpos, ypos, zpos, matrix[6], "ONLY");
  // Empty spaces (air)
  parTRP[0] = 1.20 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 23.50 / 2.0;
  parTRP[4] = 63.50 / 2.0;
  parTRP[5] = 5.00 / 2.0;
  parTRP[6] = 16.0;
  parTRP[7] = 23.50 / 2.0;
  parTRP[8] = 63.50 / 2.0;
  parTRP[9] = 5.00 / 2.0;
  parTRP[10] = 16.0;
  createVolume("USD5", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = 36.5;
  ypos = 0.0;
  zpos = -1.5;
  TVirtualMC::GetMC()->Gspos("USD5", 1, "USDB", xpos, ypos, zpos, matrix[5], "ONLY");
  // Empty spaces (air)
  parTRP[0] = 1.20 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 70.50 / 2.0;
  parTRP[4] = 4.50 / 2.0;
  parTRP[5] = 16.50 / 2.0;
  parTRP[6] = -5.0;
  parTRP[7] = 70.50 / 2.0;
  parTRP[8] = 4.50 / 2.0;
  parTRP[9] = 16.50 / 2.0;
  parTRP[10] = -5.0;
  createVolume("USD6", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = -43.7;
  ypos = 0.0;
  zpos = 0.4;
  TVirtualMC::GetMC()->Gspos("USD6", 1, "USDB", xpos, ypos, zpos, matrix[2], "ONLY");
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0;
  zpos = 0.04;
  TVirtualMC::GetMC()->Gspos("USDB", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USDB", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Upper bar (aluminum)
  parBOX[0] = 95.00 / 2.0;
  parBOX[1] = 1.20 / 2.0;
  parBOX[2] = 3.00 / 2.0;
  createVolume("USD7", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 3.00 / 2.0;
  TVirtualMC::GetMC()->Gspos("USD7", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD7", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Lower bar (aluminum)
  parBOX[0] = 90.22 / 2.0;
  parBOX[1] = 1.20 / 2.0;
  parBOX[2] = 1.74 / 2.0;
  createVolume("USD8", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0 - 0.1;
  zpos = -SHEIGHT / 2.0 + SMPLTT + 2.27;
  TVirtualMC::GetMC()->Gspos("USD8", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD8", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Lower bar (aluminum)
  parBOX[0] = 82.60 / 2.0;
  parBOX[1] = 1.20 / 2.0;
  parBOX[2] = 1.40 / 2.0;
  createVolume("USD9", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0;
  zpos = -SHEIGHT / 2.0 + SMPLTT + 1.40 / 2.0;
  TVirtualMC::GetMC()->Gspos("USD9", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USD9", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Front sheet (aluminum)
  parTRP[0] = 0.10 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 74.50 / 2.0;
  parTRP[4] = 31.70 / 2.0;
  parTRP[5] = 44.00 / 2.0;
  parTRP[6] = -5.0;
  parTRP[7] = 74.50 / 2.0;
  parTRP[8] = 31.70 / 2.0;
  parTRP[9] = 44.00 / 2.0;
  parTRP[10] = -5.0;
  createVolume("USDF", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = -32.0;
  ypos = CLENGTH[5][2] / 2.0 + 1.20 / 2.0 + 0.10 / 2.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("USDF", 1, "UTI1", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 2, "UTI1", xpos, -ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 3, "UTI2", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 4, "UTI2", xpos, -ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 5, "UTI3", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 6, "UTI3", xpos, -ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 7, "UTI4", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USDF", 8, "UTI4", xpos, -ypos, zpos, matrix[2], "ONLY");

  //
  // The flat frame in front of the chambers
  //

  // The envelope volume (aluminum)
  parTRD[0] = 90.00 / 2.0 - 0.1;
  parTRD[1] = 114.00 / 2.0 - 0.1;
  parTRD[2] = 1.50 / 2.0;
  parTRD[3] = 70.30 / 2.0;
  createVolume("USCB", "TRD1", idtmed[1], parTRD, kNparTRD);
  // Empty spaces (air)
  parTRD[0] = 87.00 / 2.0;
  parTRD[1] = 10.00 / 2.0;
  parTRD[2] = 1.50 / 2.0;
  parTRD[3] = 26.35 / 2.0;
  createVolume("USC1", "TRD1", idtmed[2], parTRD, kNparTRD);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 26.35 / 2.0 - 70.3 / 2.0;
  TVirtualMC::GetMC()->Gspos("USC1", 1, "USCB", xpos, ypos, zpos, 0, "ONLY");
  // Empty spaces (air)
  parTRD[0] = 10.00 / 2.0;
  parTRD[1] = 111.00 / 2.0;
  parTRD[2] = 1.50 / 2.0;
  parTRD[3] = 35.05 / 2.0;
  createVolume("USC2", "TRD1", idtmed[2], parTRD, kNparTRD);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 70.3 / 2.0 - 35.05 / 2.0;
  TVirtualMC::GetMC()->Gspos("USC2", 1, "USCB", xpos, ypos, zpos, 0, "ONLY");
  // Empty spaces (air)
  parTRP[0] = 1.50 / 2.0;
  parTRP[1] = 0.0;
  parTRP[2] = 0.0;
  parTRP[3] = 37.60 / 2.0;
  parTRP[4] = 63.90 / 2.0;
  parTRP[5] = 8.86 / 2.0;
  parTRP[6] = 16.0;
  parTRP[7] = 37.60 / 2.0;
  parTRP[8] = 63.90 / 2.0;
  parTRP[9] = 8.86 / 2.0;
  parTRP[10] = 16.0;
  createVolume("USC3", "TRAP", idtmed[2], parTRP, kNparTRP);
  xpos = -30.5;
  ypos = 0.0;
  zpos = -2.0;
  TVirtualMC::GetMC()->Gspos("USC3", 1, "USCB", xpos, ypos, zpos, matrix[4], "ONLY");
  TVirtualMC::GetMC()->Gspos("USC3", 2, "USCB", -xpos, ypos, zpos, matrix[5], "ONLY");
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0 + CLENGTH[5][1] + CLENGTH[5][0];
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("USCB", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USCB", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Upper bar (aluminum)
  parBOX[0] = 95.00 / 2.0;
  parBOX[1] = 1.50 / 2.0;
  parBOX[2] = 3.00 / 2.0;
  createVolume("USC4", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0 + CLENGTH[5][1] + CLENGTH[5][0];
  zpos = SHEIGHT / 2.0 - SMPLTT - 3.00 / 2.0;
  TVirtualMC::GetMC()->Gspos("USC4", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC4", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Lower bar (aluminum)
  parBOX[0] = 90.22 / 2.0;
  parBOX[1] = 1.50 / 2.0;
  parBOX[2] = 2.00 / 2.0;
  createVolume("USC5", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0 + CLENGTH[5][1] + CLENGTH[5][0];
  zpos = -SHEIGHT / 2.0 + SMPLTT + 2.60;
  TVirtualMC::GetMC()->Gspos("USC5", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC5", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");
  // Lower bar (aluminum)
  parBOX[0] = 82.60 / 2.0;
  parBOX[1] = 1.50 / 2.0;
  parBOX[2] = 1.60 / 2.0;
  createVolume("USC6", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = CLENGTH[5][2] / 2.0 + CLENGTH[5][1] + CLENGTH[5][0];
  zpos = -SHEIGHT / 2.0 + SMPLTT + 1.60 / 2.0;
  TVirtualMC::GetMC()->Gspos("USC6", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 2, "UTI1", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 4, "UTI2", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 6, "UTI3", xpos, -ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USC6", 8, "UTI4", xpos, -ypos, zpos, 0, "ONLY");

  //
  // The long corner ledges
  //

  const int kNparSCL = 3;
  float parSCL[kNparSCL];
  const int kNparSCLb = 11;
  float parSCLb[kNparSCLb];

  // Upper ledges
  // Thickness of the corner ledges
  const float kSCLthkUa = 0.6;
  const float kSCLthkUb = 0.6;
  // Width of the corner ledges
  const float kSCLwidUa = 3.2;
  const float kSCLwidUb = 4.8;
  // Position of the corner ledges
  const float kSCLposxUa = 0.7;
  const float kSCLposxUb = 3.3;
  const float kSCLposzUa = 1.65;
  const float kSCLposzUb = 0.3;
  // Vertical
  parSCL[0] = kSCLthkUa / 2.0;
  parSCL[1] = SLENGTH / 2.0;
  parSCL[2] = kSCLwidUa / 2.0;
  createVolume("USL1", "BOX ", idtmed[1], parSCL, kNparSCL);
  xpos = SWIDTH2 / 2.0 - SMPLTT - kSCLposxUa;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - kSCLposzUa;
  TVirtualMC::GetMC()->Gspos("USL1", 1, "UTI1", xpos, ypos, zpos, matrix[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL1", 3, "UTI4", xpos, ypos, zpos, matrix[0], "ONLY");
  xpos = -xpos;
  TVirtualMC::GetMC()->Gspos("USL1", 2, "UTI1", xpos, ypos, zpos, matrix[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL1", 4, "UTI4", xpos, ypos, zpos, matrix[1], "ONLY");
  // Horizontal
  parSCL[0] = kSCLwidUb / 2.0;
  parSCL[1] = SLENGTH / 2.0;
  parSCL[2] = kSCLthkUb / 2.0;
  createVolume("USL2", "BOX ", idtmed[1], parSCL, kNparSCL);
  xpos = SWIDTH2 / 2.0 - SMPLTT - kSCLposxUb;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - kSCLposzUb;
  TVirtualMC::GetMC()->Gspos("USL2", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  xpos = -xpos;
  TVirtualMC::GetMC()->Gspos("USL2", 2, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 4, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 6, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL2", 8, "UTI4", xpos, ypos, zpos, 0, "ONLY");

  // Lower ledges
  // Thickness of the corner ledges
  const float kSCLthkLa = 2.464;
  const float kSCLthkLb = 1.0;
  // Width of the corner ledges
  const float kSCLwidLa = 8.3;
  const float kSCLwidLb = 4.0;
  // Position of the corner ledges
  const float kSCLposxLa = (3.0 * kSCLthkLb - kSCLthkLa) / 4.0 + 0.05;
  const float kSCLposxLb = kSCLthkLb + kSCLwidLb / 2.0 + 0.05;
  const float kSCLposzLa = kSCLwidLa / 2.0;
  const float kSCLposzLb = kSCLthkLb / 2.0;
  // Vertical
  // Trapezoidal shape
  parSCLb[0] = SLENGTH / 2.0;
  parSCLb[1] = 0.0;
  parSCLb[2] = 0.0;
  parSCLb[3] = kSCLwidLa / 2.0;
  parSCLb[4] = kSCLthkLb / 2.0;
  parSCLb[5] = kSCLthkLa / 2.0;
  parSCLb[6] = 5.0;
  parSCLb[7] = kSCLwidLa / 2.0;
  parSCLb[8] = kSCLthkLb / 2.0;
  parSCLb[9] = kSCLthkLa / 2.0;
  parSCLb[10] = 5.0;
  createVolume("USL3", "TRAP", idtmed[1], parSCLb, kNparSCLb);
  xpos = SWIDTH1 / 2.0 - SMPLTT - kSCLposxLa;
  ypos = 0.0;
  zpos = -SHEIGHT / 2.0 + SMPLTT + kSCLposzLa;
  TVirtualMC::GetMC()->Gspos("USL3", 1, "UTI1", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 3, "UTI2", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 5, "UTI3", xpos, ypos, zpos, matrix[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 7, "UTI4", xpos, ypos, zpos, matrix[2], "ONLY");
  xpos = -xpos;
  TVirtualMC::GetMC()->Gspos("USL3", 2, "UTI1", xpos, ypos, zpos, matrix[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 4, "UTI2", xpos, ypos, zpos, matrix[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 6, "UTI3", xpos, ypos, zpos, matrix[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("USL3", 8, "UTI4", xpos, ypos, zpos, matrix[3], "ONLY");
  // Horizontal part
  parSCL[0] = kSCLwidLb / 2.0;
  parSCL[1] = SLENGTH / 2.0;
  parSCL[2] = kSCLthkLb / 2.0;
  createVolume("USL4", "BOX ", idtmed[1], parSCL, kNparSCL);
  xpos = SWIDTH1 / 2.0 - SMPLTT - kSCLposxLb;
  ypos = 0.0;
  zpos = -SHEIGHT / 2.0 + SMPLTT + kSCLposzLb;
  TVirtualMC::GetMC()->Gspos("USL4", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 3, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 5, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 7, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  xpos = -xpos;
  TVirtualMC::GetMC()->Gspos("USL4", 2, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 4, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 6, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("USL4", 8, "UTI4", xpos, ypos, zpos, 0, "ONLY");

  //
  // Aluminum plates in the front part of the super modules
  //

  const int kNparTrd = 4;
  float parTrd[kNparTrd];
  parTrd[0] = SWIDTH1 / 2.0 - 2.5;
  parTrd[1] = SWIDTH2 / 2.0 - 2.5;
  parTrd[2] = SMPLTT / 2.0;
  parTrd[3] = SHEIGHT / 2.0 - 1.0;
  createVolume("UTA1", "TRD1", idtmed[1], parTrd, kNparTrd);
  xpos = 0.0;
  ypos = SMPLTT / 2.0 - FLENGTH / 2.0;
  zpos = -0.5;
  TVirtualMC::GetMC()->Gspos("UTA1", 1, "UTF1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTA1", 2, "UTF2", xpos, -ypos, zpos, 0, "ONLY");

  const int kNparPlt = 3;
  float parPlt[kNparPlt];
  parPlt[0] = 0.0;
  parPlt[1] = 0.0;
  parPlt[2] = 0.0;
  createVolume("UTA2", "BOX ", idtmed[1], parPlt, 0);
  xpos = 0.0;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT / 2.0;
  parPlt[0] = SWIDTH2 / 2.0 - 0.2;
  parPlt[1] = FLENGTH / 2.0;
  parPlt[2] = SMPLTT / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTA2", 1, "UTF2", xpos, ypos, zpos, 0, "ONLY", parPlt, kNparPlt);
  xpos = (SWIDTH1 + SWIDTH2) / 4.0 - SMPLTT / 2.0 - 0.0016;
  ypos = 0.0;
  zpos = 0.0;
  parPlt[0] = SMPLTT / 2.0;
  parPlt[1] = FLENGTH / 2.0;
  parPlt[2] = SHEIGHT / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTA2", 2, "UTF2", xpos, ypos, zpos, matrix[0], "ONLY", parPlt, kNparPlt);
  TVirtualMC::GetMC()->Gsposp("UTA2", 3, "UTF2", -xpos, ypos, zpos, matrix[1], "ONLY", parPlt, kNparPlt);

  // Additional aluminum bar
  parBOX[0] = 80.0 / 2.0;
  parBOX[1] = 1.0 / 2.0;
  parBOX[2] = 10.0 / 2.0;
  createVolume("UTA3", "BOX ", idtmed[1], parBOX, kNparBOX);
  xpos = 0.0;
  ypos = 1.0 / 2.0 + SMPLTT - FLENGTH / 2.0;
  zpos = SHEIGHT / 2.0 - 1.5 - 10.0 / 2.0;
  TVirtualMC::GetMC()->Gspos("UTA3", 1, "UTF1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTA3", 2, "UTF2", xpos, -ypos, zpos, 0, "ONLY");
}

//_____________________________________________________________________________
void TRDGeometry::createServices(std::vector<int> const& idtmed)
{
  //
  // Create the geometry of the services
  //
  // Names of the TRD services volumina
  //
  //        UTC1    Cooling arterias (Al)
  //        UTC2    Cooling arterias (Water)
  //        UUxx    Volumes for the services at the chambers (Air)
  //        UMCM    Readout MCMs     (G10/Cu/Si)
  //        UDCS    DCSs boards      (G10/Cu)
  //        UTP1    Power bars       (Cu)
  //        UTCP    Cooling pipes    (Fe)
  //        UTCH    Cooling pipes    (Water)
  //        UTPL    Power lines      (Cu)
  //        UTGD    Gas distribution box (V2A)
  //

  int ilayer = 0;
  int istack = 0;

  float xpos = 0.0;
  float ypos = 0.0;
  float zpos = 0.0;

  const int kTag = 100;
  char cTagV[kTag];

  const int kNparBox = 3;
  float parBox[kNparBox];

  const int kNparTube = 3;
  float parTube[kNparTube];

  // Services inside the baby frame
  const float kBBMdz = 223.0;
  const float kBBSdz = 8.5;

  // Services inside the back frame
  const float kBFMdz = 118.0;
  const float kBFSdz = 8.5;

  // The rotation matrices
  const int kNmatrix = 10;
  int matrix[kNmatrix];
  TVirtualMC::GetMC()->Matrix(matrix[0], 100.0, 0.0, 90.0, 90.0, 10.0, 0.0);  // rotation around y-axis
  TVirtualMC::GetMC()->Matrix(matrix[1], 80.0, 0.0, 90.0, 90.0, 10.0, 180.0); // rotation around y-axis
  TVirtualMC::GetMC()->Matrix(matrix[2], 0.0, 0.0, 90.0, 90.0, 90.0, 0.0);
  TVirtualMC::GetMC()->Matrix(matrix[3], 180.0, 0.0, 90.0, 90.0, 90.0, 180.0);
  TVirtualMC::GetMC()->Matrix(matrix[4], 90.0, 0.0, 0.0, 0.0, 90.0, 90.0);
  TVirtualMC::GetMC()->Matrix(matrix[5], 100.0, 0.0, 90.0, 270.0, 10.0, 0.0);
  TVirtualMC::GetMC()->Matrix(matrix[6], 80.0, 0.0, 90.0, 270.0, 10.0, 180.0);
  TVirtualMC::GetMC()->Matrix(matrix[7], 90.0, 10.0, 90.0, 100.0, 0.0, 0.0); // rotation around z-axis
  TVirtualMC::GetMC()->Matrix(matrix[8], 90.0, 350.0, 90.0, 80.0, 0.0, 0.0); // rotation around z-axis
  TVirtualMC::GetMC()->Matrix(matrix[9], 90.0, 90.0, 90.0, 180.0, 0.0, 0.0); // rotation around z-axis

  //
  // The cooling arterias
  //

  // Width of the cooling arterias
  const float kCOLwid = 0.8;
  // Height of the cooling arterias
  const float kCOLhgt = 6.5;
  // Positioning of the cooling
  const float kCOLposx = 1.0;
  const float kCOLposz = -1.2;
  // Thickness of the walls of the cooling arterias
  const float kCOLthk = 0.1;
  const int kNparCOL = 3;
  float parCOL[kNparCOL];
  parCOL[0] = 0.0;
  parCOL[1] = 0.0;
  parCOL[2] = 0.0;
  createVolume("UTC1", "BOX ", idtmed[8], parCOL, 0);
  createVolume("UTC3", "BOX ", idtmed[8], parCOL, 0);
  parCOL[0] = kCOLwid / 2.0 - kCOLthk;
  parCOL[1] = -1.0;
  parCOL[2] = kCOLhgt / 2.0 - kCOLthk;
  createVolume("UTC2", "BOX ", idtmed[14], parCOL, kNparCOL);
  createVolume("UTC4", "BOX ", idtmed[14], parCOL, kNparCOL);

  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTC2", 1, "UTC1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTC4", 1, "UTC3", xpos, ypos, zpos, 0, "ONLY");

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // Along the chambers
    xpos = CWIDTH[ilayer] / 2.0 + kCOLwid / 2.0 + kCOLposx;
    ypos = 0.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kCOLhgt / 2.0 - SHEIGHT / 2.0 + kCOLposz + ilayer * (CH + VSPACE);
    parCOL[0] = kCOLwid / 2.0;
    parCOL[1] = SLENGTH / 2.0;
    parCOL[2] = kCOLhgt / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer, "UTI1", xpos, ypos, zpos, matrix[0], "ONLY", parCOL, kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + kNlayer, "UTI1", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 6 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 7 * kNlayer, "UTI2", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 8 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 9 * kNlayer, "UTI3", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 10 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC1", ilayer + 11 * kNlayer, "UTI4", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);

    // Front of supermodules
    xpos = CWIDTH[ilayer] / 2.0 + kCOLwid / 2.0 + kCOLposx;
    ypos = 0.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kCOLhgt / 2.0 - SHEIGHT / 2.0 + kCOLposz + ilayer * (CH + VSPACE);
    parCOL[0] = kCOLwid / 2.0;
    parCOL[1] = FLENGTH / 2.0;
    parCOL[2] = kCOLhgt / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 2 * kNlayer, "UTF1", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 3 * kNlayer, "UTF1", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 4 * kNlayer, "UTF2", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 5 * kNlayer, "UTF2", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
  }

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // In baby frame
    xpos = CWIDTH[ilayer] / 2.0 + kCOLwid / 2.0 + kCOLposx - 2.5;
    ypos = kBBSdz / 2.0 - kBBMdz / 2.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kCOLhgt / 2.0 - SHEIGHT / 2.0 + kCOLposz + ilayer * (CH + VSPACE);
    parCOL[0] = kCOLwid / 2.0;
    parCOL[1] = kBBSdz / 2.0;
    parCOL[2] = kCOLhgt / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 6 * kNlayer, "BBTRD", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 7 * kNlayer, "BBTRD", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
  }

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // In back frame
    xpos = CWIDTH[ilayer] / 2.0 + kCOLwid / 2.0 + kCOLposx - 0.3;
    ypos = -kBFSdz / 2.0 + kBFMdz / 2.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kCOLhgt / 2.0 - SHEIGHT / 2.0 + kCOLposz + ilayer * (CH + VSPACE);
    parCOL[0] = kCOLwid / 2.0;
    parCOL[1] = kBFSdz / 2.0;
    parCOL[2] = kCOLhgt / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 6 * kNlayer, "BFTRD", xpos, ypos, zpos, matrix[0], "ONLY", parCOL,
                                kNparCOL);
    TVirtualMC::GetMC()->Gsposp("UTC3", ilayer + 7 * kNlayer, "BFTRD", -xpos, ypos, zpos, matrix[1], "ONLY", parCOL,
                                kNparCOL);
  }

  // The upper most layer
  // Along the chambers
  xpos = CWIDTH[5] / 2.0 - kCOLhgt / 2.0 - 1.3;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.4 - kCOLwid / 2.0;
  parCOL[0] = kCOLwid / 2.0;
  parCOL[1] = SLENGTH / 2.0;
  parCOL[2] = kCOLhgt / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTC1", 6, "UTI1", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + kNlayer, "UTI1", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 6 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 7 * kNlayer, "UTI2", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 8 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 9 * kNlayer, "UTI3", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 10 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC1", 6 + 11 * kNlayer, "UTI4", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  // Front of supermodules
  xpos = CWIDTH[5] / 2.0 - kCOLhgt / 2.0 - 1.3;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.4 - kCOLwid / 2.0;
  parCOL[0] = kCOLwid / 2.0;
  parCOL[1] = FLENGTH / 2.0;
  parCOL[2] = kCOLhgt / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 2 * kNlayer, "UTF1", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 3 * kNlayer, "UTF1", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 4 * kNlayer, "UTF2", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 5 * kNlayer, "UTF2", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  // In baby frame
  xpos = CWIDTH[5] / 2.0 - kCOLhgt / 2.0 - 3.1;
  ypos = kBBSdz / 2.0 - kBBMdz / 2.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.4 - kCOLwid / 2.0;
  parCOL[0] = kCOLwid / 2.0;
  parCOL[1] = kBBSdz / 2.0;
  parCOL[2] = kCOLhgt / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 6 * kNlayer, "BBTRD", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 7 * kNlayer, "BBTRD", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  // In back frame
  xpos = CWIDTH[5] / 2.0 - kCOLhgt / 2.0 - 1.3;
  ypos = -kBFSdz / 2.0 + kBFMdz / 2.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.4 - kCOLwid / 2.0;
  parCOL[0] = kCOLwid / 2.0;
  parCOL[1] = kBFSdz / 2.0;
  parCOL[2] = kCOLhgt / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 6 * kNlayer, "BFTRD", xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);
  TVirtualMC::GetMC()->Gsposp("UTC3", 6 + 7 * kNlayer, "BFTRD", -xpos, ypos, zpos, matrix[3], "ONLY", parCOL, kNparCOL);

  //
  // The power bus bars
  //

  const float kPWRwid = 0.6;
  // Increase the height of the power bus bars to take into
  // account the material of additional cables, etc.
  const float kPWRhgtA = 5.0 + 0.2;
  const float kPWRhgtB = 5.0;
  const float kPWRposx = 2.0;
  const float kPWRposz = 0.1;
  const int kNparPWR = 3;
  float parPWR[kNparPWR];
  parPWR[0] = 0.0;
  parPWR[1] = 0.0;
  parPWR[2] = 0.0;
  createVolume("UTP1", "BOX ", idtmed[25], parPWR, 0);
  createVolume("UTP3", "BOX ", idtmed[25], parPWR, 0);

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // Along the chambers
    xpos = CWIDTH[ilayer] / 2.0 + kPWRwid / 2.0 + kPWRposx;
    ypos = 0.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kPWRhgtA / 2.0 - SHEIGHT / 2.0 + kPWRposz + ilayer * (CH + VSPACE);
    parPWR[0] = kPWRwid / 2.0;
    parPWR[1] = SLENGTH / 2.0;
    parPWR[2] = kPWRhgtA / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer, "UTI1", xpos, ypos, zpos, matrix[0], "ONLY", parPWR, kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + kNlayer, "UTI1", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 6 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 7 * kNlayer, "UTI2", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 8 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 9 * kNlayer, "UTI3", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 10 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP1", ilayer + 11 * kNlayer, "UTI4", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);

    // Front of supermodule
    xpos = CWIDTH[ilayer] / 2.0 + kPWRwid / 2.0 + kPWRposx;
    ypos = 0.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kPWRhgtA / 2.0 - SHEIGHT / 2.0 + kPWRposz + ilayer * (CH + VSPACE);
    parPWR[0] = kPWRwid / 2.0;
    parPWR[1] = FLENGTH / 2.0;
    parPWR[2] = kPWRhgtA / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 2 * kNlayer, "UTF1", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 3 * kNlayer, "UTF1", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 4 * kNlayer, "UTF2", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 5 * kNlayer, "UTF2", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
  }

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // In baby frame
    xpos = CWIDTH[ilayer] / 2.0 + kPWRwid / 2.0 + kPWRposx - 2.5;
    ypos = kBBSdz / 2.0 - kBBMdz / 2.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kPWRhgtB / 2.0 - SHEIGHT / 2.0 + kPWRposz + ilayer * (CH + VSPACE);
    parPWR[0] = kPWRwid / 2.0;
    parPWR[1] = kBBSdz / 2.0;
    parPWR[2] = kPWRhgtB / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 6 * kNlayer, "BBTRD", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 7 * kNlayer, "BBTRD", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
  }

  for (ilayer = 1; ilayer < kNlayer; ilayer++) {
    // In back frame
    xpos = CWIDTH[ilayer] / 2.0 + kPWRwid / 2.0 + kPWRposx - 0.3;
    ypos = -kBFSdz / 2.0 + kBFMdz / 2.0;
    zpos =
      VROCSM + SMPLTT - CALZPOS + kPWRhgtB / 2.0 - SHEIGHT / 2.0 + kPWRposz + ilayer * (CH + VSPACE);
    parPWR[0] = kPWRwid / 2.0;
    parPWR[1] = kBFSdz / 2.0;
    parPWR[2] = kPWRhgtB / 2.0;
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 8 * kNlayer, "BFTRD", xpos, ypos, zpos, matrix[0], "ONLY", parPWR,
                                kNparPWR);
    TVirtualMC::GetMC()->Gsposp("UTP3", ilayer + 9 * kNlayer, "BFTRD", -xpos, ypos, zpos, matrix[1], "ONLY", parPWR,
                                kNparPWR);
  }

  // The upper most layer
  // Along the chambers
  xpos = CWIDTH[5] / 2.0 + kPWRhgtB / 2.0 - 1.3;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.6 - kPWRwid / 2.0;
  parPWR[0] = kPWRwid / 2.0;
  parPWR[1] = SLENGTH / 2.0;
  parPWR[2] = kPWRhgtB / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTP1", 6, "UTI1", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + kNlayer, "UTI1", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 6 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 7 * kNlayer, "UTI2", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 8 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 9 * kNlayer, "UTI3", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 10 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP1", 6 + 11 * kNlayer, "UTI4", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  // Front of supermodules
  xpos = CWIDTH[5] / 2.0 + kPWRhgtB / 2.0 - 1.3;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.6 - kPWRwid / 2.0;
  parPWR[0] = kPWRwid / 2.0;
  parPWR[1] = FLENGTH / 2.0;
  parPWR[2] = kPWRhgtB / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 2 * kNlayer, "UTF1", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 3 * kNlayer, "UTF1", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 4 * kNlayer, "UTF2", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 5 * kNlayer, "UTF2", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  // In baby frame
  xpos = CWIDTH[5] / 2.0 + kPWRhgtB / 2.0 - 3.0;
  ypos = kBBSdz / 2.0 - kBBMdz / 2.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.6 - kPWRwid / 2.0;
  parPWR[0] = kPWRwid / 2.0;
  parPWR[1] = kBBSdz / 2.0;
  parPWR[2] = kPWRhgtB / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 6 * kNlayer, "BBTRD", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 7 * kNlayer, "BBTRD", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  // In back frame
  xpos = CWIDTH[5] / 2.0 + kPWRhgtB / 2.0 - 1.3;
  ypos = -kBFSdz / 2.0 + kBFMdz / 2.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 0.6 - kPWRwid / 2.0;
  parPWR[0] = kPWRwid / 2.0;
  parPWR[1] = kBFSdz / 2.0;
  parPWR[2] = kPWRhgtB / 2.0;
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 8 * kNlayer, "BFTRD", xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);
  TVirtualMC::GetMC()->Gsposp("UTP3", 6 + 9 * kNlayer, "BFTRD", -xpos, ypos, zpos, matrix[3], "ONLY", parPWR, kNparPWR);

  //
  // The gas tubes connecting the chambers in the super modules with holes
  // Material: Stainless steel
  //

  // PHOS holes
  parTube[0] = 0.0;
  parTube[1] = 2.2 / 2.0;
  parTube[2] = CLENGTH[5][2] / 2.0 - HSPACE / 2.0;
  createVolume("UTG1", "TUBE", idtmed[8], parTube, kNparTube);
  parTube[0] = 0.0;
  parTube[1] = 2.1 / 2.0;
  parTube[2] = CLENGTH[5][2] / 2.0 - HSPACE / 2.0;
  createVolume("UTG2", "TUBE", idtmed[9], parTube, kNparTube);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTG2", 1, "UTG1", xpos, ypos, zpos, 0, "ONLY");
  for (ilayer = 0; ilayer < kNlayer; ilayer++) {
    xpos = CWIDTH[ilayer] / 2.0 + kCOLwid / 2.0 - 1.5;
    ypos = 0.0;
    zpos = VROCSM + SMPLTT + kCOLhgt / 2.0 - SHEIGHT / 2.0 + 5.0 + ilayer * (CH + VSPACE);
    TVirtualMC::GetMC()->Gspos("UTG1", 1 + ilayer, "UTI3", xpos, ypos, zpos, matrix[4], "ONLY");
    TVirtualMC::GetMC()->Gspos("UTG1", 7 + ilayer, "UTI3", -xpos, ypos, zpos, matrix[4], "ONLY");
  }
  // Missing L4S4 chamber in sector 17
  parTube[0] = 0.0;
  parTube[1] = 2.2 / 2.0;
  parTube[2] = CLENGTH[4][4] / 2.0 - HSPACE / 2.0;
  createVolume("UTG3", "TUBE", idtmed[8], parTube, kNparTube);
  parTube[0] = 0.0;
  parTube[1] = 2.1 / 2.0;
  parTube[2] = CLENGTH[4][4] / 2.0 - HSPACE / 2.0;
  createVolume("UTG4", "TUBE", idtmed[9], parTube, kNparTube);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTG4", 1, "UTG3", xpos, ypos, zpos, 0, "ONLY");
  xpos = CWIDTH[4] / 2.0 + kCOLwid / 2.0 - 1.5;
  ypos = -CLENGTH[4][0] / 2.0 - CLENGTH[4][1] - CLENGTH[4][2] / 2.0;
  zpos = VROCSM + SMPLTT + kCOLhgt / 2.0 - SHEIGHT / 2.0 + 5.0 + 4 * (CH + VSPACE);
  TVirtualMC::GetMC()->Gspos("UTG3", 1, "UTI4", xpos, ypos, zpos, matrix[4], "ONLY");
  TVirtualMC::GetMC()->Gspos("UTG4", 2, "UTI4", -xpos, ypos, zpos, matrix[4], "ONLY");

  //
  // The volumes for the services at the chambers
  //

  const int kNparServ = 3;
  float parServ[kNparServ];

  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);

      snprintf(cTagV, kTag, "UU%02d", iDet);
      parServ[0] = CWIDTH[ilayer] / 2.0;
      parServ[1] = CLENGTH[ilayer][istack] / 2.0 - HSPACE / 2.0;
      parServ[2] = CSVH / 2.0;
      createVolume(cTagV, "BOX", idtmed[2], parServ, kNparServ);
    }
  }

  //
  // The cooling pipes inside the service volumes
  //

  // The cooling pipes
  parTube[0] = 0.0;
  parTube[1] = 0.0;
  parTube[2] = 0.0;
  createVolume("UTCP", "TUBE", idtmed[24], parTube, 0);
  // The cooling water
  parTube[0] = 0.0;
  parTube[1] = 0.2 / 2.0;
  parTube[2] = -1.0;
  createVolume("UTCH", "TUBE", idtmed[14], parTube, kNparTube);
  // Water inside the cooling pipe
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTCH", 1, "UTCP", xpos, ypos, zpos, 0, "ONLY");

  // Position the cooling pipes in the mother volume
  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);
      int iCopy = getDetector(ilayer, istack, 0) * 100;
      int nMCMrow = getRowMax(ilayer, istack, 0);
      float ySize = (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)nMCMrow);
      snprintf(cTagV, kTag, "UU%02d", iDet);
      for (int iMCMrow = 0; iMCMrow < nMCMrow; iMCMrow++) {
        xpos = 0.0;
        ypos = (0.5 + iMCMrow) * ySize - CLENGTH[ilayer][istack] / 2.0 + HSPACE / 2.0;
        zpos = 0.0 + 0.742 / 2.0;
        // The cooling pipes
        parTube[0] = 0.0;
        parTube[1] = 0.3 / 2.0; // Thickness of the cooling pipes
        parTube[2] = CWIDTH[ilayer] / 2.0;
        TVirtualMC::GetMC()->Gsposp("UTCP", iCopy + iMCMrow, cTagV, xpos, ypos, zpos, matrix[2], "ONLY", parTube,
                                    kNparTube);
      }
    }
  }

  //
  // The power lines
  //

  // The copper power lines
  parTube[0] = 0.0;
  parTube[1] = 0.0;
  parTube[2] = 0.0;
  createVolume("UTPL", "TUBE", idtmed[5], parTube, 0);

  // Position the power lines in the mother volume
  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);
      int iCopy = getDetector(ilayer, istack, 0) * 100;
      int nMCMrow = getRowMax(ilayer, istack, 0);
      float ySize = (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)nMCMrow);
      snprintf(cTagV, kTag, "UU%02d", iDet);
      for (int iMCMrow = 0; iMCMrow < nMCMrow; iMCMrow++) {
        xpos = 0.0;
        ypos = (0.5 + iMCMrow) * ySize - 1.0 - CLENGTH[ilayer][istack] / 2.0 + HSPACE / 2.0;
        zpos = -0.4 + 0.742 / 2.0;
        parTube[0] = 0.0;
        parTube[1] = 0.2 / 2.0; // Thickness of the power lines
        parTube[2] = CWIDTH[ilayer] / 2.0;
        TVirtualMC::GetMC()->Gsposp("UTPL", iCopy + iMCMrow, cTagV, xpos, ypos, zpos, matrix[2], "ONLY", parTube,
                                    kNparTube);
      }
    }
  }

  //
  // The MCMs
  //

  const float kMCMx = 3.0;
  const float kMCMy = 3.0;
  const float kMCMz = 0.3;

  const float kMCMpcTh = 0.1;
  const float kMCMcuTh = 0.0025;
  const float kMCMsiTh = 0.03;
  const float kMCMcoTh = 0.04;

  // The mother volume for the MCMs (air)
  const int kNparMCM = 3;
  float parMCM[kNparMCM];
  parMCM[0] = kMCMx / 2.0;
  parMCM[1] = kMCMy / 2.0;
  parMCM[2] = kMCMz / 2.0;
  createVolume("UMCM", "BOX", idtmed[2], parMCM, kNparMCM);

  // The MCM carrier G10 layer
  parMCM[0] = kMCMx / 2.0;
  parMCM[1] = kMCMy / 2.0;
  parMCM[2] = kMCMpcTh / 2.0;
  createVolume("UMC1", "BOX", idtmed[19], parMCM, kNparMCM);
  // The MCM carrier Cu layer
  parMCM[0] = kMCMx / 2.0;
  parMCM[1] = kMCMy / 2.0;
  parMCM[2] = kMCMcuTh / 2.0;
  createVolume("UMC2", "BOX", idtmed[18], parMCM, kNparMCM);
  // The silicon of the chips
  parMCM[0] = kMCMx / 2.0;
  parMCM[1] = kMCMy / 2.0;
  parMCM[2] = kMCMsiTh / 2.0;
  createVolume("UMC3", "BOX", idtmed[20], parMCM, kNparMCM);
  // The aluminum of the cooling plates
  parMCM[0] = kMCMx / 2.0;
  parMCM[1] = kMCMy / 2.0;
  parMCM[2] = kMCMcoTh / 2.0;
  createVolume("UMC4", "BOX", idtmed[24], parMCM, kNparMCM);

  // Put the MCM material inside the MCM mother volume
  xpos = 0.0;
  ypos = 0.0;
  zpos = -kMCMz / 2.0 + kMCMpcTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UMC1", 1, "UMCM", xpos, ypos, zpos, 0, "ONLY");
  zpos += kMCMpcTh / 2.0 + kMCMcuTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UMC2", 1, "UMCM", xpos, ypos, zpos, 0, "ONLY");
  zpos += kMCMcuTh / 2.0 + kMCMsiTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UMC3", 1, "UMCM", xpos, ypos, zpos, 0, "ONLY");
  zpos += kMCMsiTh / 2.0 + kMCMcoTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UMC4", 1, "UMCM", xpos, ypos, zpos, 0, "ONLY");

  // Position the MCMs in the mother volume
  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);
      int iCopy = getDetector(ilayer, istack, 0) * 1000;
      int nMCMrow = getRowMax(ilayer, istack, 0);
      float ySize = (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)nMCMrow);
      int nMCMcol = 8;
      float xSize = (getChamberWidth(ilayer) - 2.0 * CPADW) / ((float)nMCMcol + 6); // Introduce 6 gaps
      int iMCM[8] = {1, 2, 3, 5, 8, 9, 10, 12};                                     // 0..7 MCM + 6 gap structure
      snprintf(cTagV, kTag, "UU%02d", iDet);
      for (int iMCMrow = 0; iMCMrow < nMCMrow; iMCMrow++) {
        for (int iMCMcol = 0; iMCMcol < nMCMcol; iMCMcol++) {
          xpos = (0.5 + iMCM[iMCMcol]) * xSize + 1.0 - CWIDTH[ilayer] / 2.0;
          ypos = (0.5 + iMCMrow) * ySize + 1.0 - CLENGTH[ilayer][istack] / 2.0 + HSPACE / 2.0;
          zpos = -0.4 + 0.742 / 2.0;
          TVirtualMC::GetMC()->Gspos("UMCM", iCopy + iMCMrow * 10 + iMCMcol, cTagV, xpos, ypos, zpos, 0, "ONLY");
          // Add two additional smaller cooling pipes on top of the MCMs
          // to mimic the meandering structure
          xpos = (0.5 + iMCM[iMCMcol]) * xSize + 1.0 - CWIDTH[ilayer] / 2.0;
          ypos = (0.5 + iMCMrow) * ySize - CLENGTH[ilayer][istack] / 2.0 + HSPACE / 2.0;
          zpos = 0.0 + 0.742 / 2.0;
          parTube[0] = 0.0;
          parTube[1] = 0.3 / 2.0; // Thickness of the cooling pipes
          parTube[2] = kMCMx / 2.0;
          TVirtualMC::GetMC()->Gsposp("UTCP", iCopy + iMCMrow * 10 + iMCMcol + 50, cTagV, xpos, ypos + 1.0, zpos,
                                      matrix[2], "ONLY", parTube, kNparTube);
          TVirtualMC::GetMC()->Gsposp("UTCP", iCopy + iMCMrow * 10 + iMCMcol + 500, cTagV, xpos, ypos + 2.0, zpos,
                                      matrix[2], "ONLY", parTube, kNparTube);
        }
      }
    }
  }

  //
  // The DCS boards
  //

  const float kDCSx = 9.0;
  const float kDCSy = 14.5;
  const float kDCSz = 0.3;

  const float kDCSpcTh = 0.15;
  const float kDCScuTh = 0.01;
  const float kDCScoTh = 0.04;

  // The mother volume for the DCSs (air)
  const int kNparDCS = 3;
  float parDCS[kNparDCS];
  parDCS[0] = kDCSx / 2.0;
  parDCS[1] = kDCSy / 2.0;
  parDCS[2] = kDCSz / 2.0;
  createVolume("UDCS", "BOX", idtmed[2], parDCS, kNparDCS);

  // The DCS carrier G10 layer
  parDCS[0] = kDCSx / 2.0;
  parDCS[1] = kDCSy / 2.0;
  parDCS[2] = kDCSpcTh / 2.0;
  createVolume("UDC1", "BOX", idtmed[19], parDCS, kNparDCS);
  // The DCS carrier Cu layer
  parDCS[0] = kDCSx / 2.0;
  parDCS[1] = kDCSy / 2.0;
  parDCS[2] = kDCScuTh / 2.0;
  createVolume("UDC2", "BOX", idtmed[18], parDCS, kNparDCS);
  // The aluminum of the cooling plates
  parDCS[0] = 5.0 / 2.0;
  parDCS[1] = 5.0 / 2.0;
  parDCS[2] = kDCScoTh / 2.0;
  createVolume("UDC3", "BOX", idtmed[24], parDCS, kNparDCS);

  // Put the DCS material inside the DCS mother volume
  xpos = 0.0;
  ypos = 0.0;
  zpos = -kDCSz / 2.0 + kDCSpcTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UDC1", 1, "UDCS", xpos, ypos, zpos, 0, "ONLY");
  zpos += kDCSpcTh / 2.0 + kDCScuTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UDC2", 1, "UDCS", xpos, ypos, zpos, 0, "ONLY");
  zpos += kDCScuTh / 2.0 + kDCScoTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UDC3", 1, "UDCS", xpos, ypos, zpos, 0, "ONLY");

  // Put the DCS board in the chamber services mother volume
  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);
      int iCopy = iDet + 1;
      xpos = CWIDTH[ilayer] / 2.0 -
             1.9 * (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)getRowMax(ilayer, istack, 0));
      ypos = 0.05 * CLENGTH[ilayer][istack];
      zpos = kDCSz / 2.0 - CSVH / 2.0;
      snprintf(cTagV, kTag, "UU%02d", iDet);
      TVirtualMC::GetMC()->Gspos("UDCS", iCopy, cTagV, xpos, ypos, zpos, 0, "ONLY");
    }
  }

  //
  // The ORI boards
  //

  const float kORIx = 4.2;
  const float kORIy = 13.5;
  const float kORIz = 0.3;

  const float kORIpcTh = 0.15;
  const float kORIcuTh = 0.01;
  const float kORIcoTh = 0.04;

  // The mother volume for the ORIs (air)
  const int kNparORI = 3;
  float parORI[kNparORI];
  parORI[0] = kORIx / 2.0;
  parORI[1] = kORIy / 2.0;
  parORI[2] = kORIz / 2.0;
  createVolume("UORI", "BOX", idtmed[2], parORI, kNparORI);

  // The ORI carrier G10 layer
  parORI[0] = kORIx / 2.0;
  parORI[1] = kORIy / 2.0;
  parORI[2] = kORIpcTh / 2.0;
  createVolume("UOR1", "BOX", idtmed[19], parORI, kNparORI);
  // The ORI carrier Cu layer
  parORI[0] = kORIx / 2.0;
  parORI[1] = kORIy / 2.0;
  parORI[2] = kORIcuTh / 2.0;
  createVolume("UOR2", "BOX", idtmed[18], parORI, kNparORI);
  // The aluminum of the cooling plates
  parORI[0] = kORIx / 2.0;
  parORI[1] = kORIy / 2.0;
  parORI[2] = kORIcoTh / 2.0;
  createVolume("UOR3", "BOX", idtmed[24], parORI, kNparORI);

  // Put the ORI material inside the ORI mother volume
  xpos = 0.0;
  ypos = 0.0;
  zpos = -kORIz / 2.0 + kORIpcTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UOR1", 1, "UORI", xpos, ypos, zpos, 0, "ONLY");
  zpos += kORIpcTh / 2.0 + kORIcuTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UOR2", 1, "UORI", xpos, ypos, zpos, 0, "ONLY");
  zpos += kORIcuTh / 2.0 + kORIcoTh / 2.0;
  TVirtualMC::GetMC()->Gspos("UOR3", 1, "UORI", xpos, ypos, zpos, 0, "ONLY");

  // Put the ORI board in the chamber services mother volume
  for (istack = 0; istack < kNstack; istack++) {
    for (ilayer = 0; ilayer < kNlayer; ilayer++) {
      int iDet = getDetectorSec(ilayer, istack);
      int iCopy = iDet + 1;
      xpos = CWIDTH[ilayer] / 2.0 -
             1.92 * (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)getRowMax(ilayer, istack, 0));
      ypos = -16.0;
      zpos = kORIz / 2.0 - CSVH / 2.0;
      snprintf(cTagV, kTag, "UU%02d", iDet);
      TVirtualMC::GetMC()->Gspos("UORI", iCopy, cTagV, xpos, ypos, zpos, 0, "ONLY");
      xpos = -CWIDTH[ilayer] / 2.0 +
             3.8 * (getChamberLength(ilayer, istack) - 2.0 * RPADW) / ((float)getRowMax(ilayer, istack, 0));
      ypos = -16.0;
      zpos = kORIz / 2.0 - CSVH / 2.0;
      snprintf(cTagV, kTag, "UU%02d", iDet);
      TVirtualMC::GetMC()->Gspos("UORI", iCopy + kNdet, cTagV, xpos, ypos, zpos, 0, "ONLY");
    }
  }

  //
  // Services in front of the super module
  //

  // Gas in-/outlet pipes (INOX)
  parTube[0] = 0.0;
  parTube[1] = 0.0;
  parTube[2] = 0.0;
  createVolume("UTG3", "TUBE", idtmed[8], parTube, 0);
  // The gas inside the in-/outlet pipes (Xe)
  parTube[0] = 0.0;
  parTube[1] = 1.2 / 2.0;
  parTube[2] = -1.0;
  createVolume("UTG4", "TUBE", idtmed[9], parTube, kNparTube);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTG4", 1, "UTG3", xpos, ypos, zpos, 0, "ONLY");
  for (ilayer = 0; ilayer < kNlayer - 1; ilayer++) {
    xpos = 0.0;
    ypos = CLENGTH[ilayer][2] / 2.0 + CLENGTH[ilayer][1] + CLENGTH[ilayer][0];
    zpos = 9.0 - SHEIGHT / 2.0 + ilayer * (CH + VSPACE);
    parTube[0] = 0.0;
    parTube[1] = 1.5 / 2.0;
    parTube[2] = CWIDTH[ilayer] / 2.0 - 2.5;
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1, "UTI1", xpos, ypos, zpos, matrix[2], "ONLY", parTube, kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 1 * kNlayer, "UTI1", xpos, -ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 2 * kNlayer, "UTI2", xpos, ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 3 * kNlayer, "UTI2", xpos, -ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 4 * kNlayer, "UTI3", xpos, ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 5 * kNlayer, "UTI3", xpos, -ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 6 * kNlayer, "UTI4", xpos, ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
    TVirtualMC::GetMC()->Gsposp("UTG3", ilayer + 1 + 7 * kNlayer, "UTI4", xpos, -ypos, zpos, matrix[2], "ONLY", parTube,
                                kNparTube);
  }

  // Gas distribution box
  parBox[0] = 14.50 / 2.0;
  parBox[1] = 4.52 / 2.0;
  parBox[2] = 5.00 / 2.0;
  createVolume("UTGD", "BOX ", idtmed[8], parBox, kNparBox);
  parBox[0] = 14.50 / 2.0;
  parBox[1] = 4.00 / 2.0;
  parBox[2] = 4.40 / 2.0;
  createVolume("UTGI", "BOX ", idtmed[9], parBox, kNparBox);
  parTube[0] = 0.0;
  parTube[1] = 4.0 / 2.0;
  parTube[2] = 8.0 / 2.0;
  createVolume("UTGT", "TUBE", idtmed[8], parTube, kNparTube);
  parTube[0] = 0.0;
  parTube[1] = 3.4 / 2.0;
  parTube[2] = 8.0 / 2.0;
  createVolume("UTGG", "TUBE", idtmed[9], parTube, kNparTube);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTGI", 1, "UTGD", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTGG", 1, "UTGT", xpos, ypos, zpos, 0, "ONLY");
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTGD", 1, "UTF1", xpos, ypos, zpos, 0, "ONLY");
  xpos = -3.0;
  ypos = 0.0;
  zpos = 6.5;
  TVirtualMC::GetMC()->Gspos("UTGT", 1, "UTF1", xpos, ypos, zpos, 0, "ONLY");
  xpos = -11.25;
  ypos = 0.0;
  zpos = 0.5;
  TVirtualMC::GetMC()->Gspos("UTGT", 3, "UTF1", xpos, ypos, zpos, matrix[2], "ONLY");
  xpos = 11.25;
  ypos = 0.0;
  zpos = 0.5;
  TVirtualMC::GetMC()->Gspos("UTGT", 5, "UTF1", xpos, ypos, zpos, matrix[2], "ONLY");

  // Cooling manifolds
  parBox[0] = 5.0 / 2.0;
  parBox[1] = 23.0 / 2.0;
  parBox[2] = 70.0 / 2.0;
  createVolume("UTCM", "BOX ", idtmed[2], parBox, kNparBox);
  parBox[0] = 5.0 / 2.0;
  parBox[1] = 5.0 / 2.0;
  parBox[2] = 70.0 / 2.0;
  createVolume("UTCA", "BOX ", idtmed[8], parBox, kNparBox);
  parBox[0] = 5.0 / 2.0 - 0.3;
  parBox[1] = 5.0 / 2.0 - 0.3;
  parBox[2] = 70.0 / 2.0 - 0.3;
  createVolume("UTCW", "BOX ", idtmed[14], parBox, kNparBox);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTCW", 1, "UTCA", xpos, ypos, zpos, 0, "ONLY");
  xpos = 0.0;
  ypos = 5.0 / 2.0 - 23.0 / 2.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTCA", 1, "UTCM", xpos, ypos, zpos, 0, "ONLY");
  parTube[0] = 0.0;
  parTube[1] = 3.0 / 2.0;
  parTube[2] = 18.0 / 2.0;
  createVolume("UTCO", "TUBE", idtmed[8], parTube, kNparTube);
  parTube[0] = 0.0;
  parTube[1] = 3.0 / 2.0 - 0.3;
  parTube[2] = 18.0 / 2.0;
  createVolume("UTCL", "TUBE", idtmed[14], parTube, kNparTube);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTCL", 1, "UTCO", xpos, ypos, zpos, 0, "ONLY");
  xpos = 0.0;
  ypos = 2.5;
  zpos = -70.0 / 2.0 + 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 1, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 2, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 3, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 4, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 5, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 6, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 7, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");
  zpos += 7.0;
  TVirtualMC::GetMC()->Gspos("UTCO", 8, "UTCM", xpos, ypos, zpos, matrix[4], "ONLY");

  xpos = 40.0;
  ypos = FLENGTH / 2.0 - 23.0 / 2.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTCM", 1, "UTF1", xpos, ypos, zpos, matrix[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("UTCM", 2, "UTF1", -xpos, ypos, zpos, matrix[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("UTCM", 3, "UTF2", xpos, -ypos, zpos, matrix[5], "ONLY");
  TVirtualMC::GetMC()->Gspos("UTCM", 4, "UTF2", -xpos, -ypos, zpos, matrix[6], "ONLY");

  // Power connection boards (Cu)
  parBox[0] = 0.5 / 2.0;
  parBox[1] = 15.0 / 2.0;
  parBox[2] = 7.0 / 2.0;
  createVolume("UTPC", "BOX ", idtmed[25], parBox, kNparBox);
  for (ilayer = 0; ilayer < kNlayer - 1; ilayer++) {
    xpos = CWIDTH[ilayer] / 2.0 + kPWRwid / 2.0;
    ypos = 0.0;
    zpos = VROCSM + SMPLTT + kPWRhgtA / 2.0 - SHEIGHT / 2.0 + kPWRposz + (ilayer + 1) * (CH + VSPACE);
    TVirtualMC::GetMC()->Gspos("UTPC", ilayer, "UTF1", xpos, ypos, zpos, matrix[0], "ONLY");
    TVirtualMC::GetMC()->Gspos("UTPC", ilayer + kNlayer, "UTF1", -xpos, ypos, zpos, matrix[1], "ONLY");
  }
  xpos = CWIDTH[5] / 2.0 + kPWRhgtA / 2.0 - 2.0;
  ypos = 0.0;
  zpos = SHEIGHT / 2.0 - SMPLTT - 2.0;
  TVirtualMC::GetMC()->Gspos("UTPC", 5, "UTF1", xpos, ypos, zpos, matrix[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("UTPC", 5 + kNlayer, "UTF1", -xpos, ypos, zpos, matrix[3], "ONLY");

  // Power connection panel (Al)
  parBox[0] = 60.0 / 2.0;
  parBox[1] = 10.0 / 2.0;
  parBox[2] = 3.0 / 2.0;
  createVolume("UTPP", "BOX ", idtmed[1], parBox, kNparBox);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 18.0;
  TVirtualMC::GetMC()->Gspos("UTPP", 1, "UTF1", xpos, ypos, zpos, 0, "ONLY");

  //
  // Electronics boxes
  //

  // Casing (INOX)
  parBox[0] = 60.0 / 2.0;
  parBox[1] = 10.0 / 2.0;
  parBox[2] = 6.0 / 2.0;
  createVolume("UTE1", "BOX ", idtmed[8], parBox, kNparBox);
  // Interior (air)
  parBox[0] = parBox[0] - 0.5;
  parBox[1] = parBox[1] - 0.5;
  parBox[2] = parBox[2] - 0.5;
  createVolume("UTE2", "BOX ", idtmed[2], parBox, kNparBox);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTE2", 1, "UTE1", xpos, ypos, zpos, 0, "ONLY");
  xpos = 0.0;
  ypos = SLENGTH / 2.0 - 10.0 / 2.0 - 3.0;
  zpos = -SHEIGHT / 2.0 + 6.0 / 2.0 + 1.0;
  TVirtualMC::GetMC()->Gspos("UTE1", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE1", 2, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE1", 3, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE1", 4, "UTI4", xpos, ypos, zpos, 0, "ONLY");

  // Casing (INOX)
  parBox[0] = 50.0 / 2.0;
  parBox[1] = 15.0 / 2.0;
  parBox[2] = 20.0 / 2.0;
  createVolume("UTE3", "BOX ", idtmed[8], parBox, kNparBox);
  // Interior (air)
  parBox[0] = parBox[0] - 0.5;
  parBox[1] = parBox[1] - 0.5;
  parBox[2] = parBox[2] - 0.5;
  createVolume("UTE4", "BOX ", idtmed[2], parBox, kNparBox);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTE4", 1, "UTE3", xpos, ypos, zpos, 0, "ONLY");
  xpos = 0.0;
  ypos = -SLENGTH / 2.0 + 15.0 / 2.0 + 3.0;
  zpos = -SHEIGHT / 2.0 + 20.0 / 2.0 + 1.0;
  TVirtualMC::GetMC()->Gspos("UTE3", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE3", 2, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE3", 3, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE3", 4, "UTI4", xpos, ypos, zpos, 0, "ONLY");

  // Casing (INOX)
  parBox[0] = 20.0 / 2.0;
  parBox[1] = 7.0 / 2.0;
  parBox[2] = 20.0 / 2.0;
  createVolume("UTE5", "BOX ", idtmed[8], parBox, kNparBox);
  // Interior (air)
  parBox[0] = parBox[0] - 0.5;
  parBox[1] = parBox[1] - 0.5;
  parBox[2] = parBox[2] - 0.5;
  createVolume("UTE6", "BOX ", idtmed[2], parBox, kNparBox);
  xpos = 0.0;
  ypos = 0.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTE6", 1, "UTE5", xpos, ypos, zpos, 0, "ONLY");
  xpos = 20.0;
  ypos = -SLENGTH / 2.0 + 7.0 / 2.0 + 3.0;
  zpos = 0.0;
  TVirtualMC::GetMC()->Gspos("UTE5", 1, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 2, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 3, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 4, "UTI4", xpos, ypos, zpos, 0, "ONLY");
  xpos = -xpos;
  TVirtualMC::GetMC()->Gspos("UTE5", 5, "UTI1", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 6, "UTI2", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 7, "UTI3", xpos, ypos, zpos, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("UTE5", 8, "UTI4", xpos, ypos, zpos, 0, "ONLY");
}

//_____________________________________________________________________________
void TRDGeometry::assembleChamber(int ilayer, int istack)
{
  //
  // Group volumes UA, UD, UF, UU into an assembly that defines the
  // alignable volume of a single readout chamber
  //

  const int kTag = 100;
  char cTagV[kTag];
  char cTagM[kTag];

  double xpos = 0.0;
  double ypos = 0.0;
  double zpos = 0.0;

  int idet = getDetectorSec(ilayer, istack);

  // Create the assembly for a given ROC
  snprintf(cTagM, kTag, "UT%02d", idet);
  TGeoVolume* roc = new TGeoVolumeAssembly(cTagM);

  // Add the lower part of the chamber (aluminum frame),
  // including radiator and drift region
  xpos = 0.0;
  ypos = 0.0;
  zpos = CRAH / 2.0 + CDRH / 2.0 - CHSV / 2.0;
  snprintf(cTagV, kTag, "UA%02d", idet);
  TGeoVolume* rocA = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocA, 1, new TGeoTranslation(xpos, ypos, zpos));

  // Add the additional aluminum ledges
  xpos = CWIDTH[ilayer] / 2.0 + CALWMOD / 2.0;
  ypos = 0.0;
  zpos = CRAH + CDRH - CALZPOS - CALHMOD / 2.0 - CHSV / 2.0;
  snprintf(cTagV, kTag, "UZ%02d", idet);
  TGeoVolume* rocZ = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocZ, 1, new TGeoTranslation(xpos, ypos, zpos));
  roc->AddNode(rocZ, 2, new TGeoTranslation(-xpos, ypos, zpos));

  // Add the additional wacosit ledges
  xpos = CWIDTH[ilayer] / 2.0 + CWSW / 2.0;
  ypos = 0.0;
  zpos = CRAH + CDRH - CWSH / 2.0 - CHSV / 2.0;
  snprintf(cTagV, kTag, "UP%02d", idet);
  TGeoVolume* rocP = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocP, 1, new TGeoTranslation(xpos, ypos, zpos));
  roc->AddNode(rocP, 2, new TGeoTranslation(-xpos, ypos, zpos));

  // Add the middle part of the chamber (G10 frame),
  // including amplification region
  xpos = 0.0;
  ypos = 0.0;
  zpos = CAMH / 2.0 + CRAH + CDRH - CHSV / 2.0;
  snprintf(cTagV, kTag, "UD%02d", idet);
  TGeoVolume* rocD = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocD, 1, new TGeoTranslation(xpos, ypos, zpos));

  // Add the upper part of the chamber (aluminum frame),
  // including back panel and FEE
  xpos = 0.0;
  ypos = 0.0;
  zpos = CROH / 2.0 + CAMH + CRAH + CDRH - CHSV / 2.0;
  snprintf(cTagV, kTag, "UF%02d", idet);
  TGeoVolume* rocF = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocF, 1, new TGeoTranslation(xpos, ypos, zpos));

  // Add the volume with services on top of the back panel
  xpos = 0.0;
  ypos = 0.0;
  zpos = CSVH / 2.0 + CROH + CAMH + CRAH + CDRH - CHSV / 2.0;
  snprintf(cTagV, kTag, "UU%02d", idet);
  TGeoVolume* rocU = gGeoManager->GetVolume(cTagV);
  roc->AddNode(rocU, 1, new TGeoTranslation(xpos, ypos, zpos));

  // Place the ROC assembly into the super modules
  xpos = 0.0;
  ypos = 0.0;
  ypos = CLENGTH[ilayer][0] + CLENGTH[ilayer][1] + CLENGTH[ilayer][2] / 2.0;
  for (int ic = 0; ic < istack; ic++) {
    ypos -= CLENGTH[ilayer][ic];
  }
  ypos -= CLENGTH[ilayer][istack] / 2.0;
  zpos = VROCSM + SMPLTT + CHSV / 2.0 - SHEIGHT / 2.0 + ilayer * (CH + VSPACE);
  TGeoVolume* sm1 = gGeoManager->GetVolume("UTI1");
  TGeoVolume* sm2 = gGeoManager->GetVolume("UTI2");
  TGeoVolume* sm3 = gGeoManager->GetVolume("UTI3");
  TGeoVolume* sm4 = gGeoManager->GetVolume("UTI4");
  sm1->AddNode(roc, 1, new TGeoTranslation(xpos, ypos, zpos));
  sm2->AddNode(roc, 1, new TGeoTranslation(xpos, ypos, zpos));
  if (istack != 2) {
    // w/o middle stack
    sm3->AddNode(roc, 1, new TGeoTranslation(xpos, ypos, zpos));
  }
  if (!((ilayer == 4) && (istack == 4))) {
    // Sector 17 w/o L4S4 chamber
    sm4->AddNode(roc, 1, new TGeoTranslation(xpos, ypos, zpos));
  }
}

void TRDGeometry::fillMatrixCache(int mask)
{
  if (mask & o2::utils::bit2Mask(o2::TransformType::T2L)) {
    useT2LCache();
  }
  if (mask & o2::utils::bit2Mask(o2::TransformType::L2G)) {
    useL2GCache();
  }
  if (mask & o2::utils::bit2Mask(o2::TransformType::T2G)) {
    useT2GCache();
  }
  if (mask & o2::utils::bit2Mask(o2::TransformType::T2GRot)) {
    useT2GRotCache();
  }

  std::string volPath;
  const std::string vpStr{"ALIC_1/B077_1/BSEGMO"};
  const std::string vpApp1{"_1/BTRD"};
  const std::string vpApp2{"_1"};
  const std::string vpApp3a{"/UTR1_1/UTS1_1/UTI1_1"};
  const std::string vpApp3b{"/UTR2_1/UTS2_1/UTI2_1"};
  const std::string vpApp3c{"/UTR3_1/UTS3_1/UTI3_1"};
  const std::string vpApp3d{"/UTR4_1/UTS4_1/UTI4_1"};

  for (int ilayer = 0; ilayer < kNlayer; ilayer++) {
    for (int isector = 0; isector < kNsector; isector++) {
      for (int istack = 0; istack < kNstack; istack++) {
        Int_t lid = getDetector(ilayer, istack, isector);
        // Check for disabled supermodules
        volPath = vpStr;
        volPath += std::to_string(isector);
        volPath += vpApp1;
        volPath += std::to_string(isector);
        volPath += vpApp2;
        switch (isector) {
          case 17:
            if ((istack == 4) && (ilayer == 4)) {
              continue;
            }
            volPath += vpApp3d;
            break;
          case 13:
          case 14:
          case 15:
            // Check for holes in from of PHOS
            if (istack == 2) {
              continue;
            }
            volPath += vpApp3c;
            break;
          case 11:
          case 12:
            volPath += vpApp3b;
            break;
          default:
            volPath += vpApp3a;
        };
        if (!gGeoManager->CheckPath(volPath.c_str())) {
          continue;
        }
        const auto m = o2::base::GeometryManager::getMatrix(o2::detectors::DetID::TRD, lid);
        TGeoHMatrix rotMatrix;
        rotMatrix.RotateX(-90);
        rotMatrix.RotateY(-90);
        rotMatrix.MultiplyLeft(m);
        const TGeoHMatrix& t2l = rotMatrix.Inverse();
        if (mask & o2::utils::bit2Mask(o2::TransformType::L2G)) {
          setMatrixL2G(Mat3D(t2l), lid);
        }

        Double_t sectorAngle = 20.0 * (isector % 18) + 10.0;
        TGeoHMatrix rotSector;
        rotSector.RotateZ(sectorAngle);
        if (mask & o2::utils::bit2Mask(o2::TransformType::T2G)) {
          setMatrixT2G(Mat3D(rotSector), lid);
        }
        if (mask & o2::utils::bit2Mask(o2::TransformType::T2GRot)) {
          setMatrixT2GRot(Rot2D(sectorAngle), lid);
        }
        if (mask & o2::utils::bit2Mask(o2::TransformType::T2L)) {
          const TGeoMatrix& inv = rotSector.Inverse();
          rotMatrix.MultiplyLeft(&inv);
          setMatrixT2L(Mat3D(rotMatrix.Inverse()), lid);
        }
      }
    }
  }
}

//_____________________________________________________________________________
void TRDGeometry::addAlignableVolumes() const
{
  //
  // define alignable volumes of the TRD
  //
  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded";
  }

  std::string volPath;
  std::string vpStr{"ALIC_1/B077_1/BSEGMO"};
  const std::string vpApp1{"_1/BTRD"};
  const std::string vpApp2{"_1"};
  const std::string vpApp3a{"/UTR1_1/UTS1_1/UTI1_1"};
  const std::string vpApp3b{"/UTR2_1/UTS2_1/UTI2_1"};
  const std::string vpApp3c{"/UTR3_1/UTS3_1/UTI3_1"};
  const std::string vpApp3d{"/UTR4_1/UTS4_1/UTI4_1"};
  std::string symName;

  // in opposite to AliGeomManager, we use consecutive numbering of modules through whole TRD
  int volid = -1;

  // The super modules
  // The symbolic names are: TRD/sm00 ... TRD/sm17
  for (int isector = 0; isector < kNsector; isector++) {
    volPath = vpStr;
    volPath += std::to_string(isector);
    volPath += vpApp1;
    volPath += std::to_string(isector);
    volPath += vpApp2;
    symName = Form("TRD/sm%02d", isector);
    gGeoManager->SetAlignableEntry(symName.c_str(), volPath.c_str());
  }

  // The readout chambers
  // The symbolic names are: TRD/sm00/st0/pl0 ... TRD/sm17/st4/pl5

  for (int isector = 0; isector < kNsector; isector++) {
    if (!getSMstatus(isector))
      continue;
    for (int ilayer = 0; ilayer < kNlayer; ilayer++) {
      for (int istack = 0; istack < kNstack; istack++) {
        Int_t lid = getDetector(ilayer, istack, isector);
        int idet = getDetectorSec(ilayer, istack);

        // Check for disabled supermodules
        volPath = vpStr;
        volPath += std::to_string(isector);
        volPath += vpApp1;
        volPath += std::to_string(isector);
        volPath += vpApp2;
        switch (isector) {
          case 17:
            if ((istack == 4) && (ilayer == 4)) {
              continue;
            }
            volPath += vpApp3d;
            break;
          case 13:
          case 14:
          case 15:
            // Check for holes in from of PHOS
            if (istack == 2) {
              continue;
            }
            volPath += vpApp3c;
            break;
          case 11:
          case 12:
            volPath += vpApp3b;
            break;
          default:
            volPath += vpApp3a;
        };
        volPath += Form("/UT%02d_1", idet);

        symName = Form("TRD/sm%02d/st%d/pl%d", isector, istack, ilayer);
        int modID = o2::base::GeometryManager::getSensID(o2::detectors::DetID::TRD, lid);

        TGeoPNEntry* alignableEntry = gGeoManager->SetAlignableEntry(symName.c_str(), volPath.c_str(), modID);

        // Add the tracking to local matrix
        if (alignableEntry) {
          TGeoHMatrix* globMatrix = alignableEntry->GetGlobalOrig();
          Double_t sectorAngle = 20.0 * (isector % 18) + 10.0;
          TGeoHMatrix* t2lMatrix = new TGeoHMatrix();
          t2lMatrix->RotateZ(sectorAngle);
          const TGeoHMatrix& globmatrixi = globMatrix->Inverse();
          t2lMatrix->MultiplyLeft(&globmatrixi);
          alignableEntry->SetMatrix(t2lMatrix);
        } else {
          LOG(ERROR) << "Alignable entry is not valid: ModID:" << modID << " Sector:" << isector << " Lr:" << ilayer
                     << " Stack: " << istack << " name: " << symName.c_str() << " vol: " << volPath.c_str();
        }
      }
    }
  }
}

//_____________________________________________________________________________
bool TRDGeometry::createClusterMatrixArray()
{
  if (!gGeoManager) {
    LOG(ERROR) << "Geometry is not loaded yet";
    return false;
  }
  if (isBuilt()) {
    LOG(WARNING) << "Already built";
    return true; // already initialized
  }

  setSize(MAXMATRICES, kNdet); //Only MAXMATRICES=521 of kNdet matrices are filled
  fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L) | o2::utils::bit2Mask(o2::TransformType::L2G) |
                  o2::utils::bit2Mask(o2::TransformType::T2G) | o2::utils::bit2Mask(o2::TransformType::T2GRot));
  return true;
}

//_____________________________________________________________________________
bool TRDGeometry::chamberInGeometry(int det) const
{
  //
  // Checks whether the given detector is part of the current geometry
  //

  if (!isMatrixAvailable(det)) {
    return false;
  } else {
    return true;
  }
}
