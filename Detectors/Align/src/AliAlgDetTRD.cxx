// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDetTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD detector wrapper

#include "Align/AliAlgDetTRD.h"
#include "Align/AliAlgVol.h"
#include "Align/AliAlgSensTRD.h"
#include "Align/AliAlgSteer.h"
//#include "AliGeomManager.h"
//#include "AliESDtrack.h"
//#include "AliTRDgeometry.h"
#include <TGeoManager.h>
#include <TMath.h>

using namespace TMath;

ClassImp(o2::align::AliAlgDetTRD);

namespace o2
{
namespace align
{

const char* AliAlgDetTRD::fgkCalibDOFName[AliAlgDetTRD::kNCalibParams] = {"DZdTglNRC", "DVDriftT"};

//____________________________________________
AliAlgDetTRD::AliAlgDetTRD(const char* title)
  : AliAlgDet(), fNonRCCorrDzDtgl(0), fCorrDVT(0)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::getDetNameByDetID(AliAlgSteer::kTRD), title);
  setDetID(AliAlgSteer::kTRD);
  fExtraErrRC[0] = fExtraErrRC[1] = 0;
  //
  // ad hoc correction
  SetNonRCCorrDzDtgl();
  SetExtraErrRC();
  mNCalibDOF = kNCalibParams;
}

//____________________________________________
AliAlgDetTRD::~AliAlgDetTRD()
{
  // d-tor
}

//____________________________________________
void AliAlgDetTRD::defineVolumes()
{
  // define TRD volumes
  //
  const int kNSect = 18, kNStacks = 5, kNLayers = 6;
  AliAlgSensTRD* chamb = 0;
  //
  int labDet = getDetLabel();
  //  AddVolume( volTRD = new AliAlgVol("TRD") ); // no main volume, why?
  AliAlgVol* sect[kNSect] = {0};
  //
  for (int ilr = 0; ilr < kNLayers; ilr++) {            // layer
    for (int ich = 0; ich < kNStacks * kNSect; ich++) { // chamber
      int isector = ich / AliTRDgeometry::Nstack();
      int istack = ich % AliTRDgeometry::Nstack();
      //int lid       = AliTRDgeometry::GetDetector(ilr,istack,isector);
      int iid = labDet + (1 + ilr) * 10000 + (1 + isector) * 100 + (1 + istack);
      const char* symname = Form("TRD/sm%02d/st%d/pl%d", isector, istack, ilr);
      if (!gGeoManager->GetAlignableEntry(symname))
        continue;
      uint16_t vid = AliGeomManager::LayerToVolUID(AliGeomManager::kTRD1 + ilr, ich);
      addVolume(chamb = new AliAlgSensTRD(symname, vid, iid /*lid*/, isector));
      iid = labDet + (1 + isector) * 100;
      if (!sect[isector])
        sect[isector] = new AliAlgVol(Form("TRD/sm%02d", isector), iid);
      chamb->setParent(sect[isector]);
    } // chamber
  }   // layer
  //
  for (int isc = 0; isc < kNSect; isc++) {
    if (sect[isc])
      addVolume(sect[isc]);
  }
  //
}

//____________________________________________
bool AliAlgDetTRD::AcceptTrack(const AliESDtrack* trc, int trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return false;
  if (trc->GetTRDntracklets() < mNPointsSel[trtype])
    return false;
  return true;
}

//__________________________________________
//____________________________________________
void AliAlgDetTRD::Print(const Option_t* opt) const
{
  // print info
  AliAlgDet::Print(opt);
  printf("Extra error for RC tracklets: Y:%e Z:%e\n", fExtraErrRC[0], fExtraErrRC[1]);
}

const char* AliAlgDetTRD::getCalibDOFName(int i) const
{
  // return calibration DOF name
  return i < kNCalibParams ? fgkCalibDOFName[i] : 0;
}

//______________________________________________________
void AliAlgDetTRD::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  AliAlgDet::writePedeInfo(parOut, opt);
  //
  // write calibration parameters
  enum { kOff,
         kOn,
         kOnOn };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyParam = "parameter";
  //
  fprintf(parOut, "%s%s %s\t %d calibraction params for %s\n", comment[kOff], kKeyParam, comment[kOnOn],
          getNCalibDOFs(), GetName());
  //
  for (int ip = 0; ip < getNCalibDOFs(); ip++) {
    int cmt = isCondDOF(ip) ? kOff : kOn;
    fprintf(parOut, "%s %9d %+e %+e\t%s %s p%d\n", comment[cmt], getParLab(ip),
            getParVal(ip), getParErr(ip), comment[kOnOn], isFreeDOF(ip) ? "  " : "FX", ip);
  }
  //
}

//_______________________________________________________
double AliAlgDetTRD::getCalibDOFVal(int id) const
{
  // return preset value of calibration dof
  double val = 0;
  switch (id) {
    case kCalibNRCCorrDzDtgl:
      val = GetNonRCCorrDzDtgl();
      break;
    case kCalibDVT:
      val = GetCorrDVT();
      break;
    default:
      break;
  };
  return val;
}

//_______________________________________________________
double AliAlgDetTRD::getCalibDOFValWithCal(int id) const
{
  // return preset value of calibration dof + mp correction
  return getCalibDOFVal(id) + getParVal(id);
}

} // namespace align
} // namespace o2
