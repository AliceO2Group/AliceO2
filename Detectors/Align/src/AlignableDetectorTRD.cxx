// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetectorTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD detector wrapper

#include "Align/AlignableDetectorTRD.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorTRD.h"
#include "Align/Controller.h"
//#include "AliGeomManager.h"
//#include "AliESDtrack.h"
//#include "AliTRDgeometry.h"
#include <TGeoManager.h>
#include <TMath.h>

using namespace TMath;

ClassImp(o2::align::AlignableDetectorTRD);

namespace o2
{
namespace align
{

const char* AlignableDetectorTRD::fgkCalibDOFName[AlignableDetectorTRD::kNCalibParams] = {"DZdTglNRC", "DVDriftT"};

//____________________________________________
AlignableDetectorTRD::AlignableDetectorTRD(const char* title)
  : AlignableDetector(), fNonRCCorrDzDtgl(0), fCorrDVT(0)
{
  // default c-tor
  SetNameTitle(Controller::getDetNameByDetID(Controller::kTRD), title);
  setDetID(Controller::kTRD);
  fExtraErrRC[0] = fExtraErrRC[1] = 0;
  //
  // ad hoc correction
  SetNonRCCorrDzDtgl();
  SetExtraErrRC();
  mNCalibDOF = kNCalibParams;
}

//____________________________________________
AlignableDetectorTRD::~AlignableDetectorTRD()
{
  // d-tor
}

//____________________________________________
void AlignableDetectorTRD::defineVolumes()
{
  // define TRD volumes
  //
  const int kNSect = 18, kNStacks = 5, kNLayers = 6;
  AlignableSensorTRD* chamb = 0;
  //
  int labDet = getDetLabel();
  //  AddVolume( volTRD = new AlignableVolume("TRD") ); // no main volume, why?
  AlignableVolume* sect[kNSect] = {0};
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
      addVolume(chamb = new AlignableSensorTRD(symname, vid, iid /*lid*/, isector));
      iid = labDet + (1 + isector) * 100;
      if (!sect[isector])
        sect[isector] = new AlignableVolume(Form("TRD/sm%02d", isector), iid);
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
bool AlignableDetectorTRD::AcceptTrack(const AliESDtrack* trc, int trtype) const
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
void AlignableDetectorTRD::Print(const Option_t* opt) const
{
  // print info
  AlignableDetector::Print(opt);
  printf("Extra error for RC tracklets: Y:%e Z:%e\n", fExtraErrRC[0], fExtraErrRC[1]);
}

const char* AlignableDetectorTRD::getCalibDOFName(int i) const
{
  // return calibration DOF name
  return i < kNCalibParams ? fgkCalibDOFName[i] : 0;
}

//______________________________________________________
void AlignableDetectorTRD::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  AlignableDetector::writePedeInfo(parOut, opt);
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
double AlignableDetectorTRD::getCalibDOFVal(int id) const
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
double AlignableDetectorTRD::getCalibDOFValWithCal(int id) const
{
  // return preset value of calibration dof + mp correction
  return getCalibDOFVal(id) + getParVal(id);
}

} // namespace align
} // namespace o2
