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

#include "AliAlgDetTRD.h"
#include "AliAlgVol.h"
#include "AliAlgSensTRD.h"
#include "AliAlgSteer.h"
#include "AliGeomManager.h"
#include "AliESDtrack.h"
#include "AliTRDgeometry.h"
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
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kTRD), title);
  SetDetID(AliAlgSteer::kTRD);
  fExtraErrRC[0] = fExtraErrRC[1] = 0;
  //
  // ad hoc correction
  SetNonRCCorrDzDtgl();
  SetExtraErrRC();
  fNCalibDOF = kNCalibParams;
}

//____________________________________________
AliAlgDetTRD::~AliAlgDetTRD()
{
  // d-tor
}

//____________________________________________
void AliAlgDetTRD::DefineVolumes()
{
  // define TRD volumes
  //
  const int kNSect = 18, kNStacks = 5, kNLayers = 6;
  AliAlgSensTRD* chamb = 0;
  //
  int labDet = GetDetLabel();
  //  AddVolume( volTRD = new AliAlgVol("TRD") ); // no main volume, why?
  AliAlgVol* sect[kNSect] = {0};
  //
  for (int ilr = 0; ilr < kNLayers; ilr++) {            // layer
    for (int ich = 0; ich < kNStacks * kNSect; ich++) { // chamber
      Int_t isector = ich / AliTRDgeometry::Nstack();
      Int_t istack = ich % AliTRDgeometry::Nstack();
      //Int_t lid       = AliTRDgeometry::GetDetector(ilr,istack,isector);
      int iid = labDet + (1 + ilr) * 10000 + (1 + isector) * 100 + (1 + istack);
      const char* symname = Form("TRD/sm%02d/st%d/pl%d", isector, istack, ilr);
      if (!gGeoManager->GetAlignableEntry(symname))
        continue;
      UShort_t vid = AliGeomManager::LayerToVolUID(AliGeomManager::kTRD1 + ilr, ich);
      AddVolume(chamb = new AliAlgSensTRD(symname, vid, iid /*lid*/, isector));
      iid = labDet + (1 + isector) * 100;
      if (!sect[isector])
        sect[isector] = new AliAlgVol(Form("TRD/sm%02d", isector), iid);
      chamb->SetParent(sect[isector]);
    } // chamber
  }   // layer
  //
  for (int isc = 0; isc < kNSect; isc++) {
    if (sect[isc])
      AddVolume(sect[isc]);
  }
  //
}

//____________________________________________
Bool_t AliAlgDetTRD::AcceptTrack(const AliESDtrack* trc, Int_t trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return kFALSE;
  if (trc->GetTRDntracklets() < fNPointsSel[trtype])
    return kFALSE;
  return kTRUE;
}

//__________________________________________
//____________________________________________
void AliAlgDetTRD::Print(const Option_t* opt) const
{
  // print info
  AliAlgDet::Print(opt);
  printf("Extra error for RC tracklets: Y:%e Z:%e\n", fExtraErrRC[0], fExtraErrRC[1]);
}

const char* AliAlgDetTRD::GetCalibDOFName(int i) const
{
  // return calibration DOF name
  return i < kNCalibParams ? fgkCalibDOFName[i] : 0;
}

//______________________________________________________
void AliAlgDetTRD::WritePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  AliAlgDet::WritePedeInfo(parOut, opt);
  //
  // write calibration parameters
  enum { kOff,
         kOn,
         kOnOn };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyParam = "parameter";
  //
  fprintf(parOut, "%s%s %s\t %d calibraction params for %s\n", comment[kOff], kKeyParam, comment[kOnOn],
          GetNCalibDOFs(), GetName());
  //
  for (int ip = 0; ip < GetNCalibDOFs(); ip++) {
    int cmt = IsCondDOF(ip) ? kOff : kOn;
    fprintf(parOut, "%s %9d %+e %+e\t%s %s p%d\n", comment[cmt], GetParLab(ip),
            GetParVal(ip), GetParErr(ip), comment[kOnOn], IsFreeDOF(ip) ? "  " : "FX", ip);
  }
  //
}

//_______________________________________________________
Double_t AliAlgDetTRD::GetCalibDOFVal(int id) const
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
Double_t AliAlgDetTRD::GetCalibDOFValWithCal(int id) const
{
  // return preset value of calibration dof + mp correction
  return GetCalibDOFVal(id) + GetParVal(id);
}

} // namespace align
} // namespace o2
