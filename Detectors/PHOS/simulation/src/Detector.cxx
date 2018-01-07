// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <map>
#include <algorithm>
#include <iomanip>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualMC.h"

#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairVolume.h"

#include "PHOSBase/Geometry.h"
#include "PHOSBase/Hit.h"
#include "PHOSSimulation/Detector.h"
#include "PHOSSimulation/GeometryParams.h"

#include "SimulationDataFormat/Stack.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/irange.hpp>

using namespace o2::PHOS;

ClassImp(Detector);

Detector::Detector(Bool_t active)
  : o2::Base::DetImpl<Detector>("PHS", active),
  mHits(new std::vector<Hit>),
  mCurrentTrackID(-1),    
  mCurrentCellID(-1),
  mCurentSuperParent(-1),   
  mCurrentHit(nullptr)          
{
//  using boost::algorithm::contains;
//  memset(mParEMOD, 0, sizeof(Double_t) * 5);

//  Geometry* geo = GetGeometry();
//  if (!geo)
//    LOG(FATAL) << "Geometry is nullptr" << std::endl;
//  std::string gn = geo->GetName();
//  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);


}

void Detector::Initialize() { 
  o2::Base::Detector::Initialize(); 
  Reset(); 
}

void Detector::EndOfEvent() { 
  // Sort Hits
  // Add duplicates if any
  // Apply Poisson spearing of light production



  Reset(); 
}
void Detector::Reset(){
  mSuperParents.clear(); 
//  mHits.clear();   
  mCurrentTrackID=-1 ;    
  mCurrentCellID=-1 ; 
  mCurentSuperParent=-1 ;   
  mCurrentHit=nullptr ;          
}

void Detector::Register(){
  FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
}
Bool_t Detector::ProcessHits(FairVolume* v)
{
 //1. Remember all particles first entered PHOS (active medium)
 //2. Collect all energy depositions in Cell by all secondaries from particle first entered PHOS  

  auto* mcapp = TVirtualMC::GetMC();

  // Check if this is first entered PHOS particle ("SuperParent")
  TVirtualMCStack * stack = mcapp->GetStack();
  const Int_t partID=stack->GetCurrentTrackNumber() ;
  Int_t superParent=-1; 
  if(partID != mCurrentTrackID){ //not same track as before, check: same SuperParent or new one?
    auto itTr=mSuperParents.find(partID) ;
    if(itTr==mSuperParents.end()){
      //Search parent
      Int_t parentID = stack->GetCurrentTrack()->GetMother(0) ;
      itTr=mSuperParents.find(parentID) ;
      if(itTr==mSuperParents.end()){ //Neither track or its parent found: new SuperParent
        mSuperParents[partID]=partID ;
        superParent=partID ;
      }
      else{ //parent found, this track - not
        superParent=itTr->second ;
        mSuperParents[partID]=superParent ;
      }
    }
    else{
      superParent=itTr->second ;      
    }
  }
  else{
    superParent=mCurentSuperParent ;
  }


  Double_t lostenergy = mcapp->Edep();
  if (lostenergy < DBL_EPSILON)
    return false; // do not create hits with zero energy deposition

  if(!mGeom)
    mGeom = Geometry::GetInstance();

//  if(strcmp(mcapp->CurrentVolName(),"PXTL")!=0) //Non need to check, alwais there...
//    return false ; //  We are not inside a PBWO crystal

  Int_t  moduleNumber ;   
  mcapp->CurrentVolOffID(10, moduleNumber) ; // get the PHOS module number ;
  Int_t strip ;
  mcapp->CurrentVolOffID(3, strip);
  Int_t cell ;
  mcapp->CurrentVolOffID(2, cell);
  Int_t detID = mGeom->RelToAbsId(moduleNumber,strip,cell);

  if(superParent==mCurentSuperParent && detID == mCurrentCellID && mCurrentHit) {
    //continue with current hit
    mCurrentHit->AddEnergyLoss(lostenergy);
    return true ;
  }

  //Create new Hit
  Double_t posX, posY, posZ, momX, momY, momZ, energy;
  mcapp->TrackPosition(posX, posY, posZ);
  mcapp->TrackMomentum(momX, momY, momZ, energy);
  Double_t estart = 0.;
  if(partID==superParent) //Store energy only if this is superParent, 
                          //if this is daughter entered new volume, we can not access true superparent energy/momentum 
     mcapp->Etot();
  Double_t time = mcapp->TrackTime() * 1.e+9; // time in ns?? To be consistent with EMCAL
   
  mCurrentHit = AddHit(superParent, detID, Point3D<float>(float(posX), float(posY), float(posZ)),
                       Vector3D<float>(float(momX), float(momY), float(momZ)),estart, time, lostenergy);
  mCurentSuperParent=superParent ;
  mCurrentTrackID = partID ;
  return true;
     
}


Hit* Detector::AddHit(Int_t trackID, Int_t detID,
                      const Point3D<float>& pos, const Vector3D<float>& mom, Double_t totE, Double_t time, Double_t eLoss)
{
  LOG(DEBUG4) << "Adding hit for track " << trackID << " with position (" << pos.X() << ", "
              << pos.Y() << ", " << pos.Z() << ") and momentum (" << mom.X() << ", " << mom.Y() << ", " << mom.Z()
              << ")  with energy " << totE << " loosing " << eLoss << std::endl;
  mHits->emplace_back(trackID, detID, pos, mom, totE, time, eLoss);
  return &(mHits->back());
}



void Detector::ConstructGeometry()
{
  //Create geometry description of PHOS depector for Geant simulations.

  using boost::algorithm::contains;
  LOG(DEBUG) << "Creating PHOS geometry\n";

  PHOS::GeometryParams * geom = PHOS::GeometryParams::GetInstance("Run2"); 

  if (! geom) {
    LOG(ERROR) << "ConstructGeometry: PHOS Geometry class has not been set up.\n";
  }

  //Configure geometry So far we have onny one: Run2
  {
      mCreateCPV=kTRUE ;
      mCreateHalfMod=kTRUE;
      mActiveModule[0]=kFALSE ;
      mActiveModule[1]=kTRUE ;
      mActiveModule[2]=kTRUE ;
      mActiveModule[3]=kTRUE ;
      mActiveModule[4]=kTRUE ;
      mActiveModule[5]=kFALSE ;
      mActiveCPV[0]=kFALSE ;
      mActiveCPV[1]=kFALSE ;
      mActiveCPV[2]=kFALSE ;
      mActiveCPV[3]=kTRUE ;
      mActiveCPV[4]=kFALSE ;
      mActiveCPV[5]=kFALSE ;
  }



  //First create necessary materials
  CreateMaterials();

  //Create a PHOS modules-containers which will be filled with the stuff later.
  //Depending on configuration we should prepare containers for normal PHOS module "PHOS"
  // PHOS module with CPV in front "PHOSC" and half-module "PHOH"
  TVirtualMC::GetMC()->Gsvolu("PHOS", "TRD1", getMediumID(ID_AIR), geom->GetPHOSParams(), 4) ;        
  if(mCreateHalfMod){
    TVirtualMC::GetMC()->Gsvolu("PHOH", "TRD1", getMediumID(ID_AIR), geom->GetPHOSParams(), 4) ;        
  }
  if(mCreateCPV){
    TVirtualMC::GetMC()->Gsvolu("PHOC", "TRD1", getMediumID(ID_AIR), geom->GetPHOSParams(), 4) ;             
  }
  
  //Fill prepared containers PHOS,PHOH,PHOC
  ConstructEMCGeometry() ;
 
  //Create CPV part
  if (mCreateCPV) 
    ConstructCPVGeometry() ;
  
  ConstructSupportGeometry() ; 
  
  // --- Position  PHOS modules in ALICE setup ---
  Int_t idrotm[5] ;
  Int_t iXYZ,iAngle;
  char im[5] ;
  for (Int_t iModule = 0; iModule < 5 ; iModule++ ) {
    if(!mActiveModule[iModule+1])
      continue ;
    Float_t angle[3][2];
    for (iXYZ=0; iXYZ<3; iXYZ++)
      for (iAngle=0; iAngle<2; iAngle++)
        angle[iXYZ][iAngle] = geom->GetModuleAngle(iModule,iXYZ, iAngle);
    Matrix(idrotm[iModule],
           angle[0][0],angle[0][1],
           angle[1][0],angle[1][1],
           angle[2][0],angle[2][1]) ;
    
    Float_t pos[3];
    for (iXYZ=0; iXYZ<3; iXYZ++)
      pos[iXYZ] = geom->GetModuleCenter(iModule,iXYZ);

    if(iModule==3){ //special 1/2 module
          TVirtualMC::GetMC()->Gspos("PHOH", iModule+1, "cave", pos[0], pos[1], pos[2],
                                     idrotm[iModule], "ONLY") ;
    }
    else{
      if(mActiveCPV[iModule+1]){ //use module with CPV
            TVirtualMC::GetMC()->Gspos("PHOC", iModule+1, "cave", pos[0], pos[1], pos[2],
                                       idrotm[iModule], "ONLY") ;
      }
      else{ //module wihtout CPV
            TVirtualMC::GetMC()->Gspos("PHOS", iModule+1, "cave", pos[0], pos[1], pos[2],
                                       idrotm[iModule], "ONLY") ;
      }
    }
  }

  gGeoManager->CheckGeometry();

  // Define sensitive volume
  TGeoVolume* vsense = gGeoManager->GetVolume("PXTL"); 
  if(vsense)
    AddSensitiveVolume(vsense);
  else
    LOG(ERROR) << "PHOS Sensitive volume PXTL not found ... No hit creation!\n";

}
//-----------------------------------------
void Detector::CreateMaterials(){

  // Definitions of materials to build PHOS and associated tracking media.

  // --- The PbWO4 crystals ---
  Float_t aX[3] = {207.19, 183.85, 16.0} ;
  Float_t zX[3] = {82.0, 74.0, 8.0} ;
  Float_t wX[3] = {1.0, 1.0, 4.0} ;
  Float_t dX = 8.28 ;

  Mixture(ID_PWO, "PbWO4", aX, zX, dX, -3, wX) ;


  // --- The polysterene scintillator (CH) ---
  Float_t aP[2] = {12.011, 1.00794} ;
  Float_t zP[2] = {6.0, 1.0} ;
  Float_t wP[2] = {1.0, 1.0} ;
  Float_t dP = 1.032 ;

  Mixture(ID_CPVSC, "Polystyrene", aP, zP, dP, -2, wP) ;

  // --- Aluminium ---
  Material(ID_AL, "Al", 26.98, 13., 2.7, 8.9, 999., 0, 0) ;
  // ---          Absorption length is ignored ^

 // --- Tyvek (CnH2n) ---
  Float_t aT[2] = {12.011, 1.00794} ;
  Float_t zT[2] = {6.0, 1.0} ;
  Float_t wT[2] = {1.0, 2.0} ;
  Float_t dT = 0.331 ;

  Mixture(ID_TYVEK, "Tyvek", aT, zT, dT, -2, wT) ;

  // --- Polystyrene foam ---
  Float_t aF[2] = {12.011, 1.00794} ;
  Float_t zF[2] = {6.0, 1.0} ;
  Float_t wF[2] = {1.0, 1.0} ;
  Float_t dF = 0.12 ;

  Mixture(ID_POLYFOAM, "Foam", aF, zF, dF, -2, wF) ;

 // --- Titanium ---
  Float_t aTIT[3] = {47.88, 26.98, 54.94} ;
  Float_t zTIT[3] = {22.0, 13.0, 25.0} ;
  Float_t wTIT[3] = {69.0, 6.0, 1.0} ;
  Float_t dTIT = 4.5 ;

  Mixture(ID_TITAN, "Titanium", aTIT, zTIT, dTIT, -3, wTIT);

 // --- Silicon ---
  Material(ID_APD, "Si", 28.0855, 14., 2.33, 9.36, 42.3, 0, 0) ;

  // --- Foam thermo insulation ---
  Float_t aTI[2] = {12.011, 1.00794} ;
  Float_t zTI[2] = {6.0, 1.0} ;
  Float_t wTI[2] = {1.0, 1.0} ;
  Float_t dTI = 0.04 ;

  Mixture(ID_THERMOINS, "Thermo Insul.", aTI, zTI, dTI, -2, wTI) ;

  // --- Textolith ---
  Float_t aTX[4] = {16.0, 28.09, 12.011, 1.00794} ;
  Float_t zTX[4] = {8.0, 14.0, 6.0, 1.0} ;
  Float_t wTX[4] = {292.0, 68.0, 462.0, 736.0} ;
  Float_t dTX    = 1.75 ;

  Mixture(ID_TEXTOLIT, "Textolit", aTX, zTX, dTX, -4, wTX) ;

  //--- FR4  ---
  Float_t aFR[4] = {16.0, 28.09, 12.011, 1.00794} ;
  Float_t zFR[4] = {8.0, 14.0, 6.0, 1.0} ;
  Float_t wFR[4] = {292.0, 68.0, 462.0, 736.0} ;
  Float_t dFR = 1.8 ; 

  Mixture(9, "FR4", aFR, zFR, dFR, -4, wFR) ;

  // --- Copper ---                                                                    
  Material(ID_CUPPER, "Cu", 63.546, 29, 8.96, 1.43, 14.8, 0, 0) ;
 
  // --- G10 : Printed Circuit Materiall ---                                                  
  Float_t aG10[4] = { 12., 1., 16., 28.} ;
  Float_t zG10[4] = { 6., 1., 8., 14.} ;
  Float_t wG10[4] = { .259, .288, .248, .205} ;
  Float_t dG10  = 1.7 ; 

  
  Mixture(ID_PRINTCIRC, "G10", aG10, zG10, dG10, -4, wG10);

  // --- Lead ---                                                                     
  Material(ID_PB, "Pb", 207.2, 82, 11.35, 0.56, 0., 0, 0) ;

 // --- The gas mixture ---                                                                
 // Co2
  Float_t aCO[2] = {12.0, 16.0} ; 
  Float_t zCO[2] = {6.0, 8.0} ; 
  Float_t wCO[2] = {1.0, 2.0} ; 
  Float_t dCO = 0.001977 ; 

  Mixture(ID_CO2, "CO2", aCO, zCO, dCO, -2, wCO);

  // --- Stainless steel (let it be pure iron) ---
  Material(ID_FE, "Steel", 55.845, 26, 7.87, 1.76, 0., 0, 0) ;

  // --- Fiberglass ---
  Float_t aFG[4] = {16.0, 28.09, 12.011, 1.00794} ;
  Float_t zFG[4] = {8.0, 14.0, 6.0, 1.0} ;
  Float_t wFG[4] = {292.0, 68.0, 462.0, 736.0} ;
  Float_t dFG    = 1.9 ;

  Mixture(ID_FIBERGLASS, "Fiberglas", aFG, zFG, dFG, -4, wFG) ;

  // --- Cables in Air box  ---
  // SERVICES

  Float_t aCA[4] = { 1.,12.,55.8,63.5 };
  Float_t zCA[4] = { 1.,6.,26.,29. }; 
  Float_t wCA[4] = { .014,.086,.42,.48 };
  Float_t dCA    = 0.8 ;  //this density is raw estimation, if you know better - correct

  Mixture(ID_CABLES, "Cables", aCA, zCA, dCA, -4, wCA) ;


  // --- Air ---
  Float_t aAir[4]={12.0107,14.0067,15.9994,39.948};
  Float_t zAir[4]={6.,7.,8.,18.};
  Float_t wAir[4]={0.000124,0.755267,0.231781,0.012827};
  Float_t dAir = 1.20479E-3;
 
  Mixture(ID_AIR, "Air", aAir, zAir, dAir, 4, wAir) ;

  // DEFINITION OF THE TRACKING MEDIA

  // for PHOS: idtmed[699->798] equivalent to fIdtmed[0->100]
//  Int_t   isxfld = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Integ() ;
//  Float_t sxmgmx = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Max() ;
  Int_t isxfld = 2;
  Float_t sxmgmx = 10.0;
  o2::Base::Detector::initFieldTrackingParams(isxfld, sxmgmx);
   
  // void Medium(Int_t numed, const char *name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,

  // The scintillator of the calorimeter made of PBW04                              -> idtmed[699]
  Medium(ID_PWO, "PHOS Crystal", ID_PWO, 1,
      isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // The scintillator of the CPV made of Polystyrene scintillator                   -> idtmed[700]
  Medium(ID_CPVSC, "CPV scint.", ID_CPVSC, 1,
      isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // Various Aluminium parts made of Al                                             -> idtmed[701]
  Medium(ID_AL, "Al parts", ID_AL, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, 0, 0) ;

  // The Tywek which wraps the calorimeter crystals                                 -> idtmed[702]
  Medium(ID_TYVEK, "Tyvek wrapper", ID_TYVEK, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, 0, 0) ;

  // The Polystyrene foam around the calorimeter module                             -> idtmed[703]
  Medium(ID_POLYFOAM, "Polyst. foam", ID_POLYFOAM, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // The Titanium around the calorimeter crystal                                    -> idtmed[704]
  Medium(ID_TITAN, "Titan. cover", ID_TITAN, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.0001, 0.0001, 0, 0) ;

  // The Silicon of the APD diode to read out the calorimeter crystal               -> idtmed[705] 
  Medium(ID_APD, "Si APD", ID_APD, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.01, 0.01, 0, 0) ;

  // The thermo insulating material of the box which contains the calorimeter module -> getMediumID(ID_THERMOINS)
  Medium(ID_THERMOINS, "Thermo Insul.", ID_THERMOINS, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // The Textolit which makes up the box which contains the calorimeter module      -> idtmed[707]
  Medium(ID_TEXTOLIT, "Textolit", ID_TEXTOLIT, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

 // // FR4: The Plastic which makes up the frame of micromegas                        -> idtmed[708]
 // Medium(9, "FR4 $", 9, 0,
 //      isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, 0, 0) ; 


//  // The Composite Material for  micromegas                                         -> idtmed[709]
//  Medium(10, "CompoMat   $", 10, 0,
//       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // Copper                                                                         -> idtmed[710]
  Medium(ID_CUPPER, "Copper", ID_CUPPER, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, 0, 0) ;

  // G10: Printed Circuit material                                                  -> idtmed[711]
  Medium(ID_PRINTCIRC, "G10", ID_PRINTCIRC, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.01, 0, 0) ;

  // The Lead                                                                       -> idtmed[712]
  Medium(ID_PB, "Lead", ID_PB, 1,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // The gas mixture: ArCo2                                                         -> idtmed[715]
  Medium(ID_CO2, "ArCo2", ID_CO2, 1,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.01, 0, 0) ;
 
  // Stainless steel                                                                -> idtmed[716]
  Medium(ID_FE, "Steel", ID_FE, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, 0, 0) ;

  // Fibergalss                                                                     -> getMediumID(ID_FIBERGLASS)
  Medium(ID_FIBERGLASS, "Fiberglass", ID_FIBERGLASS, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // Cables in air                                                                  -> idtmed[718]
  Medium(ID_CABLES, "Cables", ID_CABLES, 0,
       isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;

  // Air                                                                            -> idtmed[798] 
  Medium(ID_AIR, "Air", ID_AIR, 0,
       isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 10.0, 0, 0) ;


}
//-----------------------------------------
void Detector::ConstructEMCGeometry(){
  // Create the PHOS-EMC geometry for GEANT
  // Author: Dmitri Peressounko August 2001
  // Adopted for O2 project 2017
  // The used coordinate system: 
  //   1. in Module: X along longer side, Y out of beam, Z along shorter side (along beam)
  //   2. In Strip the same: X along longer side, Y out of beam, Z along shorter side (along beam)


  PHOS::GeometryParams * geom = PHOS::GeometryParams::GetInstance();

 
  Float_t par[4]={0};
  Int_t  ipar;
  
  // ======= Define the strip ===============
  for (ipar=0; ipar<3; ipar++)
    par[ipar] = *(geom->GetStripHalfSize() + ipar);
  // --- define steel volume (cell of the strip unit)
  TVirtualMC::GetMC()->Gsvolu("PSTR", "BOX ", getMediumID(ID_FE), par, 3) ;  //Made of steel

  // --- define air cell in the steel strip 
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetAirCellHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PCEL", "BOX ", getMediumID(ID_AIR), par, 3);

  // --- define wrapped crystal and put it into steel cell
  for (ipar=0; ipar<3; ipar++)
     par[ipar] = *(geom->GetWrappedHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PWRA", "BOX ", getMediumID(ID_TYVEK), par, 3);
  const Float_t * pin    = geom->GetAPDHalfSize() ; 
  const Float_t * preamp = geom->GetPreampHalfSize() ;
  Float_t y = (geom->GetAirGapLed()-2*pin[1]-2*preamp[1])/2;
  TVirtualMC::GetMC()->Gspos("PWRA", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY") ;
    
  // --- Define crystal and put it into wrapped crystall ---
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetCrystalHalfSize() + ipar);
//  TVirtualMC::GetMC()->Gsvolu("PXTL", "BOX ", getMediumID(ID_PWO), par, 3) ;
  TVirtualMC::GetMC()->Gsvolu("PXTL", "BOX ", getMediumID(ID_PB), par, 3) ;
  TVirtualMC::GetMC()->Gspos("PXTL", 1, "PWRA", 0.0, 0.0, 0.0, 0, "ONLY") ;
  
  // --- define APD/PIN preamp and put it into AirCell
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetAPDHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PPIN", "BOX ", getMediumID(ID_APD), par, 3) ;
  const Float_t * crystal = geom->GetCrystalHalfSize() ;
  y = crystal[1] + geom->GetAirGapLed() /2 - preamp[1]; 
  TVirtualMC::GetMC()->Gspos("PPIN", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY") ;
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetPreampHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PREA", "BOX ", getMediumID(ID_PRINTCIRC), par, 3) ;   // Here I assumed preamp as a printed Circuit
  y = crystal[1] + geom->GetAirGapLed() /2 + pin[1]  ;    // May it should be changed
  TVirtualMC::GetMC()->Gspos("PREA", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY") ; // to ceramics?
  
  
  // --- Fill strip with wrapped cristals in cells
  
  const Float_t* splate = geom->GetSupportPlateHalfSize();  
  y = -splate[1] ;
  const Float_t* acel = geom->GetAirCellHalfSize() ;
  
  for(Int_t lev = 2, icel = 1; 
      icel <= geom->GetNCellsXInStrip()*geom->GetNCellsZInStrip(); 
      icel += 2, lev += 2) {
    Float_t x = (2*(lev / 2) - 1 - geom->GetNCellsXInStrip())* acel[0] ;
    Float_t z = acel[2];
   
    TVirtualMC::GetMC()->Gspos("PCEL", icel, "PSTR", x, y, +z, 0, "ONLY") ;
    TVirtualMC::GetMC()->Gspos("PCEL", icel + 1, "PSTR", x, y, -z, 0, "ONLY") ;
  }

  // --- define the support plate, hole in it and position it in strip ----
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetSupportPlateHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PSUP", "BOX ", getMediumID(ID_AL), par, 3) ;
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetSupportPlateInHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PSHO", "BOX ", getMediumID(ID_AIR), par, 3) ;
  Float_t z = geom->GetSupportPlateThickness()/2 ;
  TVirtualMC::GetMC()->Gspos("PSHO", 1, "PSUP", 0.0, 0.0, z, 0, "ONLY") ;

  y = acel[1] ;
  TVirtualMC::GetMC()->Gspos("PSUP", 1, "PSTR", 0.0, y, 0.0, 0, "ONLY") ;

  
  // ========== Fill module with strips and put them into inner thermoinsulation=============
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetInnerThermoHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PTII", "BOX ", getMediumID(ID_THERMOINS), par, 3) ;     
  
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PTIH", "BOX ", getMediumID(ID_THERMOINS), par, 3) ;     
    
  
  const Float_t * inthermo = geom->GetInnerThermoHalfSize() ;
  const Float_t * strip    = geom->GetStripHalfSize() ;
  y = inthermo[1] - strip[1] ;
  Int_t irow;
  Int_t nr = 1 ;
  Int_t icol ;
  
  for(irow = 0; irow < geom->GetNStripX(); irow ++){
    Float_t x = (2*irow + 1 - geom->GetNStripX())* strip[0] ;
    for(icol = 0; icol < geom->GetNStripZ(); icol ++){
      z = (2*icol + 1 - geom->GetNStripZ()) * strip[2] ;
      TVirtualMC::GetMC()->Gspos("PSTR", nr, "PTII", x, y, z, 0, "ONLY") ;
      nr++ ;
    }
  }
  if(mCreateHalfMod){
    nr = 1 ;
    for(irow = 0; irow < geom->GetNStripX(); irow ++){
      Float_t x = (2*irow + 1 - geom->GetNStripX())* strip[0] ;
      for(icol = 0; icol < geom->GetNStripZ(); icol ++){
        z = (2*icol + 1 - geom->GetNStripZ()) * strip[2] ;
  if(irow>=geom->GetNStripX()/2)
          TVirtualMC::GetMC()->Gspos("PSTR", nr, "PTIH", x, y, z, 0, "ONLY") ;
        nr++ ;
      }
    }
  }
  
  // ------- define the air gap between thermoinsulation and cooler
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetAirGapHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PAGA", "BOX ", getMediumID(ID_AIR), par, 3) ;   
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PAGH", "BOX ", getMediumID(ID_AIR), par, 3) ;   
  const Float_t * agap = geom->GetAirGapHalfSize() ;
  y = agap[1] - inthermo[1]  ;
  
  TVirtualMC::GetMC()->Gspos("PTII", 1, "PAGA", 0.0, y, 0.0, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PTIH", 1, "PAGH", 0.0, y, 0.0, 0, "ONLY") ;


  // ------- define the Al passive cooler 
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetCoolerHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PCOR", "BOX ", getMediumID(ID_AL), par, 3) ;   
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PCOH", "BOX ", getMediumID(ID_AL), par, 3) ;   

  const Float_t * cooler = geom->GetCoolerHalfSize() ;
  y = cooler[1] - agap[1]  ;
  
  TVirtualMC::GetMC()->Gspos("PAGA", 1, "PCOR", 0.0, y, 0.0, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PAGH", 1, "PCOH", 0.0, y, 0.0, 0, "ONLY") ;
  
  // ------- define the outer thermoinsulating cover
  for (ipar=0; ipar<4; ipar++) par[ipar] = *(geom->GetOuterThermoParams() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PTIO", "TRD1", getMediumID(ID_THERMOINS), par, 4) ;        
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PIOH", "TRD1", getMediumID(ID_THERMOINS), par, 4) ;        
  const Float_t * outparams = geom->GetOuterThermoParams() ; 
  
  Int_t idrotm=-1 ;
  Matrix(idrotm, 90.0, 0.0, 0.0, 0.0, 90.0, 270.0) ;
  // Frame in outer thermoinsulation and so on: z out of beam, y along beam, x across beam
  
  z = outparams[3] - cooler[1] ;
  TVirtualMC::GetMC()->Gspos("PCOR", 1, "PTIO", 0., 0.0, z, idrotm, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PCOH", 1, "PIOH", 0., 0.0, z, idrotm, "ONLY") ;
  
  // -------- Define the outer Aluminium cover -----
  for (ipar=0; ipar<4; ipar++) par[ipar] = *(geom->GetAlCoverParams() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PCOL", "TRD1", getMediumID(ID_AL), par, 4) ;        
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PCLH", "TRD1", getMediumID(ID_AL), par, 4) ;        

  const Float_t * covparams = geom->GetAlCoverParams() ; 
  z = covparams[3] - outparams[3] ;  
  TVirtualMC::GetMC()->Gspos("PTIO", 1, "PCOL", 0., 0.0, z, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PIOH", 1, "PCLH", 0., 0.0, z, 0, "ONLY") ;

  // --------- Define front fiberglass cover -----------
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFiberGlassHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFGC", "BOX ", getMediumID(ID_FIBERGLASS), par, 3) ;  
  z = - outparams[3] ;
  TVirtualMC::GetMC()->Gspos("PFGC", 1, "PCOL", 0., 0.0, z, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PFGC", 1, "PCLH", 0., 0.0, z, 0, "ONLY") ;

  //=============This is all with cold section==============
  

  //------ Warm Section --------------
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetWarmAlCoverHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PWAR", "BOX ", getMediumID(ID_AL), par, 3) ; 
  const Float_t * warmcov = geom->GetWarmAlCoverHalfSize() ;
  
  // --- Define the outer thermoinsulation ---
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetWarmThermoHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PWTI", "BOX ", getMediumID(ID_THERMOINS), par, 3) ; 
  const Float_t * warmthermo = geom->GetWarmThermoHalfSize() ;
  z = -warmcov[2] + warmthermo[2] ;
  
  TVirtualMC::GetMC()->Gspos("PWTI", 1, "PWAR", 0., 0.0, z, 0, "ONLY") ;     
  
  // --- Define cables area and put in it T-supports ---- 
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetTCables1HalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PCA1", "BOX ", getMediumID(ID_CABLES), par, 3) ; 
  const Float_t * cbox = geom->GetTCables1HalfSize() ;
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetTSupport1HalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PBE1", "BOX ", getMediumID(ID_AL), par, 3) ;
  const Float_t * beams = geom->GetTSupport1HalfSize() ;
  Int_t isup ;
  for(isup = 0; isup < geom->GetNTSuppots(); isup++){
    Float_t x = -cbox[0] + beams[0] + (2*beams[0]+geom->GetTSupportDist())*isup ;
    TVirtualMC::GetMC()->Gspos("PBE1", isup, "PCA1", x, 0.0, 0.0, 0, "ONLY") ;
  }
  
  z = -warmthermo[2] + cbox[2];
  TVirtualMC::GetMC()->Gspos("PCA1", 1, "PWTI", 0.0, 0.0, z, 0, "ONLY") ;     
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetTCables2HalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PCA2", "BOX ", getMediumID(ID_CABLES), par, 3) ; 
  const Float_t * cbox2 = geom->GetTCables2HalfSize() ;
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetTSupport2HalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PBE2", "BOX ", getMediumID(ID_AL), par, 3) ;
  for(isup = 0; isup < geom->GetNTSuppots(); isup++){
    Float_t x = -cbox[0] + beams[0] + (2*beams[0]+geom->GetTSupportDist())*isup ;
    TVirtualMC::GetMC()->Gspos("PBE2", isup, "PCA2", x, 0.0, 0.0, 0, "ONLY") ;
  }
  
  z = -warmthermo[2] + 2*cbox[2] + cbox2[2];
  TVirtualMC::GetMC()->Gspos("PCA2", 1, "PWTI", 0.0, 0.0, z, 0, "ONLY") ;     
  
  // --- Define frame ---
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFrameXHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFRX", "BOX ", getMediumID(ID_FE), par, 3) ; 
  const Float_t * posit1 = geom->GetFrameXPosition() ;
  TVirtualMC::GetMC()->Gspos("PFRX", 1, "PWTI", posit1[0],  posit1[1], posit1[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFRX", 2, "PWTI", posit1[0], -posit1[1], posit1[2], 0, "ONLY") ;
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFrameZHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFRZ", "BOX ", getMediumID(ID_FE), par, 3) ; 
  const Float_t * posit2 = geom->GetFrameZPosition() ;
  TVirtualMC::GetMC()->Gspos("PFRZ", 1, "PWTI",  posit2[0], posit2[1], posit2[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFRZ", 2, "PWTI", -posit2[0], posit2[1], posit2[2], 0, "ONLY") ;

 // --- Define Fiber Glass support ---
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFGupXHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFG1", "BOX ", getMediumID(ID_FIBERGLASS), par, 3) ; 
  const Float_t * posit3 = geom->GetFGupXPosition() ;
  TVirtualMC::GetMC()->Gspos("PFG1", 1, "PWTI", posit3[0],  posit3[1], posit3[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFG1", 2, "PWTI", posit3[0], -posit3[1], posit3[2], 0, "ONLY") ;
  
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFGupZHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFG2", "BOX ", getMediumID(ID_FIBERGLASS), par, 3) ; 
  const Float_t * posit4 = geom->GetFGupZPosition();
  TVirtualMC::GetMC()->Gspos("PFG2", 1, "PWTI",  posit4[0], posit4[1], posit4[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFG2", 2, "PWTI", -posit4[0], posit4[1], posit4[2], 0, "ONLY") ;
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFGlowXHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFG3", "BOX ", getMediumID(ID_FIBERGLASS), par, 3) ; 
  const Float_t * posit5 = geom->GetFGlowXPosition() ;
  TVirtualMC::GetMC()->Gspos("PFG3", 1, "PWTI", posit5[0],  posit5[1], posit5[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFG3", 2, "PWTI", posit5[0], -posit5[1], posit5[2], 0, "ONLY") ;
    
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFGlowZHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PFG4", "BOX ", getMediumID(ID_FIBERGLASS), par, 3) ; 
  const Float_t * posit6 = geom->GetFGlowZPosition() ;
  TVirtualMC::GetMC()->Gspos("PFG4", 1, "PWTI",  posit6[0], posit6[1], posit6[2], 0, "ONLY") ;
  TVirtualMC::GetMC()->Gspos("PFG4", 2, "PWTI", -posit6[0], posit6[1], posit6[2], 0, "ONLY") ;

  // --- Define Air Gap for FEE electronics -----   
  for (ipar=0; ipar<3; ipar++) par[ipar] = *(geom->GetFEEAirHalfSize() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PAFE", "BOX ", getMediumID(ID_AIR), par, 3) ; 
  const Float_t * posit7 = geom->GetFEEAirPosition() ;
  TVirtualMC::GetMC()->Gspos("PAFE", 1, "PWTI",  posit7[0], posit7[1], posit7[2], 0, "ONLY") ;
  
  // Define the EMC module volume and combine Cool and Warm sections  
  for (ipar=0; ipar<4; ipar++) par[ipar] = *(geom->GetEMCParams() + ipar);
  TVirtualMC::GetMC()->Gsvolu("PEMC", "TRD1", getMediumID(ID_AIR), par, 4) ;        
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gsvolu("PEMH", "TRD1", getMediumID(ID_AIR), par, 4) ;        
  z =  - warmcov[2] ;
  TVirtualMC::GetMC()->Gspos("PCOL", 1, "PEMC",  0., 0., z, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PCLH", 1, "PEMH",  0., 0., z, 0, "ONLY") ;
  z = covparams[3] ;
  TVirtualMC::GetMC()->Gspos("PWAR", 1, "PEMC",  0., 0., z, 0, "ONLY") ;
  if(mCreateHalfMod)
    TVirtualMC::GetMC()->Gspos("PWAR", 1, "PEMH",  0., 0., z, 0, "ONLY") ;
  
  
  // Put created EMC geometry into PHOS volume
  
  z = geom->GetCPVBoxSize(1) / 2. ;
  //normal PHOS module
  TVirtualMC::GetMC()->Gspos("PEMC", 1, "PHOS", 0., 0., z, 0, "ONLY") ; 
  if(mCreateCPV) //Module with CPV 
    TVirtualMC::GetMC()->Gspos("PEMC", 1, "PHOC", 0., 0., z, 0, "ONLY") ;   
  if(mCreateHalfMod) //half of PHOS module
    TVirtualMC::GetMC()->Gspos("PEMH", 1, "PHOH", 0., 0., z, 0, "ONLY") ; 

}
//-----------------------------------------
void Detector::ConstructCPVGeometry(){

  // Create the PHOS-CPV geometry for GEANT
  // Author: Yuri Kharlov 11 September 2000
  // Adopted for O2 project 2017
  
  PHOS::GeometryParams * geom = PHOS::GeometryParams::GetInstance();

  Float_t par[3]={0}, x=0.,y=0.,z=0.;


  // The box containing all CPV for one PHOS module filled with air 
  par[0] = geom->GetCPVBoxSize(0) / 2.0 ;  
  par[1] = geom->GetCPVBoxSize(1) / 2.0 ; 
  par[2] = geom->GetCPVBoxSize(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PCPV", "BOX ", getMediumID(ID_AIR), par, 3) ;

  const Float_t * emcParams = geom->GetEMCParams() ;
  z = - emcParams[3] ;
  Int_t rotm ;
  Matrix(rotm, 90.,0., 0., 0., 90., 90.) ;

  TVirtualMC::GetMC()->Gspos("PCPV", 1, "PHOC", 0.0, 0.0, z, rotm, "ONLY") ; 
  
  // Gassiplex board
  
  par[0] = geom->GetGassiplexChipSize(0)/2.;
  par[1] = geom->GetGassiplexChipSize(1)/2.;
  par[2] = geom->GetGassiplexChipSize(2)/2.;
  TVirtualMC::GetMC()->Gsvolu("PCPC","BOX ",getMediumID(ID_TEXTOLIT),par,3);
  
  // Cu+Ni foil covers Gassiplex board

  par[1] = geom->GetCPVCuNiFoilThickness()/2;
  TVirtualMC::GetMC()->Gsvolu("PCPD","BOX ", getMediumID(ID_CUPPER),par,3);
  y      = -(geom->GetGassiplexChipSize(1)/2 - par[1]);
  TVirtualMC::GetMC()->Gspos("PCPD",1,"PCPC",0,y,0,0,"ONLY");

  // Position of the chip inside CPV

  Float_t xStep = geom->GetCPVActiveSize(0) / (geom->GetNumberOfCPVChipsPhi() + 1);
  Float_t zStep = geom->GetCPVActiveSize(1) / (geom->GetNumberOfCPVChipsZ()   + 1);
  Int_t   copy  = 0;
  y = geom->GetCPVFrameSize(1)/2           - geom->GetFTPosition(0) +
    geom->GetCPVTextoliteThickness() / 2 + geom->GetGassiplexChipSize(1) / 2 + 0.1;
  for (Int_t ix=0; ix<geom->GetNumberOfCPVChipsPhi(); ix++) {
    x = xStep * (ix+1) - geom->GetCPVActiveSize(0)/2;
    for (Int_t iz=0; iz<geom->GetNumberOfCPVChipsZ(); iz++) {
      copy++;
      z = zStep * (iz+1) - geom->GetCPVActiveSize(1)/2;
      TVirtualMC::GetMC()->Gspos("PCPC",copy,"PCPV",x,y,z,0,"ONLY");
    }
  }

  // Foiled textolite (1 mm of textolite + 50 mkm of Cu + 6 mkm of Ni)
  
  par[0] = geom->GetCPVActiveSize(0)        / 2;
  par[1] = geom->GetCPVTextoliteThickness() / 2;
  par[2] = geom->GetCPVActiveSize(1)        / 2;
  TVirtualMC::GetMC()->Gsvolu("PCPF","BOX ",getMediumID(ID_TEXTOLIT),par,3);

  // Argon gas volume

  par[1] = (geom->GetFTPosition(2) - geom->GetFTPosition(1) - geom->GetCPVTextoliteThickness()) / 2;
  TVirtualMC::GetMC()->Gsvolu("PCPG","BOX ", getMediumID(ID_CO2),par,3);

  for (Int_t i=0; i<4; i++) {
    y = geom->GetCPVFrameSize(1) / 2 - geom->GetFTPosition(i) + geom->GetCPVTextoliteThickness()/2;
    TVirtualMC::GetMC()->Gspos("PCPF",i+1,"PCPV",0,y,0,0,"ONLY");
    if(i==1){
      y-= (geom->GetFTPosition(2) - geom->GetFTPosition(1)) / 2;
      TVirtualMC::GetMC()->Gspos("PCPG",1,"PCPV ",0,y,0,0,"ONLY");
    }
  }

  // Dummy sensitive plane in the middle of argone gas volume

  par[1]=0.001;
  TVirtualMC::GetMC()->Gsvolu("PCPQ","BOX ", getMediumID(ID_CO2),par,3);
  TVirtualMC::GetMC()->Gspos ("PCPQ",1,"PCPG",0,0,0,0,"ONLY");

  // Cu+Ni foil covers textolite

  par[1] = geom->GetCPVCuNiFoilThickness() / 2;
  TVirtualMC::GetMC()->Gsvolu("PCP1","BOX ", getMediumID(ID_CUPPER),par,3);
  y = geom->GetCPVTextoliteThickness()/2 - par[1];
  TVirtualMC::GetMC()->Gspos ("PCP1",1,"PCPF",0,y,0,0,"ONLY");

  // Aluminum frame around CPV

  par[0] = geom->GetCPVFrameSize(0)/2;
  par[1] = geom->GetCPVFrameSize(1)/2;
  par[2] = geom->GetCPVBoxSize(2)  /2;
  TVirtualMC::GetMC()->Gsvolu("PCF1","BOX ", getMediumID(ID_AL),par,3);

  par[0] = geom->GetCPVBoxSize(0)/2 - geom->GetCPVFrameSize(0);
  par[1] = geom->GetCPVFrameSize(1)/2;
  par[2] = geom->GetCPVFrameSize(2)/2;
  TVirtualMC::GetMC()->Gsvolu("PCF2","BOX ",getMediumID(ID_AL),par,3);

  for (Int_t j=0; j<=1; j++) {
    x = TMath::Sign(1,2*j-1) * (geom->GetCPVBoxSize(0) - geom->GetCPVFrameSize(0)) / 2;
    TVirtualMC::GetMC()->Gspos("PCF1",j+1,"PCPV", x,0,0,0,"ONLY");
    z = TMath::Sign(1,2*j-1) * (geom->GetCPVBoxSize(2) - geom->GetCPVFrameSize(2)) / 2;
    TVirtualMC::GetMC()->Gspos("PCF2",j+1,"PCPV",0, 0,z,0,"ONLY");
  }
}

//-----------------------------------------
void Detector::ConstructSupportGeometry(){

 // Create the PHOS support geometry for GEANT
  PHOS::GeometryParams * geom = PHOS::GeometryParams::GetInstance();

  
  Float_t par[5]={0}, x0=0.,y0=0.,z0=0. ; 
  Int_t   i,j,copy;


  // --- Dummy box containing two rails on which PHOS support moves
  // --- Put these rails to the bottom of the L3 magnet

  par[0] =  geom->GetRailRoadSize(0) / 2.0 ;
  par[1] =  geom->GetRailRoadSize(1) / 2.0 ;
  par[2] =  geom->GetRailRoadSize(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PRRD", "BOX ", getMediumID(ID_AIR), par, 3) ;

  y0     = -(geom->GetRailsDistanceFromIP() - geom->GetRailRoadSize(1) / 2.0) ;
  TVirtualMC::GetMC()->Gspos("PRRD", 1, "cave", 0.0, y0, 0.0, 0, "ONLY") ; 

  // --- Dummy box containing one rail

  par[0] =  geom->GetRailOuterSize(0) / 2.0 ;
  par[1] =  geom->GetRailOuterSize(1) / 2.0 ;
  par[2] =  geom->GetRailOuterSize(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PRAI", "BOX ", getMediumID(ID_AIR), par, 3) ;

  for (i=0; i<2; i++) {
    x0     = (2*i-1) * geom->GetDistanceBetwRails()  / 2.0 ;
    TVirtualMC::GetMC()->Gspos("PRAI", i, "PRRD", x0, 0.0, 0.0, 0, "ONLY") ; 
  }

  // --- Upper and bottom steel parts of the rail

  par[0] =  geom->GetRailPart1(0) / 2.0 ;
  par[1] =  geom->GetRailPart1(1) / 2.0 ;
  par[2] =  geom->GetRailPart1(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PRP1", "BOX ", getMediumID(ID_FE), par, 3) ;

  y0     = - (geom->GetRailOuterSize(1) - geom->GetRailPart1(1))  / 2.0 ;
  TVirtualMC::GetMC()->Gspos("PRP1", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY") ;
  y0     =   (geom->GetRailOuterSize(1) - geom->GetRailPart1(1))  / 2.0 - geom->GetRailPart3(1);
  TVirtualMC::GetMC()->Gspos("PRP1", 2, "PRAI", 0.0, y0, 0.0, 0, "ONLY") ;

  // --- The middle vertical steel parts of the rail

  par[0] =  geom->GetRailPart2(0) / 2.0 ;
  par[1] =  geom->GetRailPart2(1) / 2.0 ;
  par[2] =  geom->GetRailPart2(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PRP2", "BOX ", getMediumID(ID_FE), par, 3) ;

  y0     =   - geom->GetRailPart3(1) / 2.0 ;
  TVirtualMC::GetMC()->Gspos("PRP2", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY") ; 

  // --- The most upper steel parts of the rail

  par[0] =  geom->GetRailPart3(0) / 2.0 ;
  par[1] =  geom->GetRailPart3(1) / 2.0 ;
  par[2] =  geom->GetRailPart3(2) / 2.0 ;
  TVirtualMC::GetMC()->Gsvolu("PRP3", "BOX ", getMediumID(ID_FE), par, 3) ;

  y0     =   (geom->GetRailOuterSize(1) - geom->GetRailPart3(1))  / 2.0 ;
  TVirtualMC::GetMC()->Gspos("PRP3", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY") ; 

  // --- The wall of the cradle
  // --- The wall is empty: steel thin walls and air inside

  par[1] =  TMath::Sqrt(TMath::Power((geom->GetIPtoCPVDistance() + geom->GetOuterBoxSize(3)),2) +
      TMath::Power((geom->GetOuterBoxSize(1)/2),2))+10. ;
  par[0] =  par[1] - geom->GetCradleWall(1) ;
  par[2] =  geom->GetCradleWall(2) / 2.0 ;
  par[3] =  geom->GetCradleWall(3) ;
  par[4] =  geom->GetCradleWall(4) ;
  TVirtualMC::GetMC()->Gsvolu("PCRA", "TUBS", getMediumID(ID_FE), par, 5) ;

  par[0] +=  geom->GetCradleWallThickness() ;
  par[1] -=  geom->GetCradleWallThickness() ;
  par[2] -=  geom->GetCradleWallThickness() ;
  TVirtualMC::GetMC()->Gsvolu("PCRE", "TUBS", getMediumID(ID_FE), par, 5) ;
  TVirtualMC::GetMC()->Gspos ("PCRE", 1, "PCRA", 0.0, 0.0, 0.0, 0, "ONLY") ; 

  for (i=0; i<2; i++) {
    z0 = (2*i-1) * (geom->GetOuterBoxSize(2) + geom->GetCradleWall(2) )/ 2.0  ;
        TVirtualMC::GetMC()->Gspos("PCRA", i, "cave", 0.0, 0.0, z0, 0, "ONLY") ; 
  }

  // --- The "wheels" of the cradle
  
  par[0] = geom->GetCradleWheel(0) / 2;
  par[1] = geom->GetCradleWheel(1) / 2;
  par[2] = geom->GetCradleWheel(2) / 2;
  TVirtualMC::GetMC()->Gsvolu("PWHE", "BOX ", getMediumID(ID_FE), par, 3) ;

  y0 = -(geom->GetRailsDistanceFromIP() - geom->GetRailRoadSize(1) -
   geom->GetCradleWheel(1)/2) ;
  for (i=0; i<2; i++) {
    z0 = (2*i-1) * ((geom->GetOuterBoxSize(2) + geom->GetCradleWheel(2))/ 2.0 +
                    geom->GetCradleWall(2));
    for (j=0; j<2; j++) {
      copy = 2*i + j;
      x0 = (2*j-1) * geom->GetDistanceBetwRails()  / 2.0 ;
      TVirtualMC::GetMC()->Gspos("PWHE", copy, "cave", x0, y0, z0, 0, "ONLY") ; 
    }
  }

}



