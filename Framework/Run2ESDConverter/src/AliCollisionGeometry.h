#ifndef ALICOLLISIONGEOMETRY_H
#define ALICOLLISIONGEOMETRY_H
/* Copyright(c) 198-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */
//-------------------------------------------------------------------------
//                          Class AliCollisionGeometry
//   This is a class to handle the collison geometry defined by
//   the generator
//-------------------------------------------------------------------------

#include <Rtypes.h>

class AliCollisionGeometry
{
public:
    AliCollisionGeometry();
    AliCollisionGeometry(const AliCollisionGeometry& cgeo);
    virtual ~AliCollisionGeometry(){;}
    // Getters
    Float_t ImpactParameter() const  {return fImpactParameter;}
    Float_t ReactionPlaneAngle() const {return fReactionPlaneAngle;}
    Int_t   HardScatters() const {return fNHardScatters;}
    Int_t   ProjectileParticipants() const {return fNProjectileParticipants;}
    Int_t   TargetParticipants() const {return fNTargetParticipants;}
    Int_t   ProjSpectatorsn() const {return fProjectileSpecn;}
    Int_t   ProjSpectatorsp() const {return fProjectileSpecp;}
    Int_t   TargSpectatorsn() const {return fTargetSpecn;	 }
    Int_t   TargSpectatorsp() const {return fTargetSpecp;	 }
    Int_t   NN() const  {return fNNColl;}
    Int_t   NNw() const {return fNNwColl;}
    Int_t   NwN() const {return fNwNColl;}
    Int_t   NwNw() const {return fNwNwColl;}
    void    SetNDiffractive(Int_t sd1, Int_t sd2, Int_t sdd) {fNSD1 = sd1; fNSD2 = sd2; fNDD = sdd;}
    // Setters
    void SetImpactParameter(Float_t b)     {fImpactParameter=b;}
    void SetReactionPlaneAngle(Float_t phi)     {fReactionPlaneAngle = phi;}
    void SetHardScatters(Int_t n)  {fNHardScatters=n;}
    void SetParticipants(Int_t np, Int_t nt)
	{fNProjectileParticipants=np, fNTargetParticipants=nt;}
    void SetCollisions(Int_t nn, Int_t nnw, Int_t nwn, Int_t nwnw)
	{fNNColl=nn, fNNwColl=nnw, fNwNColl=nwn,  fNwNwColl=nwnw;}
    void SetSpectators(Int_t nprojspecn, Int_t nprojspecp, Int_t ntargspecn, Int_t ntargspecp)
	{fProjectileSpecn=nprojspecn, fProjectileSpecp=nprojspecp, 
	 fTargetSpecn=ntargspecn, fTargetSpecp=ntargspecp;}
    void    GetNDiffractive(Int_t& sd1, Int_t& sd2, Int_t& sdd) {sd1 = fNSD1; sd2 = fNSD2; sdd = fNDD;}
 protected:
    Int_t   fNHardScatters;            // Number of hard scatterings
    Int_t   fNProjectileParticipants;  // Number of projectiles participants
    Int_t   fNTargetParticipants;      // Number of target participants
    Int_t   fNNColl;                   // Number of N-N collisions
    Int_t   fNNwColl;                  // Number of N-Nwounded collisions
    Int_t   fNwNColl;                  // Number of Nwounded-N collisons
    Int_t   fNwNwColl;                 // Number of Nwounded-Nwounded collisions
    Int_t   fProjectileSpecn;	       // Num. of spectator neutrons from projectile nucleus
    Int_t   fProjectileSpecp;	       // Num. of spectator protons from projectile nucleus
    Int_t   fTargetSpecn;    	       // Num. of spectator neutrons from target nucleus
    Int_t   fTargetSpecp;    	       // Num. of spectator protons from target nucleus
    Float_t fImpactParameter;          // Impact Parameter
    Float_t fReactionPlaneAngle;       // Reaction plane angle
    Int_t   fNSD1;                     // number of SD1 in pA, AA 
    Int_t   fNSD2;                     // number of SD2 in pA, AA
    Int_t   fNDD;                      // number of DD  in pA, AA
private:
    AliCollisionGeometry& operator=(const AliCollisionGeometry& cg); //Not implemented
  ClassDef(AliCollisionGeometry,4)     // Collision Geometry
};
#endif






