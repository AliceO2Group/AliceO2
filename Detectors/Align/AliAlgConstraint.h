#ifndef ALIALGCONSTRAINT_H
#define ALIALGCONSTRAINT_H

#include <stdio.h>
#include <TNamed.h>
#include <TObjArray.h>
#include "AliAlgVol.h"

/*--------------------------------------------------------
  Descriptor of geometrical constraint: the cumulative
  corrections of children for requested DOFs in the frame of 
  parent (of LAB if parent is not defined) forced to be 0.
  The parent - child relationship need not to be real

  Constraint wil be quazi-exact (Lagrange multiplier) if 
  corresponding sigma = 0, or of gaussian type is sigma>0
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgConstraint : public TNamed
{
 public:
  enum {kNDOFGeom=AliAlgVol::kNDOFGeom};
  enum {kNoJacobianBit=BIT(14)};
  //
  AliAlgConstraint(const char* name=0,const char* title=0);
  virtual ~AliAlgConstraint();
  //
  void              SetParent(const AliAlgVol* par);
  const AliAlgVol*  GetParent()                const {return fParent;}
  //
  Int_t       GetNChildren()                   const {return fChildren.GetEntriesFast();}
  AliAlgVol*  GetChild(int i)                  const {return (AliAlgVol*)fChildren[i];}
  void        AddChild(const AliAlgVol* v)           {if (v) fChildren.AddLast((AliAlgVol*)v);}
  //
  Bool_t     IsDOFConstrained(Int_t dof)       const {return fConstraint&0x1<<dof;}
  UChar_t    GetConstraintPattern()            const {return fConstraint;}
  void       ConstrainDOF(Int_t dof)                 {fConstraint |= 0x1<<dof;}
  void       UnConstrainDOF(Int_t dof)               {fConstraint &=~(0x1<<dof);}
  void       SetConstrainPattern(UInt_t pat)         {fConstraint = pat;}
  Bool_t     HasConstraint()                   const {return  fConstraint;}
  Double_t   GetSigma(int i)                   const {return  fSigma[i];}
  void       SetSigma(int i,double s=0)              {fSigma[i] = s;}
  //
  void       SetNoJacobian(Bool_t v=kTRUE)           {SetBit(kNoJacobianBit,v);}
  Bool_t     GetNoJacobian()                   const {return TestBit(kNoJacobianBit);}
  //
  void       ConstrCoefGeom(const TGeoHMatrix &matRD, float* jac/*[kNDOFGeom][kNDOFGeom]*/) const;
  //
  virtual void   Print(const Option_t *opt="")          const;
  virtual void   WriteChildrenConstraints(FILE* conOut) const;
  virtual void   CheckConstraint()                      const;
  virtual const char* GetDOFName(int i)                 const {return AliAlgVol::GetGeomDOFName(i);}
  //
 protected:
    // ------- dummies -------
  AliAlgConstraint(const AliAlgConstraint&);
  AliAlgConstraint& operator=(const AliAlgConstraint&);
  //
 protected:
  UInt_t            fConstraint;          // bit pattern of constraint
  Double_t          fSigma[kNDOFGeom];    // optional sigma if constraint is gaussian
  const AliAlgVol*  fParent;              // parent volume for contraint, lab if 0
  TObjArray         fChildren;            // volumes subjected to constraints
  //
  ClassDef(AliAlgConstraint,2);
};

#endif
