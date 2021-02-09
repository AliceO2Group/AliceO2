// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgConstraint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Descriptor of geometrical constraint

#include "AliAlgConstraint.h"
#include "AliAlignObjParams.h"
#include "AliAlgAux.h"
#include "AliLog.h"
#include <TGeoMatrix.h>
#include <TMath.h>
#include <stdio.h>

ClassImp(o2::align::AliAlgConstraint);

using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

//___________________________________________________________________
AliAlgConstraint::AliAlgConstraint(const char* name, const char* title)
  : TNamed(name, title), fConstraint(0), fParent(0), fChildren(2)
{
  // def. c-tor
  for (int i = kNDOFGeom; i--;)
    fSigma[i] = 0;
}

//___________________________________________________________________
AliAlgConstraint::~AliAlgConstraint()
{
  // d-tor
  delete fParent;
}

//___________________________________________________________________
void AliAlgConstraint::SetParent(const AliAlgVol* par)
{
  fParent = par;
  TString nm = GetName();
  if (nm.IsNull()) {
    if (par)
      SetNameTitle(par->GetSymName(), "Automatic");
    else
      SetNameTitle("GLOBAL", "Automatic");
  }
}

//______________________________________________________
void AliAlgConstraint::WriteChildrenConstraints(FILE* conOut) const
{
  // write for PEDE eventual constraints on children movement in parent frame
  //
  enum { kOff,
         kOn,
         kOnOn };
  enum { kConstr,
         kMeas };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyConstr[2] = {"constraint", "measurement"};
  //
  Bool_t doJac = !GetNoJacobian(); // do we need jacobian evaluation?
  int nch = GetNChildren();
  float* cstrArr = new float[nch * kNDOFGeom * kNDOFGeom];
  memset(cstrArr, 0, nch * kNDOFGeom * kNDOFGeom * sizeof(float));
  // we need for each children the matrix for vector transformation from children frame
  // (in which its DOFs are defined, LOC or TRA) to this parent variation frame
  // matRel = mPar^-1*mChild
  TGeoHMatrix mPar;
  //
  // in case of parent assigned use its matrix,
  // otherwise Alice global frame is assumed to be the parent -> Unit matrix
  if (fParent && doJac) {
    if (fParent->IsFrameTRA())
      fParent->GetMatrixT2G(mPar); // tracking to global
    else
      mPar = fParent->GetMatrixL2GIdeal(); // local to global
    mPar = mPar.Inverse();
  }
  //
  float* jac = cstrArr;
  int nContCh[kNDOFGeom] = {0}; // we need at least on contributing children DOF to constrain the parent DOF
  for (int ich = 0; ich < nch; ich++) {
    AliAlgVol* child = GetChild(ich);
    //
    if (doJac) { // calculate jacobian
      TGeoHMatrix matRel;
      if (child->IsFrameTRA())
        child->GetMatrixT2G(matRel); // tracking to global
      else
        matRel = child->GetMatrixL2GIdeal(); // local to global
      matRel.MultiplyLeft(&mPar);
      ConstrCoefGeom(matRel, jac);
      //
      for (int ics = 0; ics < kNDOFGeom; ics++) { // DOF of parent to be constrained
        for (int ip = 0; ip < kNDOFGeom; ip++) {  // count contributing DOFs
          float jv = jac[ics * kNDOFGeom + ip];
          if (!IsZeroAbs(jv) && child->IsFreeDOF(ip) && child->GetParErr(ip) >= 0)
            nContCh[ip]++;
        }
      }
    } else { // simple constraint on the sum of requested DOF
      //
      for (int ip = 0; ip < kNDOFGeom; ip++) {
        if (child->IsFreeDOF(ip) && child->GetParErr(ip) >= 0)
          nContCh[ip]++;
        jac[ip * kNDOFGeom + ip] = 1.;
      }
    }
    jac += kNDOFGeom * kNDOFGeom; // matrix for next slot
  }
  //
  for (int ics = 0; ics < kNDOFGeom; ics++) {
    if (!IsDOFConstrained(ics))
      continue;
    int cmtStatus = nContCh[ics] > 0 ? kOff : kOn; // do we comment this constraint?
    //
    if (cmtStatus)
      AliInfoF("No contributors to constraint of %3s of %s", GetDOFName(ics), GetName());
    //
    if (fSigma[ics] > 0) {
      fprintf(conOut, "\n%s%s\t%e\t%e\t%s %s of %s %s\n", comment[cmtStatus], kKeyConstr[kMeas], 0.0, fSigma[ics],
              comment[kOnOn], GetDOFName(ics), GetName(), GetTitle());
    } else {
      fprintf(conOut, "\n%s%s\t%e\t%s %s of %s %s\n", comment[cmtStatus], kKeyConstr[kConstr], 0.0,
              comment[kOnOn], GetDOFName(ics), GetName(), GetTitle());
    }
    for (int ich = 0; ich < nch; ich++) { // contribution from this children DOFs to constraint
      AliAlgVol* child = GetChild(ich);
      jac = cstrArr + kNDOFGeom * kNDOFGeom * ich;
      if (cmtStatus)
        fprintf(conOut, "%s", comment[cmtStatus]); // comment out contribution
      // first write real constraints
      for (int ip = 0; ip < kNDOFGeom; ip++) {
        float jv = jac[ics * kNDOFGeom + ip];
        if (child->IsFreeDOF(ip) && !IsZeroAbs(jv) && child->GetParErr(ip) >= 0)
          fprintf(conOut, "%9d %+.3e\t", child->GetParLab(ip), jv);
      } // loop over DOF's of children contributing to this constraint
      // now, after comment, write disabled constraints
      fprintf(conOut, "%s ", comment[kOn]);
      if (doJac) {
        for (int ip = 0; ip < kNDOFGeom; ip++) {
          float jv = jac[ics * kNDOFGeom + ip];
          if (child->IsFreeDOF(ip) && !IsZeroAbs(jv) && child->GetParErr(ip) >= 0)
            continue;
          fprintf(conOut, "%9d %+.3e\t", child->GetParLab(ip), jv);
        } // loop over DOF's of children contributing to this constraint
      }
      fprintf(conOut, "%s from %s\n", comment[kOnOn], child->GetName());
    } // loop over children
  }   // loop over constraints in parent volume
  //
  delete[] cstrArr;
}

//______________________________________________________
void AliAlgConstraint::CheckConstraint() const
{
  // check how the constraints are satysfied
  //
  int nch = GetNChildren();
  if (!nch)
    return;
  //
  Bool_t doJac = !GetNoJacobian(); // do we need jacobian evaluation?
  float* cstrArr = new float[nch * kNDOFGeom * kNDOFGeom];
  memset(cstrArr, 0, nch * kNDOFGeom * kNDOFGeom * sizeof(float));
  // we need for each children the matrix for vector transformation from children frame
  // (in which its DOFs are defined, LOC or TRA) to this parent variation frame
  // matRel = mPar^-1*mChild
  TGeoHMatrix mPar;
  // in case of parent assigned use its matrix,
  // otherwise Alice global frame is assumed to be the parent -> Unit matrix
  if (fParent && doJac) {
    if (fParent->IsFrameTRA())
      fParent->GetMatrixT2G(mPar); // tracking to global
    else
      mPar = fParent->GetMatrixL2GIdeal(); // local to global
    mPar = mPar.Inverse();
  }
  //
  float* jac = cstrArr;
  double parsTotEx[kNDOFGeom] = {0}; // explicitly calculated total modification
  double parsTotAn[kNDOFGeom] = {0}; // analyticaly calculated total modification
  //
  printf("\n\n ----- Constraints Validation for %s %s ------\n", GetName(), GetTitle());
  printf(" chld| ");
  for (int jp = 0; jp < kNDOFGeom; jp++)
    printf("  %3s:%3s An/Ex  |", GetDOFName(jp), IsDOFConstrained(jp) ? "ON " : "OFF");
  printf(" | ");
  for (int jp = 0; jp < kNDOFGeom; jp++)
    printf("  D%3s   ", GetDOFName(jp));
  printf(" ! %s\n", GetName());
  for (int ich = 0; ich < nch; ich++) {
    AliAlgVol* child = GetChild(ich);
    double parsC[kNDOFGeom] = {0}, parsPAn[kNDOFGeom] = {0}, parsPEx[kNDOFGeom] = {0};
    for (int jc = kNDOFGeom; jc--;)
      parsC[jc] = child->GetParVal(jc); // child params in child frame
    printf("#%3d | ", ich);
    //
    if (doJac) {
      TGeoHMatrix matRel;
      if (child->IsFrameTRA())
        child->GetMatrixT2G(matRel); // tracking to global
      else
        matRel = child->GetMatrixL2GIdeal(); // local to global
      //
      matRel.MultiplyLeft(&mPar);
      ConstrCoefGeom(matRel, jac); // Jacobian for analytical constraint used by MillePeded
                                   //
      TGeoHMatrix tau;
      child->Delta2Matrix(tau, parsC); // child correction matrix in the child frame
      const TGeoHMatrix& matreli = matRel.Inverse();
      tau.Multiply(&matreli);
      tau.MultiplyLeft(&matRel); //  child correction matrix in the parent frame
      AliAlignObjParams tmpPar;
      tmpPar.SetMatrix(tau);
      tmpPar.GetTranslation(&parsPEx[0]);
      tmpPar.GetAngles(&parsPEx[3]); // explicitly calculated child params in parent frame
      //
      // analytically calculated child params in parent frame
      for (int jp = 0; jp < kNDOFGeom; jp++) {
        for (int jc = 0; jc < kNDOFGeom; jc++)
          parsPAn[jp] += jac[jp * kNDOFGeom + jc] * parsC[jc];
        parsTotAn[jp] += parsPAn[jp]; // analyticaly calculated total modification
        parsTotEx[jp] += parsPEx[jp]; // explicitly calculated total modification
        //
        printf("%+.1e/%+.1e ", parsPAn[jp], parsPEx[jp]);
        //
      }
      //
      jac += kNDOFGeom * kNDOFGeom; // matrix for next slot
    } else {
      for (int jc = 0; jc < kNDOFGeom; jc++) {
        Bool_t acc = child->IsFreeDOF(jc) && child->GetParErr(jc) >= 0;
        if (acc) {
          printf("    %+.3e    ", parsC[jc]);
          parsTotAn[jc] += parsC[jc];
        } else
          printf(" /* %+.3e */ ", parsC[jc]); // just for info, not in the constraint
      }
    }
    printf(" | ");
    for (int jc = 0; jc < kNDOFGeom; jc++)
      printf("%+.1e ", parsC[jc]); // child proper corrections
    printf(" ! %s\n", child->GetSymName());
  }
  //
  printf(" Tot | ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    if (doJac)
      printf("%+.1e/%+.1e ", parsTotAn[jp], parsTotEx[jp]);
    else {
      if (IsDOFConstrained(jp))
        printf("    %+.3e    ", parsTotAn[jp]);
      else
        printf(" /* %+.3e */ ", parsTotAn[jp]);
    }
  }
  printf(" | ");
  if (fParent)
    for (int jp = 0; jp < kNDOFGeom; jp++)
      printf("%+.1e ", fParent->GetParVal(jp)); // parent proper corrections
  else
    printf(" no parent -> %s ", doJac ? "Global" : "Simple");
  printf(" ! <----- %s\n", GetName());
  //
  printf(" Sig | ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    if (IsDOFConstrained(jp))
      printf("    %+.3e    ", fSigma[jp]);
    else
      printf(" /* %+.3e */ ", fSigma[jp]);
  }
  printf(" ! <----- \n");

  //
  delete[] cstrArr;
  //
}

//_________________________________________________________________
void AliAlgConstraint::ConstrCoefGeom(const TGeoHMatrix& matRD, float* jac /*[kNDOFGeom][kNDOFGeom]*/) const
{
  // If the transformation R brings the vector from "local" frame to "master" frame as V=R*v
  // then application of the small LOCAL correction tau to vector v is equivalent to
  // aplication of correction TAU in MASTER framce V' = R*tau*v = TAU*R*v
  // with TAU = R*tau*R^-1
  // Constraining the LOCAL modifications of child volumes to have 0 total movement in their parent
  // frame is equivalent to request that sum of all TAU matrices is unity matrix, or TAU-I = 0.
  //
  // This routine calculates derivatives of the TAU-I matrix over local corrections x,y,z, psi,tht,phi
  // defining matrix TAU. In small corrections approximation the constraint is equivalent to
  // Sum_over_child_volumes{ [dTAU/dParam]_ij * deltaParam } = 0
  // for all elements ij of derivative matrices. Since only 6 out of 16 matrix params are independent,
  // we request the constraint only for  [30](X), [31](Y), [32](Z), [12](psi), [02](tht), [01](phi)
  // Choice defined by convention of AliAlgObg::Angles2Matrix (need elements ~ linear in corrections)
  //
  TGeoHMatrix matRI = matRD.Inverse();
  const int ij[kNDOFGeom][2] = {{3, 0}, {3, 1}, {3, 2}, {1, 2}, {0, 2}, {0, 1}};
  //
  const double *rd = matRD.GetRotationMatrix(), *ri = matRI.GetRotationMatrix();
  const double /**td=matRD.GetTranslation(),*/* ti = matRI.GetTranslation();
  //
  // the angles are in degrees, while we use sinX->X approximation...
  const double cf[kNDOFGeom] = {1, 1, 1, DegToRad(), DegToRad(), DegToRad()};
  //
  // since the TAU is supposed to convert local corrections in the child frame to corrections
  // in the parent frame, we scale angular degrees of freedom back to degrees and assign the
  // sign of S in the S*sin(angle) in the matrix, so that the final correction has a correct
  // sign, due to the choice of euler angles in the AliAlignObj::AnglesToMatrix
  //   costhe*cosphi;                        -costhe*sinphi;                               sinthe;
  //   sinpsi*sinthe*cosphi + cospsi*sinphi; -sinpsi*sinthe*sinphi + cospsi*cosphi; -costhe*sinpsi;
  //  -cospsi*sinthe*cosphi + sinpsi*sinphi;  cospsi*sinthe*sinphi + sinpsi*cosphi;  costhe*cospsi;
  //
  const double kJTol = 1e-4; // treat derivatives below this threshold as 0
  const double sgc[kNDOFGeom] = {1., 1., 1., -RadToDeg(), RadToDeg(), -RadToDeg()};
  //
  double dDPar[kNDOFGeom][4][4] = {
    // dDX[4][4]
    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {rd[0], rd[3], rd[6], 0}},
    // dDY[4][4]
    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {rd[1], rd[4], rd[7], 0}},
    // dDZ[4][4]
    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {rd[2], rd[5], rd[8], 0}},
    // dDPSI[4][4]
    {{rd[2] * ri[3] - rd[1] * ri[6], rd[2] * ri[4] - rd[1] * ri[7], rd[2] * ri[5] - rd[1] * ri[8], 0},
     {rd[5] * ri[3] - rd[4] * ri[6], rd[5] * ri[4] - rd[4] * ri[7], rd[5] * ri[5] - rd[4] * ri[8], 0},
     {rd[8] * ri[3] - rd[7] * ri[6], rd[8] * ri[4] - rd[7] * ri[7], rd[8] * ri[5] - rd[7] * ri[8], 0},
     {rd[2] * ti[1] - rd[1] * ti[2], rd[5] * ti[1] - rd[4] * ti[2], rd[8] * ti[1] - rd[7] * ti[2], 0}},
    // dDTHT[4][4]
    {{rd[0] * ri[6] - rd[2] * ri[0], rd[0] * ri[7] - rd[2] * ri[1], rd[0] * ri[8] - rd[2] * ri[2], 0},
     {rd[3] * ri[6] - rd[5] * ri[0], rd[3] * ri[7] - rd[5] * ri[1], rd[3] * ri[8] - rd[5] * ri[2], 0},
     {rd[6] * ri[6] - rd[8] * ri[0], rd[6] * ri[7] - rd[8] * ri[1], rd[6] * ri[8] - rd[8] * ri[2], 0},
     {rd[0] * ti[2] - rd[2] * ti[0], rd[3] * ti[2] - rd[5] * ti[0], rd[6] * ti[2] - rd[8] * ti[0], 0}},
    // dDPHI[4][4]
    {{rd[1] * ri[0] - rd[0] * ri[3], rd[1] * ri[1] - rd[0] * ri[4], rd[1] * ri[2] - rd[0] * ri[5], 0},
     {rd[4] * ri[0] - rd[3] * ri[3], rd[4] * ri[1] - rd[3] * ri[4], rd[4] * ri[2] - rd[3] * ri[5], 0},
     {rd[7] * ri[0] - rd[6] * ri[3], rd[7] * ri[1] - rd[6] * ri[4], rd[7] * ri[2] - rd[6] * ri[5], 0},
     {rd[1] * ti[0] - rd[0] * ti[1], rd[4] * ti[0] - rd[3] * ti[1], rd[7] * ti[0] - rd[6] * ti[1], 0}},
  };
  //
  for (int cs = 0; cs < kNDOFGeom; cs++) {
    int i = ij[cs][0], j = ij[cs][1];
    for (int ip = 0; ip < kNDOFGeom; ip++) {
      double jval = sgc[cs] * dDPar[ip][i][j] * cf[ip];
      jac[cs * kNDOFGeom + ip] = (Abs(jval) > kJTol) ? jval : 0; // [cs][ip]
    }
  }
}

//______________________________________________________
void AliAlgConstraint::Print(const Option_t*) const
{
  // print info
  printf("Constraint on ");
  for (int i = 0; i < kNDOFGeom; i++)
    if (IsDOFConstrained(i))
      printf("%3s (Sig:%+e) ", GetDOFName(i), GetSigma(i));
  printf(" | %s %s\n", GetName(), GetTitle());
  if (GetNoJacobian())
    printf("!!! This is explicit constraint on sum of DOFs (no Jacobian)!!!\n");
  for (int i = 0; i < GetNChildren(); i++) {
    const AliAlgVol* child = GetChild(i);
    printf("%3d %s\n", i, child->GetName());
  }
}

} // namespace align
} // namespace o2
