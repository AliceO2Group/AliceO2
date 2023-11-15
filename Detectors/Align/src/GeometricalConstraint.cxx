// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   GeometricalConstraint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Descriptor of geometrical constraint

#include "Align/GeometricalConstraint.h"
#include "Align/AlignConfig.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include <TGeoMatrix.h>
#include <TMath.h>
#include <cstdio>

ClassImp(o2::align::GeometricalConstraint);

using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//______________________________________________________
void GeometricalConstraint::writeChildrenConstraints(FILE* conOut) const
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
  bool doJac = !getNoJacobian(); // do we need jacobian evaluation?
  int nch = getNChildren();
  std::unique_ptr<double[]> cstrArr(new double[nch * kNDOFGeom * kNDOFGeom]);
  memset(cstrArr.get(), 0, nch * kNDOFGeom * kNDOFGeom * sizeof(double));
  // we need for each children the matrix for vector transformation from children frame
  // (in which its DOFs are defined, LOC or TRA) to this parent variation frame
  // matRel = mPar^-1*mChild
  TGeoHMatrix mPar;
  //
  const auto& algConf = AlignConfig::Instance();
  // in case of parent assigned use its matrix,
  // otherwise Alice global frame is assumed to be the parent -> Unit matrix
  if (mParent && doJac) {
    if (mParent->isFrameTRA()) {
      mParent->getMatrixT2G(mPar);
    } // tracking to global
    else {
      mPar = mParent->getMatrixL2GIdeal();
    } // local to global
    mPar = mPar.Inverse();
  }
  //
  auto jac = cstrArr.get();
  int nContCh[kNDOFGeom] = {0}; // we need at least one contributing children DOF to constrain the parent DOF
  for (int ich = 0; ich < nch; ich++) {
    auto child = getChild(ich);
    if (doJac) { // calculate jacobian
      TGeoHMatrix matRel;
      if (child->isFrameTRA()) {
        child->getMatrixT2G(matRel);
      } // tracking to global
      else {
        matRel = child->getMatrixL2GIdeal();
      } // local to global
      matRel.MultiplyLeft(&mPar);
      constrCoefGeom(matRel, jac);
      //
      for (int ics = 0; ics < kNDOFGeom; ics++) { // DOF of parent to be constrained
        for (int ip = 0; ip < kNDOFGeom; ip++) {  // count contributing DOFs
          double jv = jac[ics * kNDOFGeom + ip];
          if (!isZeroAbs(jv) && child->isFreeDOF(ip) && child->getParErr(ip) >= 0) {
            nContCh[ip]++;
          }
        }
      }
    } else { // simple constraint on the sum of requested DOF
      for (int ip = 0; ip < kNDOFGeom; ip++) {
        if (child->isFreeDOF(ip) && child->getParErr(ip) >= 0) {
          nContCh[ip]++;
        }
        jac[ip * kNDOFGeom + ip] = 1.;
      }
    }
    jac += kNDOFGeom * kNDOFGeom; // matrix for next slot
  }
  for (int ics = 0; ics < kNDOFGeom; ics++) {
    if (!isDOFConstrained(ics)) {
      continue;
    }
    int cmtStatus = nContCh[ics] > 0 ? kOff : kOn; // do we comment this constraint?
    if (cmtStatus) {
      if (algConf.verbose > 0) {
        LOG(info) << "No contributors to constraint of " << getDOFName(ics) << " of " << getName();
      }
    }
    if (mSigma[ics] > 0) {
      fprintf(conOut, "\n%s%s\t%e\t%e\t%s %s of %s Auto\n", comment[cmtStatus], kKeyConstr[kMeas], 0.0, mSigma[ics],
              comment[kOnOn], getDOFName(ics), getName().c_str());
    } else {
      fprintf(conOut, "\n%s%s\t%e\t%s %s of %s Auto\n", comment[cmtStatus], kKeyConstr[kConstr], 0.0,
              comment[kOnOn], getDOFName(ics), getName().c_str());
    }
    for (int ich = 0; ich < nch; ich++) { // contribution from this children DOFs to constraint
      auto child = getChild(ich);
      jac = cstrArr.get() + kNDOFGeom * kNDOFGeom * ich;
      if (cmtStatus) {
        fprintf(conOut, "%s", comment[cmtStatus]);
      } // comment out contribution
      // first write real constraints
      for (int ip = 0; ip < kNDOFGeom; ip++) {
        double jv = jac[ics * kNDOFGeom + ip];
        if (child->isFreeDOF(ip) && !isZeroAbs(jv) && child->getParErr(ip) >= 0) {
          fprintf(conOut, "%9d %+.3e\t", child->getParLab(ip), jv);
        }
      } // loop over DOF's of children contributing to this constraint
      // now, after comment, write disabled constraints
      fprintf(conOut, "%s ", comment[kOn]);
      if (doJac) {
        for (int ip = 0; ip < kNDOFGeom; ip++) {
          double jv = jac[ics * kNDOFGeom + ip];
          if (child->isFreeDOF(ip) && !isZeroAbs(jv) && child->getParErr(ip) >= 0) {
            continue;
          }
          fprintf(conOut, "%9d %+.3e\t", child->getParLab(ip), jv);
        } // loop over DOF's of children contributing to this constraint
      }
      fprintf(conOut, "%s from %s\n", comment[kOnOn], child->getSymName());
    } // loop over children
  }   // loop over constraints in parent volume
}

//______________________________________________________
void GeometricalConstraint::checkConstraint() const
{
  // check how the constraints are satysfied
  int nch = getNChildren();
  if (!nch) {
    return;
  }
  //
  bool doJac = !getNoJacobian(); // do we need jacobian evaluation?
  std::unique_ptr<double[]> cstrArr(new double[nch * kNDOFGeom * kNDOFGeom]);
  memset(cstrArr.get(), 0, nch * kNDOFGeom * kNDOFGeom * sizeof(double));
  // we need for each children the matrix for vector transformation from children frame
  // (in which its DOFs are defined, LOC or TRA) to this parent variation frame
  // matRel = mPar^-1*mChild
  TGeoHMatrix mPar;
  // in case of parent assigned use its matrix,
  // otherwise Alice global frame is assumed to be the parent -> Unit matrix
  if (mParent && doJac) {
    if (mParent->isFrameTRA()) {
      mParent->getMatrixT2G(mPar);
    } // tracking to global
    else {
      mPar = mParent->getMatrixL2GIdeal();
    } // local to global
    mPar = mPar.Inverse();
  }
  //
  auto jac = cstrArr.get();
  double parsTotEx[kNDOFGeom] = {0}; // explicitly calculated total modification
  double parsTotAn[kNDOFGeom] = {0}; // analyticaly calculated total modification
  //
  printf("\n\n ----- Constraints Validation for %s Auto ------\n", getName().c_str());
  printf(" chld| ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    printf("  %3s:%3s An/Ex  |", getDOFName(jp), isDOFConstrained(jp) ? "ON " : "OFF");
  }
  printf(" | ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    printf("  D%3s   ", getDOFName(jp));
  }
  printf(" ! %s\n", getName().c_str());
  for (int ich = 0; ich < nch; ich++) {
    auto child = getChild(ich);
    double parsC[kNDOFGeom] = {0}, parsPAn[kNDOFGeom] = {0}, parsPEx[kNDOFGeom] = {0};
    for (int jc = kNDOFGeom; jc--;) {
      parsC[jc] = child->getParVal(jc);
    } // child params in child frame
    printf("#%3d | ", ich);
    //
    if (doJac) {
      TGeoHMatrix matRel;
      if (child->isFrameTRA()) {
        child->getMatrixT2G(matRel);
      } // tracking to global
      else {
        matRel = child->getMatrixL2GIdeal();
      } // local to global
      //
      matRel.MultiplyLeft(&mPar);
      constrCoefGeom(matRel, jac); // Jacobian for analytical constraint used by MillePede
                                   //
      TGeoHMatrix tau;
      child->delta2Matrix(tau, parsC); // child correction matrix in the child frame
      const TGeoHMatrix& matreli = matRel.Inverse();
      tau.Multiply(&matreli);
      tau.MultiplyLeft(&matRel); //  child correction matrix in the parent frame
      detectors::AlignParam tmpPar;
      //      tmpPar.SetMatrix(tau);
      // SetMatrix does setTranslation and setRotation afterwars;
      tmpPar.setTranslation(tau);
      tmpPar.setRotation(tau);
      //tmpPar.GetTranslation(&parsPEx[0]);
      // get Translation gets x,y,z;
      parsPEx[0] = tmpPar.getX();
      parsPEx[1] = tmpPar.getY();
      parsPEx[2] = tmpPar.getZ();
      //tmpPar.GetAngles(&parsPEx[3]); // explicitly calculated child params in parent frame
      // gets angles
      parsPEx[3] = tmpPar.getPsi();
      parsPEx[4] = tmpPar.getTheta();
      parsPEx[5] = tmpPar.getPhi();
      //
      // analytically calculated child params in parent frame
      for (int jp = 0; jp < kNDOFGeom; jp++) {
        for (int jc = 0; jc < kNDOFGeom; jc++) {
          parsPAn[jp] += jac[jp * kNDOFGeom + jc] * parsC[jc];
        }
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
        bool acc = child->isFreeDOF(jc) && child->getParErr(jc) >= 0;
        if (acc) {
          printf("    %+.3e    ", parsC[jc]);
          parsTotAn[jc] += parsC[jc];
        } else {
          printf(" /* %+.3e */ ", parsC[jc]);
        } // just for info, not in the constraint
      }
    }
    printf(" | ");
    for (int jc = 0; jc < kNDOFGeom; jc++) {
      printf("%+.1e ", parsC[jc]);
    } // child proper corrections
    printf(" ! %s\n", child->getSymName());
  }
  //
  printf(" Tot | ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    if (doJac) {
      printf("%+.1e/%+.1e ", parsTotAn[jp], parsTotEx[jp]);
    } else {
      if (isDOFConstrained(jp)) {
        printf("    %+.3e    ", parsTotAn[jp]);
      } else {
        printf(" /* %+.3e */ ", parsTotAn[jp]);
      }
    }
  }
  printf(" | ");
  if (mParent) {
    for (int jp = 0; jp < kNDOFGeom; jp++) {
      printf("%+.1e ", mParent->getParVal(jp));
    }
  } // parent proper corrections
  else {
    printf(" no parent -> %s ", doJac ? "Global" : "Simple");
  }
  printf(" ! <----- %s\n", getName().c_str());
  //
  printf(" Sig | ");
  for (int jp = 0; jp < kNDOFGeom; jp++) {
    if (isDOFConstrained(jp)) {
      printf("    %+.3e    ", mSigma[jp]);
    } else {
      printf(" /* %+.3e */ ", mSigma[jp]);
    }
  }
  printf(" ! <----- \n");
}

//_________________________________________________________________
void GeometricalConstraint::constrCoefGeom(const TGeoHMatrix& matRD, double* jac /*[kNDOFGeom][kNDOFGeom]*/) const
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
  //// the angles are in radians, while we use sinX->X approximation...
  // const double cf[kNDOFGeom] = {1., 1., 1., DegToRad(), DegToRad(), DegToRad()};
  // const double sgc[kNDOFGeom] = {1., 1., 1., -RadToDeg(), RadToDeg(), -RadToDeg()};

  // the angles are in radians
  const double cf[kNDOFGeom] = {1., 1., 1., 1., 1., 1.};
  const double sgc[kNDOFGeom] = {1., 1., 1., -1., 1., -1.};

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
void GeometricalConstraint::print() const
{
  // print info
  printf("Constraint on ");
  for (int i = 0; i < kNDOFGeom; i++) {
    if (isDOFConstrained(i)) {
      printf("%3s (Sig:%+e) ", getDOFName(i), getSigma(i));
    }
  }
  printf(" | %s Auto\n", getName().c_str());
  if (getNoJacobian()) {
    printf("!!! This is explicit constraint on sum of DOFs (no Jacobian)!!!\n");
  }
  for (int i = 0; i < getNChildren(); i++) {
    auto child = getChild(i);
    printf("%3d %s\n", i, child->getSymName());
  }
}

} // namespace align
} // namespace o2
