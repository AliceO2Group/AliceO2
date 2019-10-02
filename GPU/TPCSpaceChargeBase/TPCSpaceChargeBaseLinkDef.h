// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCSpaceChargeBaseLinkDef.h
/// \author Ernst Hellbaer

#if defined(__CINT__) || defined(__CLING__)

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class AliTPC3DCylindricalInterpolator + ;
#pragma link C++ class AliTPC3DCylindricalInterpolatorIrregular + ;
#pragma link C++ struct AliTPC3DCylindricalInterpolatorIrregular::KDTreeNode + ;
#pragma link C++ class AliTPCLookUpTable3DInterpolatorD + ;
#pragma link C++ class AliTPCLookUpTable3DInterpolatorIrregularD + ;
#pragma link C++ class AliTPCPoissonSolver + ;
#pragma link C++ struct AliTPCPoissonSolver::MGParameters + ;
#pragma link C++ class AliTPCSpaceCharge3DCalc + ;

#endif
