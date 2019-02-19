//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliGPUTPCGMPolynomialFieldManager_H
#define AliGPUTPCGMPolynomialFieldManager_H

class AliMagF;
class AliGPUTPCGMPolynomialField;

/**
 * @class AliGPUTPCGMPolynomialFieldManager
 *
 */

class AliGPUTPCGMPolynomialFieldManager
{
public:

  enum StoredField_t  {kUnknown, kUniform, k2kG, k5kG }; // known fitted polynomial fields, stored in constants

  AliGPUTPCGMPolynomialFieldManager(){}
  
  /* Get appropriate pre-calculated polynomial field for the given field value nominalFieldkG
   */
  static int GetPolynomialField( float nominalFieldkG, AliGPUTPCGMPolynomialField &field );


#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

  /* Get pre-calculated polynomial field for the current ALICE field (if exists)
   */
  static int GetPolynomialField( AliGPUTPCGMPolynomialField &field );

  /* Fit given field for TPC
   */
  static int FitFieldTpc( AliMagF* fld, AliGPUTPCGMPolynomialField &field, double step=1. );

  /* Fit given field for TRD
   */
  static int FitFieldTrd( AliMagF* fld, AliGPUTPCGMPolynomialField &field, double step=1. );

  /* Fit given field for ITS
   */
  static int FitFieldIts( AliMagF* fld, AliGPUTPCGMPolynomialField &field, double step=1. );

#endif
  
  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
   */
  static int GetPolynomialField( StoredField_t type, float nominalFieldkG, AliGPUTPCGMPolynomialField &field );
 };

#endif
