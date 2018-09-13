//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPolynomialFieldManager_H
#define AliHLTTPCGMPolynomialFieldManager_H

class AliMagF;
class AliHLTTPCGMPolynomialField;

/**
 * @class AliHLTTPCGMPolynomialFieldManager
 *
 */

class AliHLTTPCGMPolynomialFieldManager
{
public:

  enum StoredField_t  {kUnknown, kUniform, k2kG, k5kG }; // known fitted polynomial fields, stored in constants

  AliHLTTPCGMPolynomialFieldManager(){}
  
  /* Get appropriate pre-calculated polynomial field for the given field value nominalFieldkG
   */
  static int GetPolynomialField( float nominalFieldkG, AliHLTTPCGMPolynomialField &field );


#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)

  /* Get pre-calculated polynomial field for the current ALICE field (if exists)
   */
  static int GetPolynomialField( AliHLTTPCGMPolynomialField &field );

  /* Fit given field for TPC
   */
  static int FitFieldTPC( AliMagF* fld, AliHLTTPCGMPolynomialField &field, double step=1. );

  /* Fit given field for TRD
   */
  static int FitFieldTRD( AliMagF* fld, AliHLTTPCGMPolynomialField &field, double step=1. );

#endif
  
  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
   */
  static int GetPolynomialField( StoredField_t type, float nominalFieldkG, AliHLTTPCGMPolynomialField &field );
 };

#endif
