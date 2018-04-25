//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPolynomialFieldCreator_H
#define AliHLTTPCGMPolynomialFieldCreator_H

class AliMagF;
class AliHLTTPCGMPolynomialField;

/**
 * @class AliHLTTPCGMPolynomialFieldCreator
 *
 */

class AliHLTTPCGMPolynomialFieldCreator
{
public:

  enum StoredField_t  {kUnknown, kUniform, k2kG, k5kG }; // known fitted polynomial fields, stored in constants

  AliHLTTPCGMPolynomialFieldCreator(){}
  
  /* Get appropriate pre-calculated polynomial field for the given field value nominalFieldkG
   */
  static int GetPolynomialField( float nominalFieldkG, AliHLTTPCGMPolynomialField &field );


#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)

  /* Get pre-calculated polynomial field for the current ALICE field (if exists)
   */  
  static int GetPolynomialField( AliHLTTPCGMPolynomialField &field );

  /* Fit given field
   */
  static int FitField( AliMagF* fld, AliHLTTPCGMPolynomialField &field ); 

#endif
  
  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
   */
  static int GetPolynomialField( StoredField_t type, float nominalFieldkG, AliHLTTPCGMPolynomialField &field );
 };

#endif
