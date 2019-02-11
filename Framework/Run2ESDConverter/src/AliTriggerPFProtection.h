#ifndef ALITRIGGERPFPROTECTION_H
#define ALITRIGGERPFPROTECTION_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
//
//  Class to define a Trigger Past Future Protection
//
//                              name      description         INT1  INT2
//    Ej:
//  AliTriggerPFProtection sc( "BARREL", "BARREL DET Pb-Pb",  "SC","PE" );
//  sc.SetINTa("PE");  // Peripheral
//  sc.SetINTb("SC");  // Semicentral
//  sc.SetINT("PE");
//  sc.SetNa1( 5 );
//  sc.SetNa2( 5 );
//  sc.SetTa( 88 );
//  sc.SetNb1( 1 );
//  sc.SetNb2( 0 );
//  sc.SetTb( 88 );
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include <TNamed.h>
#endif

class AliTriggerPFProtection : public TNamed {

public:
                          AliTriggerPFProtection();
                          AliTriggerPFProtection( TString & name);
                          AliTriggerPFProtection( TString & name,TString & inta, TString & intb, TString & interaction );
                          AliTriggerPFProtection( TString & name,TString & inter, TString & bcm,UInt_t PBefore,UInt_t PAfter,UInt_t NBefore,UInt_t NAfter,UInt_t OBefore,UInt_t OAfter);
			  AliTriggerPFProtection(TString& name, UInt_t* pfdef);  
               virtual   ~AliTriggerPFProtection() {}

  //  Setters
                  void   SetINTa( TString& inta ) { fINTa = inta; }
                  void   SetINTb( TString& intb ) { fINTb = intb; }
                  void   SetINT ( TString& inti ) { fINT = inti; }
                  void   SetNa1 ( UInt_t n )    { fNa1 = n; }
                  void   SetNa2 ( UInt_t n )    { fNa2 = n; }
                  void   SetTa  ( UInt_t n )    { fTa  = n; }
                  void   SetNb1 ( UInt_t n )    { fNb1 = n; }
                  void   SetNb2 ( UInt_t n )    { fNb2 = n; }
                  void   SetTb  ( UInt_t n )    { fTb  = n; }
  //  Getters

        const  TString&  GetINTa() const { return fINTa; }
        const  TString&  GetINTb() const { return fINTb; }
        const  TString&  GetINT () const { return fINT; }
                UInt_t   GetNa1 () const { return fNa1; }
                UInt_t   GetNa2 () const { return fNa2; }
                UInt_t   GetTa  () const { return fTa ; }
                UInt_t   GetNb1 () const { return fNb1; }
                UInt_t   GetNb2 () const { return fNb2; }
                UInt_t   GetTb  () const { return fTb ; }
               
                Bool_t   CheckInteractions(TObjArray &interactions) const;
	  virtual void   Print( const Option_t* opt ="" ) const;
               
private:
                TString     fINTa;                // Logical function of INT1 and INT2;
			                          // ver>7: bcmask
                TString     fINTb;                // Logical function of INT1 and INT2;
		                                  // ver>7: interaction
                TString     fINT;                 // Logical function of INT1 and INT2;
		                                  //ver>7: not used
                 UInt_t     fNa1;                 // Max number of INTa for the event of type INT
			 			  // ver>7: NintAfter
                 UInt_t     fNa2;                 // max number of INTa for the event of type NOT INT
		                                  // ver>7:OffsetAfter
                 UInt_t     fTa;                  // mprotection interval for INTa 
		 				  // ver>7: PeriodAfter
                 UInt_t     fNb1;                 // mmax number of INTb   for  the event of type INT 
		 				  // ver>7: NintBefore
                 UInt_t     fNb2;                 // mmax number of INTb for the event of  type NOT INT
		                                  // ver>7:OffsetBefore
                 UInt_t     fTb;                  // mprotection interval for INTb
		                                  // ver>7: PeriodBefore  
		 UInt_t     fPFdef[12];            // 12 words=3 levels * 4; pfcommon,pfblocka,pfblockb,pflut
		 // LML0 PF protection
		 UInt_t fInter;                     // 1=INT1, 2=INT2
		 UInt_t fBCmask;                    // 12 bit BCmask applied to interaction
		 UInt_t fVersion;                   // 7=12 hexa words, 8= human interface

		AliTriggerPFProtection(const AliTriggerPFProtection & inp );
		AliTriggerPFProtection& operator = (const AliTriggerPFProtection& inp);

   ClassDef( AliTriggerPFProtection, 3 )  // Define a Trigger Past Future Protection 
};


#endif
