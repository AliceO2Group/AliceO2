#ifndef ALITIMESTAMP_H
#define ALITIMESTAMP_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: AliTimeStamp.h 22322 2007-11-22 11:43:14Z cvetan $ */

///////////////////////////////////////////////////////////////////////////////
//
//  Class to define Event Timestamp from : 
//
//               Orbit
//               Period counter
//               Seconds    |
//                  +       | ===> Bunch cross 
//               Microsecs  | 
//
//////////////////////////////////////////////////////////////////////////////
#include <cmath>

class TObject;

class AliTimeStamp : public TObject {

public:
                         AliTimeStamp();
                         AliTimeStamp( UInt_t orbit, UInt_t period, ULong64_t bunchCross );   
                         AliTimeStamp( UInt_t orbit, UInt_t period, 
                                       UInt_t seconds, UInt_t microsecs );   
              virtual   ~AliTimeStamp() {}
                         AliTimeStamp( const AliTimeStamp &timestamp );
         AliTimeStamp&   operator=(const AliTimeStamp& timestamp);
              
      // Getters
               UInt_t    GetOrbit()      const { return fOrbit;     }        
               UInt_t    GetPeriod()     const { return fPeriod;    }       
            ULong64_t    GetBunchCross() const { return fBunchCross; }  
               UInt_t    GetSeconds()    const { return (UInt_t)(fBunchCross/1000000000.*fNanosecPerBC);   }      
               UInt_t    GetMicroSecs()  const { return (UInt_t)(fmod(fBunchCross *fNanosecPerBC, 1000000000.)/1000 ); }     
       virtual Bool_t    IsSortable()    const { return kTRUE; }
     // Setters
                 void    SetTimeStamp( UInt_t orbit, UInt_t period, ULong64_t bunchcross );
                 void    SetTimeStamp( UInt_t orbit, UInt_t period, 
                                       UInt_t seconds, UInt_t microsecs );
              
              
        virtual Int_t    Compare( const TObject* obj ) const;
         virtual void    Print( const Option_t* opt ="" ) const;
                               
   static const Int_t    fNanosecPerBC;   //! nanosecs per bunch cross
              
protected:
              UInt_t    fOrbit;         // Orbit
              UInt_t    fPeriod;        // Period counter
           ULong64_t    fBunchCross;    // Bunch Cross 
//              UInt_t    fSeconds;       // Seconds 
//              UInt_t    fMicroSecs;     // Microsecs  
                         
private:                         

   ClassDef( AliTimeStamp, 1 )  // Define a timestamp
};                                                                         


#endif
