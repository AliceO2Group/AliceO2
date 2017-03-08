#ifndef ALIRAWDATAERRORLOG_H
#define ALIRAWDATAERRORLOG_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/////////////////////////////////////////////////////////////////////
//                                                                 //
// class AliRawDataErrorLog                                        //
// This is a class for logging raw-data related errors.            //
// It is used to record and retrieve of the errors                 //
// during the reading and reconstruction of raw-data and ESD       //
// analysis.                                                       //
// Further description of the methods and functionality are given  //
// inline.                                                         //
//                                                                 //
// cvetan.cheshkov@cern.ch                                         //
//                                                                 //
/////////////////////////////////////////////////////////////////////

#include <TNamed.h>

class AliRawDataErrorLog: public TNamed {

 public:

  enum ERawDataErrorLevel {
    kMinor = 1, 
    kMajor = 2, 
    kFatal = 3
  };

  AliRawDataErrorLog();
  AliRawDataErrorLog(Int_t eventNumber, Int_t ddlId,
		     ERawDataErrorLevel errorLevel,
		     Int_t errorCode,
		     const char *message = NULL);
  AliRawDataErrorLog(const AliRawDataErrorLog & source);
  AliRawDataErrorLog & operator=(const AliRawDataErrorLog & source);
  virtual ~AliRawDataErrorLog() {};
  virtual void Copy(TObject &obj) const;
  
  Int_t              GetEventNumber() const { return fEventNumber; }
  Int_t              GetDdlID()       const { return fDdlID; }
  ERawDataErrorLevel GetErrorLevel()  const { return fErrorLevel; }
  const char*        GetErrorLevelAsString() const;
  Int_t              GetErrorCode()   const { return fErrorCode; }
  const char *       GetMessage()     const { return fName.Data(); }
  Int_t              GetCount()       const { return fCount; }

  Bool_t            IsSortable() const {return kTRUE;}
  Int_t             Compare(const TObject* obj) const;

  void               AddCount() { fCount++; }

  void Print(Option_t* opt="") const;
  
 private:

  Int_t              fEventNumber; // Event number as it appears in the input raw-data file
  Int_t              fDdlID;       // ID of the DLL in which the error occured
  ERawDataErrorLevel fErrorLevel;  // Level of the raw data error
  Int_t              fErrorCode;   // Code of the raw data error (detector-specific)
  Int_t              fCount;       // Counter of identical errors (occurances)

  ClassDef(AliRawDataErrorLog, 3)
};

#endif
