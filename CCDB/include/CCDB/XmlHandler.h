// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICE_O2_XML_HANDLER_H_
#define ALICE_O2_XML_HANDLER_H_
//  The SAX XML file handler used by the OCDB Manager                     //
//  get the OCDB Folder <-> Run Range correspondance                      //

#include <TObject.h> // for TObject
#include "Rtypes.h"  // for Int_t, XmlHandler::Class, ClassDef, etc
#include "TString.h" // for TString

class TList;

#include <cstddef> // for NULL

namespace o2
{
namespace ccdb
{

class IdRunRange;

class XmlHandler : public TObject
{

 public:
  XmlHandler();

  XmlHandler(Int_t run);

  XmlHandler(const XmlHandler& sh);

  ~XmlHandler() override;

  XmlHandler& operator=(const XmlHandler& sh);

  // functions to interface to TSAXHandler
  void OnStartDocument();

  void OnEndDocument();

  void OnStartElement(const char* name, const TList* attributes);

  void OnEndElement(const char* name);

  void OnCharacters(const char* name);

  void OnComment(const char* name);

  void OnWarning(const char* name);

  void OnError(const char* name);

  void OnFatalError(const char* name);

  void OnCdataBlock(const char* name, Int_t len);

  Int_t getStartIdRunRange() const
  {
    return mStartIdRunRange;
  }

  Int_t getEndIdRunRange() const
  {
    return mEndIdRunRange;
  }

  TString getOcdbFolder() const
  {
    return mOCDBFolder;
  }

  void setRun(Int_t run)
  {
    mRun = run;
  }

 private:
  Int_t mRun;             // run for which the LHC Period Folder has to be found
  Int_t mStartIdRunRange; // start run corresponding to the request
  Int_t mEndIdRunRange;   // end run corresponding to the request
  TString mOCDBFolder;    // OCDB folder corresponding to the request

  ClassDefOverride(XmlHandler, 0); // The XML file handler for the OCDB
};
} // namespace ccdb
} // namespace o2
#endif
