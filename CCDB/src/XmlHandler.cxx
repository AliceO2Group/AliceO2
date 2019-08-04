// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  The SAX XML file handler used in the CDBManager                       //
#include "CCDB/XmlHandler.h"
#include <fairlogger/Logger.h> // for LOG
#include <TList.h>             // for TList
#include <TXMLAttr.h>          // for TXMLAttr

using namespace o2::ccdb;
ClassImp(XmlHandler);

XmlHandler::XmlHandler() : TObject(), mRun(-1), mStartIdRunRange(-1), mEndIdRunRange(-1), mOCDBFolder("")
{
  //
  // XmlHandler default constructor
  //
}

XmlHandler::XmlHandler(Int_t run) : TObject(), mRun(run), mStartIdRunRange(-1), mEndIdRunRange(-1), mOCDBFolder("")
{
  //
  // XmlHandler constructor with requested run
  //
}

XmlHandler::XmlHandler(const XmlHandler&) = default;

XmlHandler& XmlHandler::operator=(const XmlHandler& sh)
{
  //
  // Assignment operator
  //
  if (&sh == this) {
    return *this;
  }

  new (this) XmlHandler(sh);
  return *this;
}

XmlHandler::~XmlHandler() = default;

void XmlHandler::OnStartDocument()
{
  // if something should happen right at the beginning of the
  // XML document, this must happen here
  LOG(INFO) << "Reading XML file for LHCPeriod <-> Run Range correspondence";
}

void XmlHandler::OnEndDocument()
{
  // if something should happen at the end of the XML document
  // this must be done here
}

void XmlHandler::OnStartElement(const char* name, const TList* attributes)
{
  // when a new XML element is found, it is processed here

  // set the current system if necessary
  TString strName(name);
  LOG(DEBUG) << "name = " << strName.Data();
  Int_t startRun = -1;
  Int_t endRun = -1;
  TXMLAttr* attr;
  TIter next(attributes);
  while ((attr = (TXMLAttr*)next())) {
    TString attrName = attr->GetName();
    LOG(DEBUG) << "Name = " << attrName.Data();
    if (attrName == "StartIdRunRange") {
      startRun = (Int_t)(((TString)(attr->GetValue())).Atoi());
      LOG(DEBUG) << "startRun = " << startRun;
    }
    if (attrName == "EndIdRunRange") {
      endRun = (Int_t)(((TString)(attr->GetValue())).Atoi());
      LOG(DEBUG) << "endRun = " << endRun;
    }
    if (attrName == "OCDBFolder") {
      if (mRun >= startRun && mRun <= endRun && startRun != -1 && endRun != -1) {
        mOCDBFolder = (TString)(attr->GetValue());
        LOG(DEBUG) << "OCDBFolder = " << mOCDBFolder.Data();
        mStartIdRunRange = startRun;
        mEndIdRunRange = endRun;
      }
    }
  }
  return;
}

void XmlHandler::OnEndElement(const char* name)
{
  // do everything that needs to be done when an end tag of an element is found
  TString strName(name);
  LOG(DEBUG) << "name = " << strName.Data();
}

void XmlHandler::OnCharacters(const char* characters)
{
  // copy the text content of an XML element
  // mContent = characters;
  TString strCharacters(characters);
  LOG(DEBUG) << "characters = " << strCharacters.Data();
}

void XmlHandler::OnComment(const char* /*text*/)
{
  // comments within the XML file are ignored
}

void XmlHandler::OnWarning(const char* text)
{
  // process warnings here
  LOG(INFO) << "Warning: " << text;
}

void XmlHandler::OnError(const char* text)
{
  // process errors here
  LOG(ERROR) << "Error: " << text;
}

void XmlHandler::OnFatalError(const char* text)
{
  // process fatal errors here
  LOG(FATAL) << "Fatal error: " << text;
}

void XmlHandler::OnCdataBlock(const char* /*text*/, Int_t /*len*/)
{
  // process character data blocks here
  // not implemented and should not be used here
}
