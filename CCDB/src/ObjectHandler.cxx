// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CCDB/ObjectHandler.h"

#include "TBufferFile.h"
#include "TFile.h"

#include <FairMQLogger.h>

#include <zlib.h>

using namespace o2::ccdb;

ObjectHandler::ObjectHandler() = default;

ObjectHandler::~ObjectHandler() = default;

void ObjectHandler::GetObject(const std::string& path, std::string& object)
{
  TFile* file = new TFile(path.c_str());

  // If file was not found or empty
  if (file->IsZombie()) {
    LOG(ERROR) << "The object was not found at " << path;
  }

  // Get the AliCDBEntry from the root file
  // we cast it directly to TObject (to avoid a link dependency on AliRoot here)
  TObject* entry = file->Get("AliCDBEntry");

  // Create an outcoming buffer
  TBufferFile* buffer = new TBufferFile(TBuffer::kWrite);

  // Stream and serialize the AliCDBEntry object to the buffer
  buffer->WriteObject((const TObject*)entry);

  // Obtain a pointer to the buffer
  char* pointer = buffer->Buffer();

  // Store the object to the referenced string
  object.assign(pointer, buffer->Length());

  // LOG(INFO) << "Object length: " << object.size();

  // Release the open file
  delete file;

  delete buffer;

  delete entry;
}
