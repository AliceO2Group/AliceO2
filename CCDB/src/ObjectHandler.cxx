#include "CCDB/ObjectHandler.h"

#include "AliCDBEntry.h"
#include "TBufferFile.h"
#include "TFile.h"

#include "FairMQLogger.h"

#include <zlib.h>

using namespace AliceO2::CDB;

ObjectHandler::ObjectHandler() {}

ObjectHandler::~ObjectHandler() {}

void ObjectHandler::GetObject(const std::string& path, std::string& object)
{
  TFile* file = new TFile(path.c_str());

  // If file was not found or empty
  if (file->IsZombie()) {
    LOG(ERROR) << "The object was not found at " << path;
  }

  // Get the AliCDBEntry from the root file
  AliCDBEntry* entry = (AliCDBEntry*)file->Get("AliCDBEntry");

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
