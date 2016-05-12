#include "CCDB/ObjectHandler.h"

#include "AliCDBEntry.h"
#include "TBufferFile.h"
#include "TFile.h"
#include "TMessage.h"
#include "TSystem.h"
#include "Rtypes.h"

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

// Compression/decompression code taken from https://panthema.net/2007/0328-ZLibString.html

void ObjectHandler::Compress(const std::string& uncompressed_string, std::string& compressed_string)
{
  // z_stream is zlib's control structure
  z_stream zs;
  memset(&zs, 0, sizeof(zs));

  if (deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK) {
    LOG(ERROR) << "deflateInit failed while compressing";
  }

  zs.next_in = (Bytef*)uncompressed_string.data();
  zs.avail_in = uncompressed_string.size();

  int ret;
  char outbuffer[32768];
  std::string outstring;

  // Get the compressed bytes in blocks of 32768 bytes using repeated calls to deflate
  do {
    zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = deflate(&zs, Z_FINISH);

    if (outstring.size() < zs.total_out) {
      // append the block to the output string
      outstring.append(outbuffer, zs.total_out - outstring.size());
    }
  } while (ret == Z_OK);

  deflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    LOG(ERROR) << "Exception during zlib compression: (" << ret << ") " << zs.msg;
  }

  compressed_string.assign(outstring);
}

void ObjectHandler::Decompress(std::string& uncompressed_string, const std::string& compressed_string)
{
  // z_stream is zlib's control structure
  z_stream zs;
  memset(&zs, 0, sizeof(zs));

  if (inflateInit(&zs) != Z_OK) {
    LOG(ERROR) << "deflateInit failed while decompressing";
  }

  zs.next_in = (Bytef*)compressed_string.data();
  zs.avail_in = compressed_string.size();

  int ret;
  char outbuffer[32768];
  std::string outstring;

  // Get the decompressed bytes in blocks of 32768 bytes using repeated calls to inflate
  do {
    zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = inflate(&zs, 0);

    if (outstring.size() < zs.total_out) {
      outstring.append(outbuffer, zs.total_out - outstring.size());
    }
  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    LOG(ERROR) << "Exception during zlib compression: (" << ret << ") " << zs.msg;
  }

  uncompressed_string.assign(outstring);
}
