// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_MAPPING_TEST_INPUT_DOCUMENT_H
#define O2_MCH_MAPPING_TEST_INPUT_DOCUMENT_H

#include <rapidjson/document.h>
#include <cstdio>
#include <rapidjson/filereadstream.h>

template <typename StreamType>
class InputDocument
{
 public:
  InputDocument(const char* filename)
    : mFile(fopen(filename, "r")),
      mReadBuffer(new char[65536]),
      mStream(mFile, mReadBuffer, sizeof(mReadBuffer)),
      mDocument()
  {
    mDocument.ParseStream(mStream);
  }

  rapidjson::Document& document() { return mDocument; }

  virtual ~InputDocument()
  {
    delete[] mReadBuffer;
    fclose(mFile);
  }

 private:
  FILE* mFile;
  char* mReadBuffer;
  StreamType mStream;
  rapidjson::Document mDocument;
};

using InputWrapper = InputDocument<rapidjson::FileReadStream>;

#endif
