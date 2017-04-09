//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   runComponent.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  run a component encapsulating ALICE HLT code

// TODO: this file is intended to evolve into a unit test

#include "aliceHLTwrapper/SystemInterface.h"
#include "aliceHLTwrapper/Component.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>

using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;
using std::stringstream;
using std::vector;

int main(int argc, char** argv)
{
  int iResult = 0;
  // parse options
  const char* inputFileName = nullptr;
  const char* outputFileName = nullptr;

  vector<char*> componentOptions;
  for (int i = 0; i < argc; i++) {
    char* arg = argv[i];
    switch (arg[0]) {
      case '-':
        if (arg[1] != 0 && arg[2] == 0) { // one char after the '-'
          if (arg[1] == 'i' || arg[1] == 'o') {
            if (i + 1 >= argc) {
              cerr << "missing file name for option " << arg << endl;
            } else if (arg[1] == 'i')
              inputFileName = argv[i + 1];
            else
              outputFileName = argv[i + 1];
            break;
          }
        }
      // intended fall-through
      default:
        componentOptions.push_back(arg);
    }
  }

  ALICE::HLT::Component component;
  if ((iResult = component.init(componentOptions.size(), &componentOptions[0])) < 0) {
    cerr << "error: init failed with " << iResult << endl;
    // the ALICE HLT external interface uses the following error definition
    // 0 success
    // >0 error number
    return -iResult;
  }

  vector<o2::AliceHLT::MessageFormat::BufferDesc_t> blockData;
  char* inputBuffer = nullptr;
  if (inputFileName) {
    std::ifstream input(inputFileName, std::ifstream::binary);
    if (input) {
      // get length of file:
      input.seekg(0, input.end);
      int length = input.tellg();
      input.seekg(0, input.beg);

      // allocate memory:
      inputBuffer = new char[length];
      input.read(inputBuffer, length);
      input.close();
      blockData.emplace_back(reinterpret_cast<unsigned char*>(inputBuffer), length);
    }
  }
  if ((iResult = component.process(blockData)) < 0) {
    cerr << "error: init failed with " << iResult << endl;
  }
  if (inputBuffer) delete[] inputBuffer;
  inputBuffer = nullptr;
  if (iResult < 0) return -iResult;

  // for now, only the first buffer is written
  if (blockData.size() > 0) {
    if (outputFileName != nullptr) {
      ofstream outputFile(outputFileName);
      if (outputFile.good()) {
        outputFile.write(reinterpret_cast<const char*>(blockData[0].mP), blockData[0].mSize);
        outputFile.close();
      }
    } else {
      cerr << "WARNING: dropping " << blockData.size()
           << " data block(s) produced by component, use option '-o' to specify output file" << endl;
    }
  }
}
