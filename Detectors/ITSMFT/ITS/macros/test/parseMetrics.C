#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "TNtuple.h"
#include "TFile.h"
#endif

enum class Task {
  vtTrackletFinder = 0,
  vtTrackletSel = 1,
  vtFinding = 2,
  lifeTime = 3
};

void parseFileAndFillNtuple(const char* inputFilename, const char* outputFilename)
{
  // Open the input file
  std::ifstream inputFile(inputFilename);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Could not open input file " << inputFilename << std::endl;
    return;
  }

  // Create an NTuple with the desired column names
  TNtuple* ntuple = new TNtuple("metrics", "metrics", "task:offset:stream:elapsed:memory");

  // Parse the input file line by line
  std::string line;
  while (std::getline(inputFile, line)) {
    // Parse the columns from the line using tabs as the delimiter
    std::string name;
    float time;
    int stream;
    int memory;
    std::istringstream iss(line);
    iss >> name >> stream >> time >> memory;

    // Split the name column by "_" character
    size_t underscorePos = name.find('_');
    if (underscorePos != std::string::npos) {
      std::string task = name.substr(0, underscorePos);
      int taskNum{-1};
      if (task == "vtTrackletFinder") {
        taskNum = static_cast<int>(Task::vtTrackletFinder);
      } else if (task == "vtTrackletSel") {
        taskNum = static_cast<int>(Task::vtTrackletSel);
      } else if (task == "vtFinding") {
        taskNum = static_cast<int>(Task::vtFinding);
      } else if (task == "lifeTime") {
        taskNum = static_cast<int>(Task::lifeTime);
      }

      std::string offset = name.substr(underscorePos + 1);
      // Fill the NTuple with the parsed values, including the split columns
      ntuple->Fill(taskNum, std::atoi(offset.c_str()), stream, time, memory);
    } else {
      std::cerr << "Warning: Invalid name format in line: " << line << std::endl;
    }
  }

  // Close the input file
  inputFile.close();

  // Create the output ROOT file
  TFile* outputFile = new TFile(outputFilename, "RECREATE");

  // Write the filled NTuple to the output ROOT file
  ntuple->Write();

  // Close the output ROOT file
  outputFile->Close();

  // Print the contents of the NTuple
  ntuple->Scan();
}

void parseMetrics(std::string fname = "metrics.txt")
{
  // Call the function with the filenames of your input and output files
  parseFileAndFillNtuple(fname.c_str(), "metrics.root");
}
