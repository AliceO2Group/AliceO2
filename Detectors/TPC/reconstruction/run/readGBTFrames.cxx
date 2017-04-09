///
/// @file   readGBTFrames.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>

#include <thread>
#include <mutex>
#include <chrono>

#include "TPCReconstruction/GBTFrameContainer.h"
#include "TPCReconstruction/HalfSAMPAData.h"
#include "TPCReconstruction/DigitData.h"

namespace bpo = boost::program_options;

std::mutex mtx;

bool isVector1 (std::vector<int>& vec) {
  for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    if ((*it) != 1) return false;
  }
  return true;
}

void addData(o2::TPC::GBTFrameContainer& container, std::string& infile, int& done) {
  done = 0;
  mtx.lock();
  std::cout << infile << std::endl;
  mtx.unlock();

  container.addGBTFramesFromBinaryFile(infile);

  done = 1;
}

void readData(o2::TPC::GBTFrameContainer& container, std::vector<std::ofstream*>& outfiles, int& run, int& done) {
  done = 0;
//  std::vector<AliceO2::TPC::HalfSAMPAData> data;
  std::vector<o2::TPC::DigitData> data;
  int i;
  while (!run) {
    std::this_thread::sleep_for(std::chrono::microseconds{100});
    while (container.getData(data)){
//      for (i = 0; i < 5; ++i) {
//        if (outfiles[i] != nullptr) {
//          outfiles[i]->write(reinterpret_cast<const char*>(&data[i].getData()[0]), 16*sizeof(data[i].getData()[0]));
////          outfiles[i]->write((char*)&data[i].getData(),16*sizeof(short));
//        }
//      }
      data.clear();
    }
  }
  done = 1;
}

int main(int argc, char *argv[])
{
  // Arguments parsing
  std::vector<unsigned> size(1,0);
  std::vector<unsigned> CRU(1,0);
  std::vector<unsigned> link(1,0);
  std::vector<std::string> infile(1,"NOFILE");
  std::vector<std::string> outfile(1,"NOFILE");
  std::string adcInFile = "NOFILE";
  bool checkAdcClock = false;
  bool compileAdcValues = false;
  bool keepGbtFrames = false;

  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::vector<std::string>>(&infile),   "Input data files")
    ("outfile,o",   bpo::value<std::vector<std::string>>(&outfile),  "Output data files")
    (",n",          bpo::value<std::vector<unsigned>>(&size),        "Container sizes")
    ("CRU",         bpo::value<std::vector<unsigned>>(&CRU),         "CRUs")
    ("link",        bpo::value<std::vector<unsigned>>(&link),        "links")
    ("clock,c",     bpo::bool_switch(&checkAdcClock),                "check ADC clock")
    ("ADC,a",       bpo::bool_switch(&compileAdcValues),             "compiles the ADC values")
    ("keep,k",      bpo::bool_switch(&keepGbtFrames),                "keeps the GBT frames in memory")
    ("file,f",      bpo::value<std::string>(&adcInFile),             "ADC input file");
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  if (adcInFile == "NOFILE") {

  // Actual "work"
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::vector<o2::TPC::GBTFrameContainer*> container;
  unsigned iSize;
  unsigned iCRU;
  unsigned iLink;
  std::vector<int> addData_done(infile.size(), 0);
  std::vector<int> reading_done(infile.size(), 0);
  for (int i = 0; i < infile.size(); ++i) {
    if (size.size() >= infile.size()) iSize = size[i];
    else iSize = size[0];
    if (CRU.size() >= infile.size()) iCRU = CRU[i];
    else iCRU = CRU[0];
    if (link.size() >= infile.size()) iLink = link[i];
    else iLink = link[0];

    container.push_back(new o2::TPC::GBTFrameContainer(iSize,iCRU,iLink));
    container.back()->setEnableAdcClockWarning(checkAdcClock);
    container.back()->setEnableSyncPatternWarning(false);
    container.back()->setEnableStoreGBTFrames(keepGbtFrames);
    container.back()->setEnableCompileAdcValues(compileAdcValues);
  }

  std::vector<std::thread> threads;

  start = std::chrono::system_clock::now();
  for (int i = 0; i < infile.size(); ++i) {
    if (infile[i] != "NOFILE")
      threads.emplace_back(addData,std::ref(*container[i]), std::ref(infile[i]), std::ref(addData_done[i]));
  }

  std::vector<std::vector<std::ofstream*>> outfiles(infile.size());
  std::ofstream* out;
  for (int i = 0; i < infile.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      std::string outname;
      if (i >= outfile.size()) {
        outfiles[i].push_back(nullptr);
      } else if (outfile[i] == "NOFILE") {
        outfiles[i].push_back(nullptr);
      } else {
        outname = outfile[i];
        outname += "_SAMPA_";
        outname += std::to_string(j);
        outname += "_LOW";
        outname += ".adc";

        out = new std::ofstream(outname, std::ios::out | std::ios::binary);
        outfiles[i].push_back(out);
      }

      if (j == 2) continue;
      outname = "";
      if (i >= outfile.size()) {
        outfiles[i].push_back(nullptr);
      } else if (outfile[i] == "NOFILE") {
        outfiles[i].push_back(nullptr);
      } else {
        outname = outfile[i];
        outname += "_SAMPA_";
        outname += std::to_string(j);
        outname += "_HIGH";
        outname += ".adc";

        out = new std::ofstream(outname, std::ios::out | std::ios::binary);
        outfiles[i].push_back(out);
      }
    }
    threads.emplace_back(readData,std::ref(*container[i]), std::ref(outfiles[i]), std::ref(addData_done[i]), std::ref(reading_done[i]));
  }

  std::this_thread::sleep_for(std::chrono::seconds{1});
  std::cout << std::endl;
  std::cout << " Container | stored frames | processed frames | avail. ADC values " << std::endl;
  std::cout << "-----------|---------------|------------------|-------------------" << std::endl;
  while (not isVector1(reading_done)) {
    std::this_thread::sleep_for(std::chrono::seconds{1});
    for (std::vector<o2::TPC::GBTFrameContainer*>::iterator it = container.begin(); it != container.end(); ++it) {
      mtx.lock();
      std::cout << " " << std::right 
        << std::setw(9) << std::distance(container.begin(), it) << " | " 
        << std::setw(13) << (*it)->getSize() << " | " 
        << std::setw(16) << (*it)->getNFramesAnalyzed() << " | " 
        << std::setw(17) << (*it)->getNentries() << std::endl;
      mtx.unlock();
    }
  }

  for (auto &aThread : threads) {
    aThread.join();
  }
  end = std::chrono::system_clock::now();

  std::cout << std::endl << std::endl;
  std::cout << "Summary:" << std::endl;

  unsigned framesProcessed = 0;
  for (std::vector<o2::TPC::GBTFrameContainer*>::iterator it = container.begin(); it != container.end(); ++it) {
    framesProcessed += (*it)->getNFramesAnalyzed();
    std::cout << "Container " << std::distance(container.begin(), it) << " analyzed " << (*it)->getNFramesAnalyzed() << " GBT Frames" << std::endl;
    delete (*it);
  }
  std::chrono::duration<float> elapsed_seconds = end-start;
  std::cout << "In total: " << framesProcessed << " Frames processed in " << elapsed_seconds.count() << "s => " << framesProcessed/elapsed_seconds.count()<< " frames/s" << std::endl;

  for (std::vector<std::vector<std::ofstream*>>::iterator it = outfiles.begin(); it != outfiles.end(); ++it) {
    for (std::vector<std::ofstream*>::iterator itt = (*it).begin(); itt != (*it).end(); ++itt) {
      if ((*itt) != nullptr) {
        (*itt)->close();
        delete (*itt);
      }
    }
  }

  } else {
    std::cout << "Reading from file " << adcInFile << std::endl;
    std::ifstream inFile(adcInFile, std::ios::in | std::ios::binary);

    if (!inFile.is_open()) { 
      std::cout << "ERROR: can't read file " << adcInFile << std::endl;
      return EXIT_FAILURE;
    }

    short adcValues[16];
    int i;
//    while (!inFile.eof()) {
    for (int j = 0; j < 100; ++j){
      inFile.read(reinterpret_cast<char*>(&adcValues[0]), 16*sizeof(adcValues[0]));
//      inFile.read((char*)&adcValues,16*sizeof(short));
      for (i=0; i < 16; ++i) {
        std::cout << adcValues[i] << "\t";
      }
      std::cout << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
