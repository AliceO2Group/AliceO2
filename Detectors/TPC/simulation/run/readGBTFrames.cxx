///
/// @file   runSim.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>

#include <thread>
#include <mutex>
#include "TPCSimulation/GBTFrameContainer.h"
#include "TPCSimulation/HalfSAMPAData.h"

namespace bpo = boost::program_options;

std::mutex mtx;

void addData(AliceO2::TPC::GBTFrameContainer& container, std::string& infile, int& done) {
  done = 0;
  mtx.lock();
  std::cout << infile << std::endl;
  mtx.unlock();

  container.addGBTFramesFromBinaryFile(infile);

  done = 1;
}

void readData(AliceO2::TPC::GBTFrameContainer& container, std::vector<std::ofstream*>& outfiles, int& run, int& done) {
  done = 0;
  std::vector<AliceO2::TPC::HalfSAMPAData> data(5);
  while (!run) {
    while (container.getData(&data)){
      for (int i = 0; i < 5; ++i) {
        //std::copy(data[i].getData().begin(), data[i].getData().end(), std::ostreambuf_iterator<char>(*outfiles[i]));
        std::copy(data[i].getData().begin(), data[i].getData().end(), std::ostream_iterator<int>(*outfiles[i],"\t"));
        (*outfiles[i]) << "\n";
      }

//      std::cout 
//        << data[0].getID() << " "
//        << data[1].getID() << " "
//        << data[2].getID() << " "
//        << data[3].getID() << " "
//        << data[4].getID() << std::endl;
//      std::cout << data[0] << std::endl << std::endl;
//      std::cout << data[1] << std::endl << std::endl;
//      std::cout << data[2] << std::endl << std::endl;
//      std::cout << data[3] << std::endl << std::endl;
//      std::cout << data[4] << std::endl << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::microseconds{10});
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

  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::vector<std::string>>(&infile),   "Input data files")
    ("outfile,o",   bpo::value<std::vector<std::string>>(&outfile),  "Output data files")
    (",n",          bpo::value<std::vector<unsigned>>(&size),        "Container sizes")
    ("CRU",         bpo::value<std::vector<unsigned>>(&CRU),         "CRUs")
    ("link",        bpo::value<std::vector<unsigned>>(&link),        "links");
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  // Actual "work"
  std::vector<AliceO2::TPC::GBTFrameContainer*> container;
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

    container.push_back(new AliceO2::TPC::GBTFrameContainer(iSize,iCRU,iLink));
    container.back()->setEnableAdcClockWarning(false);
    container.back()->setEnableStoreGBTFrames(false);
  }

  std::vector<std::thread> threads;

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
        outname = infile[i];
      } else if (outfile[i] != "NOFILE") {
        outname = outfile[i];
      } else {
        outname = infile[i];
      }
      outname += "_SAMPA_";
      outname += std::to_string(j);
      outname += "_LOW";
      outname += ".adc";

      out = new std::ofstream(outname, std::ios::out | std::ofstream::binary);
      outfiles[i].push_back(out);//, std::ofstream::out);

      if (j == 2) continue;
      outname = "";
      if (i >= outfile.size()) {
        outname = infile[i];
      } else if (outfile[i] != "NOFILE") {
        outname = outfile[i];
      } else {
        outname = infile[i];
      }
      outname += "_SAMPA_";
      outname += std::to_string(j);
      outname += "_HIGH";
      outname += ".adc";

      out = new std::ofstream(outname, std::ios::out | std::ofstream::binary);
      outfiles[i].push_back(out);//, std::ofstream::out);
    }
    threads.emplace_back(readData,std::ref(*container[i]), std::ref(outfiles[i]), std::ref(addData_done[i]), std::ref(reading_done[i]));
  }

  while (not reading_done[0]) {
    std::this_thread::sleep_for(std::chrono::seconds{1});
    for (std::vector<AliceO2::TPC::GBTFrameContainer*>::iterator it = container.begin(); it != container.end(); ++it) {
      mtx.lock();
      std::cout << std::distance(container.begin(), it) << " " << (*it)->getSize() << " " << (*it)->getNFramesAnalyzed() << " " << (*it)->getNentries() << std::endl;
      mtx.unlock();
    }
  }

  for (auto &aThread : threads) {
    aThread.join();
  }

  for (std::vector<AliceO2::TPC::GBTFrameContainer*>::iterator it = container.begin(); it != container.end(); ++it) {
    std::cout << "Container " << std::distance(container.begin(), it) << " analyzed " << (*it)->getNFramesAnalyzed() << " GBT Frames" << std::endl;
    delete (*it);
  }

  for (std::vector<std::vector<std::ofstream*>>::iterator it = outfiles.begin(); it != outfiles.end(); ++it) {
    for (std::vector<std::ofstream*>::iterator itt = (*it).begin(); itt != (*it).end(); ++itt) {
      (*itt)->close();
      delete (*itt);
    }
  }

  return EXIT_SUCCESS;
}
