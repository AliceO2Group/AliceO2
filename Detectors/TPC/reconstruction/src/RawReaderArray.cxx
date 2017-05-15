/// \file RawReaderArray.cxx
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "TPCReconstruction/RawReaderArray.h"

#include "FairLogger.h" 

using namespace o2::TPC;

RawReaderArray::RawReaderArray()
  : mReaders()
{}

bool RawReaderArray::addInputFile(const std::vector<std::string>* infiles) {
  bool ret = false;
  for (const auto &f : *infiles) {
    ret |= addInputFile(f);
  }

  return ret;
}

bool RawReaderArray::addInputFile(std::string infile) {
  typedef boost::tokenizer<boost::char_separator<char> >  tokenizer;
  boost::char_separator<char> sep{":"};
  tokenizer tok{infile,sep};
  if (std::distance(tok.begin(), tok.end()) < 3) {
    LOG(ERROR) << "Not enough arguments" << FairLogger::endl;
    return false;
  }

  tokenizer::iterator it1= tok.begin();
  std::string path = boost::lexical_cast<std::string>(*it1);

  int region, link;
  try {
    ++it1;
    region = boost::lexical_cast<int>(*it1);
  } catch(boost::bad_lexical_cast) {
    LOG(ERROR) << "Please enter a region for " << path << FairLogger::endl;
    return false;
  }

  try {
    ++it1;
    link = boost::lexical_cast<int>(*it1);
  } catch(boost::bad_lexical_cast) {
    LOG(ERROR) << "Please enter a link number for " << FairLogger::endl;
    return false;
  }

  return addInputFile(region, link, path);
}

bool RawReaderArray::addInputFile(int region, int link, std::string path) {
  auto reader = mReaders.insert(std::make_pair(std::make_pair(region,link),RawReader()));
  return reader.first->second.addInputFile(region,link,path);
}
