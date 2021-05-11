/// \file dictionary_integrity_test.C
/// Macro to check the integrity of the dictionary of cluster-topology. The ID of each entry is checked.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <string>
#include <fstream>
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#endif

using o2::itsmft::ClusterPattern;
using o2::itsmft::LookUp;
using o2::itsmft::TopologyDictionary;

void dictionary_integrity_test(std::string dictfile = "", std::string output_name = "dictionary_test.txt")
{

  TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  dict.readBinaryFile(dictfile);
  LookUp finder(dictfile);

  int mistake_counter = 0;

  std::ofstream output_file(output_name.c_str());

  int dict_size = dict.getSize();
  for (int ID_input = 0; ID_input < dict_size; ID_input++) {

    //***************** input **************************

    ClusterPattern cp_input = dict.getPattern(ID_input);
    unsigned char patt_input[ClusterPattern::MaxPatternBytes];
    memcpy(&patt_input[0], &cp_input.getPattern()[2], ClusterPattern::MaxPatternBytes);
    int nRow_input = cp_input.getRowSpan();
    int nCol_input = cp_input.getColumnSpan();
    unsigned long input_hash = dict.getHash(ID_input);
    bool isGroup_input = dict.isGroup(ID_input);

    //**************** output **************************

    int ID_output = finder.findGroupID(nRow_input, nCol_input, patt_input);
    ClusterPattern cp_output = dict.getPattern(ID_output);
    int nRow_output = cp_input.getRowSpan();
    int nCol_output = cp_input.getColumnSpan();
    unsigned long output_hash = dict.getHash(ID_output);
    bool isGroup_output = dict.isGroup(ID_output);

    if (ID_output != ID_input) {
      output_file << "*****************************************" << std::endl;
      output_file << "                 INPUT" << std::endl;
      output_file << "ID : " << ID_input << std::endl;
      output_file << "nRow : " << nRow_input << std::endl;
      output_file << "nCol : " << nCol_input << std::endl;
      output_file << "Hash : " << input_hash << std::endl;
      output_file << "IsGroup : " << isGroup_input << std::endl;
      output_file << cp_input << std::endl;
      output_file << "                 OUTPUT" << std::endl;
      output_file << "ID : " << ID_output << std::endl;
      output_file << "nRow : " << nRow_output << std::endl;
      output_file << "nCol : " << nCol_output << std::endl;
      output_file << "Hash : " << output_hash << std::endl;
      output_file << "IsGroup : " << isGroup_output << std::endl;
      output_file << cp_output << std::endl;

      mistake_counter++;
    }
  }
  if (!mistake_counter) {
    std::cout << "Perfect : everything works" << std::endl;
  } else {
    std::cout << mistake_counter << " out of " << dict_size << " wrong: good luck" << std::endl;
  }
}
