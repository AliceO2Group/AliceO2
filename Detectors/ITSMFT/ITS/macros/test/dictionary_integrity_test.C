/// \file CheckLookUp.C
/// Macro to check the integrity of the dictionary of cluster-topology. The ID of each entry is checked.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <string>
#include <fstream>
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#endif

using o2::itsmft::Cluster;
using o2::itsmft::ClusterPattern;
using o2::itsmft::LookUp;
using o2::itsmft::TopologyDictionary;

void dictionary_integrity_test(string intput_name = "complete_dictionary.bin", string output_name = "dictionary_test.txt")
{

  TopologyDictionary dict;
  dict.ReadBinaryFile(intput_name.c_str());
  LookUp finder(intput_name.c_str());

  int mistake_counter = 0;

  std::ofstream output_file(output_name.c_str());

  int dict_size = dict.GetSize();
  for (int ID_input = 0; ID_input < dict_size; ID_input++) {

    //***************** input **************************

    ClusterPattern cp_input = dict.GetPattern(ID_input);
    unsigned char patt_input[Cluster::kMaxPatternBytes];
    memcpy(&patt_input[0], &cp_input.getPattern()[2], Cluster::kMaxPatternBytes);
    int nRow_input = cp_input.getRowSpan();
    int nCol_input = cp_input.getColumnSpan();
    unsigned long input_hash = dict.GetHash(ID_input);
    if (input_hash == 0)
      continue;

    //**************** output **************************

    int ID_output = finder.findGroupID(nRow_input, nCol_input, patt_input);
    ClusterPattern cp_output = dict.GetPattern(ID_output);
    int nRow_output = cp_input.getRowSpan();
    int nCol_output = cp_input.getColumnSpan();
    unsigned long output_hash = dict.GetHash(ID_output);

    if (ID_output != ID_input) {
      output_file << "*****************************************" << std::endl;
      output_file << "                 INPUT" << std::endl;
      output_file << "ID : " << ID_input << std::endl;
      output_file << "nRow : " << nRow_input << std::endl;
      output_file << "nCol : " << nCol_input << std::endl;
      output_file << "Hash : " << input_hash << std::endl;
      output_file << cp_input << std::endl;
      output_file << "                 OUTPUT" << std::endl;
      output_file << "ID : " << ID_output << std::endl;
      output_file << "nRow : " << nRow_output << std::endl;
      output_file << "nCol : " << nCol_output << std::endl;
      output_file << "Hash : " << output_hash << std::endl;
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
