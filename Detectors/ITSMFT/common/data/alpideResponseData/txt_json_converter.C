#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <iomanip> // for std::setw

// this function needs a .txt file containing the names of Miko's data_pixels_*_*.txt files
// here these names are stored in Vbb0files.txt (for Vbb=0V) and Vbb3files.txt (for Vbb=-3V)
// function returning a vector with the files names (without .txt in the end)
std::vector<std::string> fill_list_of_files(std::string Vbb)
{
  std::vector<std::string> list_of_files;
  // declare input file (containing the files names)
  std::ifstream Vbbfiles;
  std::string name = Vbb + "files.txt";
  // open the file
  Vbbfiles.open(name.data(), std::ifstream::in);
  std::string data_file;
  std::string data_file_no_txt;
  // store all the files names in the list
  while (Vbbfiles >> data_file && Vbbfiles.good()) {
    data_file_no_txt = data_file.substr(0, data_file.size() - 4);
    list_of_files.push_back(data_file_no_txt.data());
  }
  Vbbfiles.close();
  return list_of_files;
}

// function returning a json variable filled with files given in the arguments
nlohmann::json fill_json(std::string pathVbb, std::vector<std::string> list_of_files)
{
  nlohmann::json j;
  for (auto const file : list_of_files) {
    std::string txt = ".txt";
    // full path of the file
    std::string inpfname = pathVbb + file + txt;

    // read the file
    std::ifstream inpGrid;
    inpGrid.open(inpfname, std::ifstream::in);

    // define variables
    int nz = 0;
    float val, gx, gy, gz;
    int lost, untrck, dead, nele;

    const int npix = 5; // value set in AlpideSimReponse.h
    int val_size = npix * npix;
    std::vector<float> val_v;

    // read the first line of the file
    inpGrid >> nz;
    j[file]["nz"] = nz;
    // looping over the lines
    for (int iz = 0; iz < nz; iz++) {
      // looping over the 25 first elements of a line
      for (int ip = 0; ip < npix * npix; ip++) {
        inpGrid >> val;
        val_v.push_back(val);
        // store the 25 first elements of a line in a vector in the json file under "val"
        j[file][std::to_string(iz)]["val"] = val_v;
      }
      // emptying the vector for the next line
      val_v.clear();
      // storing the last 7 elements of a line
      inpGrid >> lost >> dead >> untrck >> nele >> gx >> gy >> gz;
      // storing these last 7 elements in the json file
      j[file][std::to_string(iz)]["lost"] = lost;
      j[file][std::to_string(iz)]["dead"] = dead;
      j[file][std::to_string(iz)]["untrck"] = untrck;
      j[file][std::to_string(iz)]["nele"] = nele;
      j[file][std::to_string(iz)]["gx"] = gx;
      j[file][std::to_string(iz)]["gy"] = gy;
      j[file][std::to_string(iz)]["gz"] = gz;
    }
    // close the file
    inpGrid.close();
  }
  return j;
}

// function that reads a .json file given in input
void read_json(std::string Vbb, std::vector<std::string> list_of_files, nlohmann::json j, std::string outputPath)
{
  // define variables
  int nz;
  float val, gx, gy, gz;
  int lost, untrck, dead, nele;
  const int npix = 5;

  std::string space = " ";
  std::ofstream output;
  std::string txt = ".txt";
  std::string slash = "/";
  std::string output_name;
  std::string outputFolder = outputPath + Vbb + slash;

  for (auto const file : list_of_files) {
    output_name = outputFolder + file + txt;
    output.open(output_name);
    // read the first line of the file
    nz = j[file]["nz"];
    output << nz << std::endl;
    // looping over the lines
    for (int iz = 0; iz < nz; iz++) {
      // looping over the 25 first elements of a line
      for (int ip = 0; ip < npix * npix; ip++) {
        val = j[file][std::to_string(iz)]["val"][ip];
        output << val << space;
      }
      // storing these last 7 elements in the json file
      lost = j[file][std::to_string(iz)]["lost"];
      dead = j[file][std::to_string(iz)]["dead"];
      untrck = j[file][std::to_string(iz)]["untrck"];
      nele = j[file][std::to_string(iz)]["nele"];
      gx = j[file][std::to_string(iz)]["gx"];
      gy = j[file][std::to_string(iz)]["gy"];
      gz = j[file][std::to_string(iz)]["gz"];
      output << lost << space << dead << space << untrck << space << nele << space << gx << space << gy << space << gz << std::endl;
    }
    output.close();
  }
}

// function converting a .json file into data_pixels_*-*.txt files (with Miko's convention)
// arguments:
//- Vbb: only Vbb0 and Vbb3 for now
//- jsonFile: name of the .json file chosen as input
//- outputPath: path of the (existing) directory where the .txt files will be saved (do not forget a "/" in the end)
void json2txt(std::string Vbb, std::string jsonFile, std::string outputPath)
{
  // outputPath = "/home/abigot/Documents/ALICE/json_test/nlohmann/all_from_ROOT_macro/ComparisonChecks/"
  std::ifstream fVbb(jsonFile);
  nlohmann::json jVbb;
  fVbb >> jVbb;
  fVbb.close();
  std::vector<std::string> list_of_files = fill_list_of_files(Vbb);
  read_json(Vbb, list_of_files, jVbb, outputPath);
}

// function converting Miko's data_pixels_*-*.txt files into 1 .json file
// arguments:
//- Vbb: only Vbb0 and Vbb3 for now
//- pathVbb: path to the folder containing Miko's .txt files (for the chosen Vbb value)
// command example: txt2json("Vbb3", "/home/abigot/alice/O2/Detectors/ITSMFT/common/data/alpideResponseData/Vbb-3.0V/")
void txt2json(std::string Vbb, std::string pathVbb)
{
  // std::string pathVbb0 = "/home/abigot/alice/O2/Detectors/ITSMFT/common/data/alpideResponseData/Vbb-0.0V/";
  //  declare a list to store the files names in strings
  std::vector<std::string> list_of_files = fill_list_of_files(Vbb);
  // filling a json object
  nlohmann::json jVbb = fill_json(pathVbb, list_of_files);
  // write prettified JSON to another file
  std::string output_name = Vbb + "_charge_collection_tables.json";
  std::ofstream outputVbb(output_name);
  outputVbb << std::setw(2) << jVbb;
  outputVbb.close();
}
