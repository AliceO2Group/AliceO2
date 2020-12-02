// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>
#include <iostream>
#include <stdexcept>
#include <TGeoManager.h>
#include <TFile.h>
#include <sstream>
#include <tuple>
#include <vector>
#include <string>
#include "TGeoPhysicalNode.h"
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <gsl/span>
#include <cmath>
#include <array>
#include "MCHGeometryTransformer/Transformations.h"

namespace po = boost::program_options;

std::vector<std::string> splitString(const std::string& src, char delim)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}

TGeoManager* readFromFile(std::string filename)
{
  TFile* f = TFile::Open(filename.c_str());
  if (f->IsZombie()) {
    throw std::runtime_error("can not open " + filename);
  }

  auto possibleGeoNames = {"ALICE", "FAIRGeom", "MCH-ONLY", "MCH-BASICS"};

  TGeoManager* geo{nullptr};

  for (auto name : possibleGeoNames) {
    geo = static_cast<TGeoManager*>(f->Get(name));
    if (geo) {
      break;
    }
  }
  if (!geo) {
    f->ls();
    throw std::runtime_error("could not find ALICE geometry (using ALICE or FAIRGeom names)");
  }
  return geo;
}

template <typename WRITER>
void matrix2json(const TGeoHMatrix& matrix, WRITER& w)
{

  constexpr double rad2deg = 180.0 / 3.14159265358979323846;

  const Double_t* t = matrix.GetTranslation();
  const Double_t* m = matrix.GetRotationMatrix();
  gsl::span<double> mat(const_cast<double*>(m), 9);
  auto [yaw, pitch, roll] = o2::mch::geo::matrix2angles(mat);
  w.Key("tx");
  w.Double(t[0]);
  w.Key("ty");
  w.Double(t[1]);
  w.Key("tz");
  w.Double(t[2]);
  w.Key("yaw");
  w.Double(rad2deg * yaw);
  w.Key("pitch");
  w.Double(rad2deg * pitch);
  w.Key("roll");
  w.Double(rad2deg * roll);
}

std::tuple<bool, uint16_t> isMCH(std::string alignableName)
{
  auto parts = splitString(alignableName, '/');
  bool aliroot = parts[0] == "MUON";
  bool o2 = parts[0] == "MCH";
  if (!o2 && !aliroot) {
    return {false, 0};
  }
  auto id = std::stoi(parts[1].substr(2));
  bool ok = (aliroot && (id <= 15)) || (o2 && (id <= 19));
  uint16_t deId{0};
  if (ok && parts.size() > 2) {
    deId = std::stoi(parts[2].substr(2));
  }
  return {ok, deId};
}

template <typename WRITER>
void writeMatrix(const char* name, const TGeoHMatrix* matrix, WRITER& w)
{
  if (matrix) {
    w.Key(name);
    w.StartObject();
    matrix2json(*matrix, w);
    w.EndObject();
  }
}

/** convert geometry into a json document.
  */
void convertGeom(const TGeoManager& geom)
{
  rapidjson::OStreamWrapper osw(std::cout);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  writer.StartObject();
  writer.Key("alignables");
  writer.StartArray();

  for (auto i = 0; i < geom.GetNAlignable(); i++) {
    auto ae = geom.GetAlignableEntry(i);
    std::string symname = ae->GetName();
    auto [mch, deId] = isMCH(symname);
    if (!mch) {
      continue;
    }
    writer.StartObject();
    if (deId > 0) {
      writer.Key("deid");
      writer.Int(deId);
    }
    writer.Key("symname");
    writer.String(symname.c_str());
    auto matrix = ae->GetMatrix();
    bool aligned{true};
    if (!matrix) {
      matrix = ae->GetGlobalOrig();
      aligned = false;
    }
    writeMatrix("transform", matrix, writer);
    writer.Key("aligned");
    writer.Bool(aligned);
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description options;

  // clang-format off
    options.add_options()
     ("help,h","help")
     ("geom",po::value<std::string>()->required(),"geometry.root file");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(options);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program extract MCH geometry transformation from "
                 "a geometry root file and write them in json format.\n";
    std::cout << "\n";
    std::cout << options << "\n";
    std::cout << "\n";
    std::cout << "Note that the json format can then be further manipulated using e.g."
                 "the jq utility\n";
    std::cout << "For instance sorting by deid:\n";
    std::cout << "cat output.json | jq '.alignables|=sort_by(.deid)'\n";
    std::cout << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    std::cout << options << "\n";
    exit(1);
  }

  TGeoManager* geom = readFromFile(vm["geom"].as<std::string>());

  if (geom) {
    convertGeom(*geom);
    return 0;
  } else {
    return 3;
  }
  return 0;
}
