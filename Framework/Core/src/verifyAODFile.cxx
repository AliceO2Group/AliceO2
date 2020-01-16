#include "Framework/AnalysisDataModel.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/Logger.h"
#include "Framework/ASoA.h"
#include <iostream>
#include <memory>

using namespace o2::framework;
using namespace o2::soa;

int
main(int argc, char **argv) {
  if (argc != 2) {
    LOG(ERROR) << "Bad number of arguments";
  }
  auto infile = std::make_unique<TFile>(argv[1]);
  if (infile.get() == nullptr || infile->IsOpen() == false) {
    LOG(ERROR) << "File not found: " << argv[1];
    return 1;
  }

  /// FIXME: Substitute here the actual data you want to convert for the AODReader
  {
    std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2events", infile.get());
    TableBuilder collisionBuilder;
    RootTableBuilderHelpers::convertASoA<o2::aod::Collisions>(collisionBuilder, *reader);
    auto table = collisionBuilder.finalize();
    std::cout << table->schema()->ToString();
  }

  return 0;

  //{
  //  std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2tracks", infile.get());
  //  auto& trackParBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKPAR"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::Tracks>(trackParBuilder, *reader);
  //  auto table = trackParBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}

  //{
  //  std::unique_ptr<TTreeReader> covReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
  //  auto& trackParCovBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKPARCOV"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::TracksCov>(trackParCovBuilder, *covReader);
  //  auto table = trackParCovBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}

  //{
  //  std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
  //  auto& extraBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::TracksExtra>(extraBuilder, *extraReader);
  //  auto table = extraBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}

  //{
  //  std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2calo", infile.get());
  //  auto& extraBuilder = outputs.make<TableBuilder>(Output{"AOD", "CALO"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::Calos>(extraBuilder, *extraReader);
  //  auto table = extraBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}

  //{
  //  std::unique_ptr<TTreeReader> muReader = std::make_unique<TTreeReader>("O2muon", infile.get());
  //  auto& muBuilder = outputs.make<TableBuilder>(Output{"AOD", "MUON"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::Muons>(muBuilder, *muReader);
  //  auto table = muBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}

  //{
  //  std::unique_ptr<TTreeReader> vzReader = std::make_unique<TTreeReader>("O2vzero", infile.get());
  //  auto& vzBuilder = outputs.make<TableBuilder>(Output{"AOD", "VZERO"});
  //  RootTableBuilderHelpers::convertASoA<o2::aod::Muons>(vzBuilder, *vzReader);
  //  auto table = vzBuilder.finalise();
  //  std::cout << table.asArrowTable().schema()->ToString();
  //}
}
