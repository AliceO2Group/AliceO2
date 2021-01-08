// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataRefUtils.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include <Monitoring/Monitoring.h>
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"
#include "Framework/SerializationMethods.h"
#include <boost/serialization/access.hpp>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

/// Dummy boost-serializable struct to perform some tests
class Foo
{
 public:
  int fBar1;
  double fBar2[2];
  std::vector<float> fBar3;
  std::string fBar4;

  Foo()
  {
    fBar1 = 1;
    fBar2[0] = 2.1;
    fBar2[1] = 2.2;
    fBar3 = {3.1, 3.2, 3.3};
    fBar4 = "This is FooBar!";
  };
  Foo(int bar1, double bar21, double bar22, std::vector<float>& bar3, std::string& bar4) : fBar1(bar1),
                                                                                           fBar4(bar4),
                                                                                           fBar3(bar3)
  {
    fBar2[0] = bar21;
    fBar2[1] = bar22;
  };

  friend class boost::serialization::access;

  /// Serializes the struct
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar& fBar1;
    ar& fBar2;
    ar& fBar3;
    ar& fBar4;
  }
};

/// Example of how to send around strings using DPL.
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    //
    DataProcessorSpec{
      "boost_serialized_producer", //
      Inputs{},                    //
      {
        OutputSpec{{"make"}, "TES", "BOOST"}, //
      },
      AlgorithmSpec{[](ProcessingContext& ctx) {
        auto& out1 = ctx.outputs().make<BoostSerialized<std::vector<Foo>>>({"TES", "BOOST"});
        // auto& out1 = ctx.outputs().make_boost<std::array<int,6>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<BoostSerialized<std::array<int,6>>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<std::array<int,6>>({ "TES", "BOOST" });
        for (size_t i = 0; i < 17; i++) {
          float iFloat = (float)i;
          std::vector<float> floatVect = {iFloat * 3.f, iFloat * 3.1f, iFloat * 3.2f};
          std::string string = "This is Foo!";
          out1.emplace_back(Foo{(int)i, 2. * iFloat, 2.1 * iFloat, floatVect, string});
        }
      }} //
    },   //
    DataProcessorSpec{
      "boost_serialized_consumer", //
      {
        InputSpec{{"make"}, "TES", "BOOST"}, //
      },                                     //
      Outputs{},                             //
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          LOG(INFO) << "Buffer ready to receive";

          auto in = ctx.inputs().get<BoostSerialized<std::vector<Foo>>>("make");
          std::vector<Foo> check;
          for (size_t i = 0; i < 17; i++) {
            float iFloat = (float)i;
            std::vector<float> floatVect = {iFloat * 3.f, iFloat * 3.1f, iFloat * 3.2f};
            std::string string = "This is Foo!";
            check.emplace_back(Foo{(int)i, 2. * iFloat, 2.1 * iFloat, floatVect, string});
          }

          size_t i = 0;
          for (auto const& test : in) {
            assert((test.fBar1 == check[i].fBar1));       // fBar1 wrong
            assert((test.fBar2[0] == check[i].fBar2[0])); // fBar2[0] wrong
            assert((test.fBar2[1] == check[i].fBar2[1])); // fBar2[1] wrong
            size_t j = 0;
            for (auto const& fBar3It : test.fBar3) {
              assert((fBar3It == check[i].fBar3[j])); // fBar3[j] wrong
              j++;
            }
            assert((test.fBar4 == check[i].fBar4)); // fBar4 wrong
            i++;
          }
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
        } //
      }   //
    }     //
  };
}
