#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include <iostream>
#include <fmt/format.h>
#include "MCHMappingInterface/Segmentation.h"

int main(int argc, char** argv)
{
  rapidjson::OStreamWrapper osw(std::cout);
  rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);

  int deId = std::atoi(argv[1]);

  o2::mch::mapping::Segmentation seg{deId};

  w.StartObject();
  w.Key("channels");
  w.StartArray();

  seg.forEachPad([&](const int& padindex) {
    w.StartObject();
    w.Key("de");
    w.Int(deId);
    w.Key("bending");
    w.String(seg.isBendingPad(padindex) ? "true" : "false");
    w.Key("x");
    w.Double(seg.padPositionX(padindex));
    w.Key("y");
    w.Double(seg.padPositionY(padindex));
    w.Key("padindex");
    w.Int(padindex);
    w.Key("dsid");
    w.Int(seg.padDualSampaId(padindex));
    w.Key("dsch");
    w.Int(seg.padDualSampaChannel(padindex));
    w.EndObject();
  });

  w.EndArray();
  w.EndObject();
  return 0;
}
