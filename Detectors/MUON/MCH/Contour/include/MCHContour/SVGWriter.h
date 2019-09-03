// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_CONTOUR_SVGWRITER_H
#define O2_MCH_CONTOUR_SVGWRITER_H

#include "Contour.h"
#include "BBox.h"
#include <ostream>
#include <sstream>
#include "ContourCreator.h"
#include <utility>
#include <boost/format.hpp>
#include <cmath>

namespace o2
{
namespace mch
{
namespace contour
{

class SVGWriter
{

 public:
  SVGWriter(o2::mch::contour::BBox<double> viewingBox, int size = 1024)
    : mSVGBuffer{},
      mStyleBuffer{},
      mWidth{size},
      mHeight{static_cast<int>(std::round(size * viewingBox.height() / viewingBox.width()))},
      mViewingBox{viewingBox}
  {
  }

  template <typename T>
  SVGWriter& operator<<(T a)
  {
    mSVGBuffer << a;
    return *this;
  }

  void svgGroupStart(const std::string& classname)
  {
    mSVGBuffer << R"(<g class=")" << classname << R"(">)"
               << "\n";
  }

  void svgGroupEnd() { mSVGBuffer << "</g>\n"; }

  void box(double x, double y, double w, double h)
  {
    mSVGBuffer << boost::format(R"(<rect x="%f" y="%f" width="%f" height="%f"/>)") % x % y % w % h << "\n";
  }

  void text(const std::string& text, double x, double y)
  {
    mSVGBuffer << boost::format(R"(<text x="%f" y="%f">%s</text>)") % x % y % text.c_str() << "\n";
  }

  template <typename T>
  void polygon(const o2::mch::contour::Polygon<T>& p)
  {
    mSVGBuffer << R"(<polygon points=")";
    auto vertices = getVertices(p);
    for (auto j = 0; j < vertices.size(); ++j) {
      auto v = vertices[j];
      mSVGBuffer << v.x << "," << v.y << ' ';
    }
    mSVGBuffer << R"("/>)"
               << "\n";
  }

  template <typename T>
  void contour(const o2::mch::contour::Contour<T>& c)
  {
    for (auto& p : c.getPolygons()) {
      polygon(p);
    }
  }

  void points(const std::vector<std::pair<double, double>>& pts, double radius = 0.05)
  {
    for (auto& p : pts) {
      mSVGBuffer << boost::format(R"(<circle cx="%f" cy="%f" r="%f"/>)") % p.first % p.second % radius << "\n";
    }
  }

  void addStyle(const std::string& style) { mStyleBuffer << style << "\n"; }

  void writeStyle(std::ostream& os) { os << "<style>\n"
                                         << mStyleBuffer.str() << "</style>\n"; }

  void writeSVG(std::ostream& os)
  {
    os << boost::format(R"(<svg width="%d" height="%d" viewBox="%f %f %f %f">
)") % mWidth %
            mHeight % mViewingBox.xmin() % mViewingBox.ymin() % mViewingBox.width() % mViewingBox.height();

    os << mSVGBuffer.str();

    os << "</svg>";
  }

  void writeHTML(std::ostream& os)
  {
    os << "<html>\n";
    writeStyle(os);
    os << "<body>\n";
    writeSVG(os);
    os << "</body>\n";
    os << "</html>\n";
  }

 private:
  std::stringstream mSVGBuffer;
  std::stringstream mStyleBuffer;
  int mWidth;
  int mHeight;
  o2::mch::contour::BBox<double> mViewingBox;
};

} // namespace contour
} // namespace mch
} // namespace o2

#endif
