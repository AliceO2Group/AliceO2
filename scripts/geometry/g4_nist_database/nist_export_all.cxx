#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

// Geant4
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"

static std::string json_escape(const std::string& s)
{
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage:\n  " << argv[0] << " out.json\n";
    return 2;
  }

  const std::string out_json = argv[1];

  auto* nist = G4NistManager::Instance();

  // This returns all known NIST material names.
  std::vector<G4String> names = nist->GetNistMaterialNames();
  std::sort(names.begin(), names.end());

  std::ofstream out(out_json);
  if (!out) {
    std::cerr << "Cannot write: " << out_json << "\n";
    return 2;
  }

  out << std::fixed << std::setprecision(10);
  out << "{\n"
      << "  \"schema\": \"g4_nist_export_v1\",\n"
      << "  \"count_requested\": " << names.size() << ",\n"
      << "  \"materials\": {\n";

  bool first_mat = true;
  size_t built_ok = 0;
  size_t built_fail = 0;

  for (const auto& g4name : names) {
    // Build the material (some may fail depending on Geant4 build/config).
    G4Material* mat = nist->FindOrBuildMaterial(g4name, /*warning=*/false, /*isotopes=*/false);
    if (!mat) {
      ++built_fail;
      continue;
    }
    ++built_ok;

    const std::string name = g4name; // convert G4String -> std::string

    // Export in convenient units
    const double density_g_cm3 = mat->GetDensity() / (g / cm3);
    const double radlen_cm = mat->GetRadlen() / cm;
    const double intlen_cm = mat->GetNuclearInterLength() / cm;

    const size_t ne = mat->GetNumberOfElements();
    const auto* elems = mat->GetElementVector();
    const auto* fracs = mat->GetFractionVector(); // mass fractions (nullptr for some edge cases)

    if (!first_mat)
      out << ",\n";
    first_mat = false;

    out << "    \"" << json_escape(name) << "\": {\n";
    out << "      \"name\": \"" << json_escape(name) << "\",\n";
    out << "      \"density_g_cm3\": " << density_g_cm3 << ",\n";
    out << "      \"radlen_cm\": " << radlen_cm << ",\n";
    out << "      \"intlen_cm\": " << intlen_cm << ",\n";
    out << "      \"elements\": [\n";

    for (size_t i = 0; i < ne; ++i) {
      const G4Element* el = (*elems)[i];
      const int Z = static_cast<int>(el->GetZ());
      const double A_g_mol = el->GetA() / (g / mole);
      const double w = fracs ? fracs[i] : 0.0;

      out << "        {"
          << "\"symbol\": \"" << json_escape(el->GetSymbol()) << "\", "
          << "\"Z\": " << Z << ", "
          << "\"A_g_mol\": " << A_g_mol << ", "
          << "\"mass_fraction\": " << w
          << "}";

      if (i + 1 != ne)
        out << ",";
      out << "\n";
    }

    out << "      ]\n";
    out << "    }";
  }

  out << "\n  },\n"
      << "  \"count_built_ok\": " << built_ok << ",\n"
      << "  \"count_built_fail\": " << built_fail << "\n"
      << "}\n";

  std::cerr << "Wrote: " << out_json << "\n"
            << "NIST names: " << names.size() << ", built ok: " << built_ok
            << ", failed: " << built_fail << "\n";
  return 0;
}