# CAD to Simulation — the `Detectors/CADSupport` tutorial

Start at **[docs/index.md](docs/index.md)**, or read the pages in order:

**Start**

1. [Install the software](docs/install.md)
2. [Convert your first model](docs/first-conversion.md)

**Converting**

3. [How a part is represented](docs/representation.md)
4. [Convert only part of a model](docs/partial.md)
5. [Give it materials](docs/materials.md)
6. [Field and cuts](docs/field-and-cuts.md)
7. [The geom.C file](docs/geom-c.md)

**Simulating**

8. [Add passive geometry](docs/passive.md)
9. [Make it produce hits](docs/hits.md)
10. [Grow it into a real detector](docs/real-detector.md)

**Worked example**

11. [The ITS, out and back again](docs/its-round-trip.md)

**Reference**

12. [Check your geometry](docs/checks.md)
13. [Limits and pain points](docs/limits.md)

## Reading it

Every page is plain Markdown and renders correctly in the GitHub file view: alerts use GitHub's own
`> [!NOTE]` syntax, the diagrams are ```mermaid fences, and the figures are ordinary images in
`docs/images/`. Nothing has to be published for someone to read this.

## Building the site

The same sources build a browsable site with search and a sidebar:

```bash
pip install mkdocs-material
mkdocs serve        # http://127.0.0.1:8000
mkdocs build        # static site in ./site
```

`hooks/github_alerts.py` turns the GitHub alerts into Material admonitions at build time, so the
Markdown stays GitHub-native and no extra plugin is needed.
