# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
# Author: Sandro Wenzel <sandro.wenzel@cern.ch>
# Since: 2026-09

"""Render GitHub alert blockquotes as Material admonitions.

The pages are written with GitHub's own `> [!NOTE]` syntax so that they read correctly
when someone simply clicks the file in the repository. MkDocs does not know that syntax,
so this hook rewrites it into `!!! note` before the Markdown is parsed. The optional bold
line directly under the marker becomes the admonition title.
"""

import re

KIND = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "info",
        "WARNING": "warning", "CAUTION": "danger"}


def on_page_markdown(markdown, **kwargs):
    lines, out, i = markdown.split("\n"), [], 0
    while i < len(lines):
        m = re.match(r"^> \[!(\w+)\]\s*$", lines[i])
        if not m or m.group(1) not in KIND:
            out.append(lines[i])
            i += 1
            continue
        kind = KIND[m.group(1)]
        i += 1
        body = []
        while i < len(lines) and lines[i].startswith(">"):
            body.append(lines[i][2:] if lines[i].startswith("> ") else lines[i][1:])
            i += 1
        title = ""
        if body and re.match(r"^\*\*.+\*\*$", body[0].strip()):
            title = body.pop(0).strip()[2:-2]
            while body and not body[0].strip():
                body.pop(0)
        out.append(f'!!! {kind} "{title}"' if title else f"!!! {kind}")
        out.append("")
        out.extend("    " + b if b.strip() else "" for b in body)
        out.append("")
    return "\n".join(out)
