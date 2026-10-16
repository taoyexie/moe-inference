# KITE disclosure template contract

Reference: `D:\working\专利技术交底书模板.docx`
SHA256: `43CCFD509278626E93F8B5F693DD7248228B6A7C95B7AD907230C7F937E569FE`
Reference render: `template_render/template.pdf`, `page-1.png` through `page-4.png`.
All four source pages inspected. Packaged LibreOffice renderer attempted and failed because no soffice is installed/bundled. Read-only native Word export succeeded; use the same Word export and bundled Poppler for final visual QA, without saving through Word.

## Geometry and source patterns

One continuous section, portrait A4, 11905 x 16837 twips. Top/bottom 1440, left/right 1230; header 850, footer 991. One column, grid linePitch 312. No headers/footers, page number fields, drawings, comments or content controls. Preserve final sectPr byte-equivalent.
Source Normal style ID 1 is Times New Roman / 宋体 10.5pt; visible answer paragraphs override to 12pt, 1.5-line spacing and firstLine420 twips. Use that existing answer pattern. All text black.
Source p0 is Heading1 24pt bold; replace its text with 专利技术交底书, preserve role. Source p2 centered14pt bold 技 术 交 底 书. Source p3–6 left12ptbold contact fields. Questions p8,p10,p16,p49,p53,p55,p57,p59 are 12ptbold,1.5 lines. Clone heading role for numbered subsections, keep next. Answers clone p9 12ptnormal,1.5line,420twipsfirstline; no artificial new design system.
Source includes two LED example tables with colored cells and black borders. Their CONTENT and structures are sample slots, not required patent elements. Replace with two compact technical comparison/record tables at full usable width, visible D9D9D9 borders, pale gray headers, 10.5pt cells, repeat header and no split rows. Preserve source Table Grid style. New figures are black-and-white boxed system/flow diagrams, inserted inline with captions, not decorative boxes.

## Slot map and scope

word/document.xml / body / p[1] main title: replace template label.
p[3]: preserve technical disclosure subtitle. p[4–7]: invention name, contact placeholder, phone/email placeholders. p[9,11,17,50,54,56,58,60]: preserve eight-question structure. All sample answers, example LED tables, yellow highlighted instructions and empty placeholders: replace with KITE material; length expansion allowed by cloning body and heading patterns. Remove empty example spacers only; do not retain unrelated sample text.
Answers: 1 problem; 2 background and technical limitations; 3 system, object/capacity definitions, MRC prediction, four path calibration, historical proposal, batch correction, fusion, safe execution, online update, illustrative nonmeasurement example, implementation boundary, reserved experimental subsection; 4 corresponding expected advantages; 5 proposed creative combination; 6 alternatives; 7 terminology; 8 prior literature.
No layer-sensitive KV reuse R/N1/N2 or borrowed performance numbers. No phase-specific history buckets. No GPU measurements, novelty guarantee or measured gains. Current signal uses explicit scheduled P/D request ratio in this implementation, not GPU time. Experimental results left explicitly pending. Contact facts not inferred from PPT presenter identity.

## Package preservation

Baseline package has document.xml, document.xml.rels, styles.xml, settings.xml, fontTable.xml, theme1.xml, core/app/custom props, root rels and content types. ZIP directory entries preserved.
Preserve-only: styles.xml SHA f1bf34ab6bcdb3225488c744eee704db9967339203e7e395893cba29d3fab144; settings.xml SHA d3a3970510daaf83349237dcb04481bbc7c0fb2f347db4aa1178428d72c9b2fa; fontTable.xml SHA 160621235277ee86ebd4deb595cf3c68c7f33a0de9cf3d91ca4cb7939ebe452c; theme1.xml SHA bea0a0590dea0ed9e97fa4391c1b7980640249387153014388eaaa628837b32c; custom props, root rels, app props unchanged.
Editable: document.xml, document.xml.rels (only append image rels), content types (only append media format), core properties (specific title and remove old author); add word/media assets. Generate from retained template object then transplant only editable parts into baseline ZIP. Verify every preserve-only part hash against baseline and source SHA against original.

## Fidelity and correctness gates

Eight questions all filled, original four contact fields preserved with placeholders, no LED/sample text or yellow highlight. Tables/diagrams readable at final physical width. No mathematical assertions of exact MRC under bypass admission, exact linear cost everywhere, workload-ratio optimality, free warm-cache resize or guaranteed global optimum. Explain uniform timing and recomputation prerequisites. Every final page rendered and inspected; use new QA directory per iteration. No tiny table text, clipped Chinese glyphs, orphan headings or excessive gaps. Deliver final DOCX only.
