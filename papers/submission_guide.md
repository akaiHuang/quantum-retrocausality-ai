# Submission Guide

This document provides instructions for submitting the paper to three target venues: arXiv, JOSS, and Computer Physics Communications.

---

## 1. arXiv (quant-ph)

### Overview
arXiv is a preprint server. Posting here establishes priority and makes the work immediately available to the community. The appropriate primary category is **quant-ph** (Quantum Physics), with possible cross-listings to **cs.SE** (Software Engineering) or **physics.comp-ph** (Computational Physics).

### Account Setup
1. If you do not have an arXiv account, register at https://arxiv.org/user/register.
2. New submitters require endorsement from an existing arXiv author in the quant-ph category. Request endorsement from a colleague who has previously posted to quant-ph, or use the automated endorsement system if eligible.

### Submission Steps
1. **Prepare the source files.** arXiv accepts LaTeX source. Gather the following into a single directory:
   - `retrocausality_framework.tex` (the main paper)
   - Any `.bib` files (if using BibTeX separately; this paper uses inline `thebibliography`)
   - Any figure files (`.pdf`, `.png`, `.eps`) if added later
   - The `elsarticle.cls` class file is available on arXiv by default, but if using a custom version, include it.

2. **Create a compressed archive:**
   ```bash
   cd papers/
   tar czf retrocausality_framework.tar.gz retrocausality_framework.tex
   ```

3. **Upload to arXiv:**
   - Go to https://arxiv.org/submit
   - Select primary category: **quant-ph**
   - Optional cross-list categories: **physics.comp-ph**, **cs.SE**
   - Upload the `.tar.gz` file
   - arXiv will attempt to compile the LaTeX. Check the preview for errors.

4. **Metadata:**
   - **Title:** `quantum-retrocausality-ai: An Open-Source Computational Framework for Retrocausal Quantum Mechanics`
   - **Abstract:** Use the abstract from the paper
   - **Comments:** `12 pages, 5 tables, open-source software available at https://github.com/akaihuang/quantum-retrocausality-ai`
   - **MSC/PACS/ACM classes:** Optional but helpful. Relevant PACS: `03.65.Ta` (Foundations of quantum mechanics), `03.65.Ud` (Entanglement and quantum nonlocality)
   - **Journal-ref:** Leave blank (preprint)
   - **DOI:** Leave blank until journal publication

5. **Review the preview.** arXiv generates a PDF preview. Check that all equations render correctly, tables are formatted, and references are complete.

6. **Submit.** The paper will appear on arXiv within 1-2 business days after submission.

### Timeline
- Submission to posting: 1-2 business days
- No peer review (preprint server)
- Updates can be submitted at any time as new versions

---

## 2. JOSS (Journal of Open Source Software)

### Overview
JOSS publishes short papers (typically 1-2 pages) describing open-source software with a clear research application. The review process focuses on software quality, documentation, and community value rather than scientific novelty. JOSS is indexed in major databases and provides a citable DOI.

### Eligibility Requirements
Before submitting, verify the following:
- [ ] The software is open-source (MIT license -- confirmed)
- [ ] The software is hosted in a public repository (GitHub)
- [ ] The repository has a clear README with installation instructions
- [ ] The software has automated tests (47 passing tests -- confirmed)
- [ ] The software has example usage (notebooks/ directory)
- [ ] There is a `CITATION.cff` or similar citation file (create if missing)
- [ ] The repository has a `LICENSE` file
- [ ] The repository has a `CONTRIBUTING.md` file (create if missing)

### Preparation Steps

1. **Create a JOSS paper file.** JOSS papers are written in Markdown, not LaTeX. Create a file `paper.md` in the repository root with the following YAML frontmatter:

   ```yaml
   ---
   title: 'quantum-retrocausality-ai: An Open-Source Computational Framework for Retrocausal Quantum Mechanics'
   tags:
     - Python
     - quantum mechanics
     - retrocausality
     - weak values
     - Bell inequality
     - quantum foundations
   authors:
     - name: Akai Huang
       orcid: 0000-0000-0000-0000
       affiliation: 1
   affiliations:
     - name: Independent Researcher
       index: 1
   date: 2026-02-08
   bibliography: paper.bib
   ---
   ```

   The body should be 1000-2000 words summarizing the software's purpose, functionality, and research context. It is much shorter than the full LaTeX paper.

2. **Create `paper.bib`** with the BibTeX entries from the LaTeX paper.

3. **Create `CITATION.cff`** in the repository root:
   ```yaml
   cff-version: 1.2.0
   message: "If you use this software, please cite it as below."
   title: "quantum-retrocausality-ai"
   version: 1.0.0
   date-released: 2026-02-08
   url: "https://github.com/akaihuang/quantum-retrocausality-ai"
   license: MIT
   authors:
     - family-names: Huang
       given-names: Akai
   ```

### Submission Steps

1. Go to https://joss.theoj.org/papers/new
2. Enter the GitHub repository URL
3. JOSS will run automated checks (whedon/buffy) on the repository
4. Two reviewers will be assigned (typically within 1-2 weeks)
5. The review is conducted as a GitHub issue, focusing on:
   - Installation and functionality
   - Documentation quality
   - Test coverage
   - Statement of need (why this software matters)
   - Community guidelines (CONTRIBUTING.md)
6. Address reviewer comments by updating the repository
7. Once approved, JOSS assigns a DOI and publishes the paper

### Timeline
- Submission to first review: 1-2 weeks
- Review period: 2-8 weeks (depends on reviewers)
- Total from submission to publication: typically 4-12 weeks

### Important Notes
- JOSS does not charge publication fees
- The JOSS paper should reference the arXiv preprint for the full technical details
- JOSS reviewers test the software by installing and running it, so ensure `pip install -r requirements.txt && pytest` works cleanly
- The JOSS paper is not a substitute for a full research paper; it is a software paper

---

## 3. Computer Physics Communications

### Overview
Computer Physics Communications (CPC) publishes papers describing computational methods and programs in physics. It is a well-established journal (Elsevier) with an impact factor of approximately 6.3. CPC papers typically describe software in detail with validation against known results, which aligns well with our framework.

### Submission Category
Submit under the **Computer Programs in Physics** section. The paper should describe the algorithms, implementation, and validation of the software.

### Preparation Steps

1. **Format the paper.** The LaTeX paper (`retrocausality_framework.tex`) is already formatted using the `elsarticle` document class, which is the standard template for Elsevier journals including CPC.

2. **Compile the paper locally** to verify formatting:
   ```bash
   cd papers/
   pdflatex retrocausality_framework.tex
   pdflatex retrocausality_framework.tex  # run twice for references
   ```

3. **Prepare supplementary material:**
   - A link to the GitHub repository
   - Optionally, a snapshot of the code archived on Zenodo (recommended for long-term reproducibility)

4. **Create a Zenodo archive** (recommended):
   - Go to https://zenodo.org
   - Link your GitHub repository
   - Create a release on GitHub (e.g., v1.0.0)
   - Zenodo will automatically archive the release and assign a DOI
   - Reference this DOI in the paper for reproducibility

### Submission Steps

1. **Go to the CPC submission portal:** https://www.editorialmanager.com/cpc/
2. **Create an account** if you do not have one.
3. **Start a new submission:**
   - Article type: **Computer Programs in Physics**
   - Upload the LaTeX source file(s)
   - Upload any figure files
   - Enter metadata (title, abstract, keywords, author information)
4. **Cover letter.** Write a brief cover letter emphasizing:
   - The three novel contributions (first open-source TSVF, first executable retrocausal models, complete quantum eraser simulation)
   - The 47 passing tests demonstrating correctness
   - The pedagogical and research value of the framework
   - The MIT license and public availability
5. **Suggested reviewers.** Suggest 3-5 potential reviewers with expertise in:
   - Quantum foundations (TSVF, weak values)
   - Retrocausal models (Bell theorem, hidden variables)
   - Scientific software development
   - Quantum information theory
6. **CPC Program Library.** CPC maintains a program library. After acceptance, you will be asked to deposit the code in the CPC Program Library with a unique identifier.

### CPC-Specific Requirements
- **Program summary:** CPC requires a structured program summary including: program title, licensing, programming language, nature of the problem, solution method, and references.
- **Test run output:** Include a description of test runs and expected output. Reference the 47 unit tests and key numerical results.
- **Restrictions:** State any limitations on the size of problems that can be handled (e.g., dense matrix storage limits practical system size to ~10-12 qubits).

### Timeline
- Submission to first decision: 6-12 weeks
- Revision period: 2-4 weeks
- Total from submission to publication: typically 3-6 months
- Publication fees: Elsevier charges for open access ($2,990 for CC-BY) but traditional publication is free (author pays no fee; readers access via subscription)

---

## Recommended Submission Order

1. **arXiv first.** Submit to arXiv (quant-ph) immediately to establish priority and get community feedback. This takes 1-2 days and has no review process.

2. **JOSS second.** Submit to JOSS within a week of the arXiv posting. The JOSS review focuses on software quality and is typically faster than traditional journals. The JOSS DOI provides a citable reference for the software itself.

3. **CPC third.** Submit to Computer Physics Communications after incorporating any feedback from the arXiv posting and JOSS review. The CPC paper provides the full archival publication with detailed algorithmic descriptions.

Note: JOSS and CPC serve different purposes and publishing in both is acceptable. The JOSS paper cites the software; the CPC paper describes the algorithms and validation in detail. Reference the arXiv preprint in both submissions.

---

## Checklist Before Submission

### Repository Preparation
- [ ] All 47 tests pass: `pytest tests/ -v`
- [ ] README.md is complete with installation instructions
- [ ] LICENSE file exists (MIT)
- [ ] requirements.txt is complete and minimal
- [ ] Code is well-documented with docstrings
- [ ] Example notebooks run without errors
- [ ] No sensitive data or API keys in the repository
- [ ] .gitignore excludes compiled files and virtual environments

### Paper Preparation
- [ ] LaTeX compiles without errors or warnings
- [ ] All references are cited and complete
- [ ] Tables and equations are correctly formatted
- [ ] Numerical results match the test suite output
- [ ] GitHub repository URL is correct in the paper
- [ ] Author information and affiliations are accurate
- [ ] ORCID is included (register at https://orcid.org if needed)

### Archival
- [ ] Create a GitHub release (v1.0.0) tagging the version described in the paper
- [ ] Archive the release on Zenodo for a permanent DOI
- [ ] Update the paper with the Zenodo DOI before CPC submission
