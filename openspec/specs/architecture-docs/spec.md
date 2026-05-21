# Architecture Docs

## Purpose

Specifies the structure and conventions for architecture explainer documents in `docs/architectures/`. These docs target readers with solid Python knowledge but limited prior exposure to neural network architectures or LLMs, and provide the agent conventions needed to produce consistent, high-quality documentation.

## Requirements

### Requirement: Architecture explainer files
The project SHALL provide one markdown explainer per model architecture (bigram, MLP, transformer) in `docs/architectures/`. Each file SHALL target readers with solid Python knowledge but limited prior exposure to neural network architectures or LLMs.

#### Scenario: Each architecture has a dedicated file
- **WHEN** the `docs/architectures/` directory is listed
- **THEN** files `bigram.md`, `mlp.md`, and `transformer.md` are present

### Requirement: TLDR section at top
Each architecture explainer SHALL open with a TLDR section that summarises: what the architecture is, what makes it interesting or distinctive, and where it sits on the capability spectrum relative to the others. The TLDR MUST be self-contained — a reader who stops there SHALL have a useful mental model.

#### Scenario: TLDR is the first section
- **WHEN** any architecture explainer is opened
- **THEN** the first section heading is "TLDR" and contains a concise summary

### Requirement: Detail sections below TLDR
Each explainer SHALL include deeper sections below the TLDR covering at minimum: how the architecture works mechanically, key hyperparameters and what they control, what the model can and cannot learn, and how it relates to the previous architecture in the sequence.

#### Scenario: Detail sections are present
- **WHEN** an architecture explainer is read past the TLDR
- **THEN** at least two additional sections with substantive content are present

### Requirement: Diagrams in architecture explainers
Each architecture explainer SHALL use ASCII diagrams to illustrate concepts inline (data flow, weight matrices, attention patterns, etc.). For diagrams too complex to render clearly in ASCII, an SVG file SHALL be provided in `docs/architectures/` and referenced from the markdown with a standard `![alt](filename.svg)` image tag.

#### Scenario: ASCII diagram present
- **WHEN** an architecture explainer is read
- **THEN** at least one ASCII diagram illustrating the model's structure or data flow is present

#### Scenario: SVG used for complex diagrams
- **WHEN** a concept requires a diagram that is not legible as ASCII
- **THEN** an SVG file is referenced from the markdown and present in `docs/architectures/`

### Requirement: Docs practices instructions file
The project SHALL maintain `agents/docs-practices.md` documenting the conventions for writing files in `docs/`. This file is read by agents contributing to the project and SHALL specify: audience assumptions, required document structure (TLDR + detail), tone and terminology guidance, and where new docs files belong.

#### Scenario: Instructions file exists and is findable
- **WHEN** the `agents/` directory is listed
- **THEN** `docs-practices.md` is present

#### Scenario: Instructions cover the required topics
- **WHEN** `agents/docs-practices.md` is read
- **THEN** it addresses audience, structure, tone, file placement, and diagram conventions (ASCII-first, SVG for complexity)
