# UI Visual Refinement Plan

Last updated: 2026-06-13

This document turns the Canva visual direction exercise into a practical UI refinement plan for Stats Learning Lab. The goal is modest: make the app clearer, calmer, and easier to learn from without making it feel like a black-box product or a flashy marketing page.

## Canva Visual Direction

Generated Canva draft title:

```text
Stats Learning Lab Visual Direction
```

Purpose:

- Create a restrained visual guide for a learning-first statistics and ML Streamlit app.
- Use only abstract/synthetic content.
- Avoid private data, screenshots with real user data, or uploaded datasets.
- Keep the style educational, calm, transparent, and professional.

Generated Canva candidates:

| Candidate | URL | Suggested use |
|---|---|---|
| Candidate 1 | https://www.canva.com/d/RUtTmGwfD-f9eb1 | Recommended starting direction |
| Candidate 2 | https://www.canva.com/d/aFkPvRd6-dh_Mxb | Alternative layout direction |
| Candidate 3 | https://www.canva.com/d/NrZ_CfdTdHXnTmi | Alternative visual treatment |
| Candidate 4 | https://www.canva.com/d/Jn6q44oz2tbLl4W | Alternative visual treatment |

The recommended next step is to review Candidate 1 first. If it feels too polished or too much like marketing, keep only its structure and simplify the app styling further.

## Visual Identity Direction

Preferred feel:

- A guided research notebook.
- A careful applied statistics lab.
- A transparent learning workspace.

Avoid:

- Claims that the app chooses the best model automatically.
- Dense developer-console pages.
- Overly decorative backgrounds.
- Large hero sections that distract from actual work.
- Too many badges, cards, or explanatory blocks on already complex pages.

Suggested palette:

- Primary: deep blue or blue-slate.
- Secondary: muted teal or soft indigo.
- Background: near-white.
- Section background: light slate-gray.
- Warning: muted amber.
- Error: muted red.
- Success: calm green.
- Text: slate/charcoal with high contrast.

## Component Guidance

Sidebar:

- Keep navigation grouped and readable.
- Use title case labels.
- Show dataset loaded/not loaded and row/column count only.
- Avoid adding too many counters or workflow badges in the sidebar.

Page headers:

- Keep each page title short.
- Use one sentence for purpose.
- Put longer guidance in a small expander or info card.
- Avoid repeating the same long educational text on every page.

Chart cards:

- Use a consistent title, variables used, method, sample size, and short interpretation note.
- Keep warnings visible but concise.
- Use chart height and axis labels consistently.
- Avoid duplicate Plotly element keys.

Warning and info cards:

- Use warnings for real statistical or data risks.
- Use info cards for workflow guidance.
- Do not make warnings sound like hard errors unless the app truly cannot continue.

Tables:

- Keep primary tables visible.
- Put very wide or raw tables in expanders.
- Offer CSV download where useful.
- Use clear column names that match report terminology.

Report sections:

- Use a stable order: overview, plan, data, cleaning, transformations, models, diagnostics, predictions, limitations.
- Clearly distinguish statistical inference coefficients from predictive ML coefficients.
- Keep limitations visible and conservative.

Empty states:

- Say what is missing.
- Say what the user should do next.
- Avoid technical stack traces in normal unsupported states.

## Priority UI Refinements

1. Keep the Home page short and practical.
   - Show what the app is, how to start, and the workflow.
   - Avoid long marketing sections and repeated module cards.

2. Make Analysis Plan feel lightweight.
   - It should invite planning, not feel like a form that must be perfect.
   - Use simple labels and examples.

3. Simplify crowded modeling pages.
   - Keep model controls at the top.
   - Put diagnostics, formulas, plots, and raw tables in clearly named tabs or expanders.

4. Improve PCA and clustering explanations.
   - PCA: explain PC1/PC2 using top positive and negative loadings.
   - Clustering: label clusters as algorithmic groupings, not real-world categories.

5. Improve Prediction model selection labels.
   - Show model name, target, task type, and timestamp instead of relying on run IDs.
   - After selection, show formula or model summary when available.

6. Keep report preview readable.
   - Use section headings and concise empty states.
   - Avoid dumping very long raw tables by default.

## Implementation Notes

- Prefer Streamlit native elements and the existing `src/ui/` components.
- Keep custom CSS lightweight and stable.
- Do not target fragile Streamlit internal class names unless absolutely necessary.
- Do not use external images from the internet.
- Do not insert user-uploaded data into custom HTML.
- Do not change modeling or calculation logic as part of visual refinements.

## Acceptance Criteria

- A new user can identify the next step within five seconds on each major page.
- Important statistical caveats are visible but not overwhelming.
- Repeated components look consistent across pages.
- Public demo users are reminded not to upload sensitive data.
- The UI feels like a learning lab, not an AutoML product.
