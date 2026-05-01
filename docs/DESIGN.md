# Design System Strategy: The Luminescent Studio

## 1. Overview & Creative North Star
**Creative North Star: "The Spectral Atelier"**

This design system is built to transform a standard utility into a high-end creative environment. We are moving away from the "SaaS-dashboard" aesthetic toward a sophisticated, editorial "studio" feel. The interface acts as a dark, atmospheric gallery where the AI-generated art is the hero, and the tools are sophisticated instruments that emerge from the shadows.

To achieve this, we reject rigid grids and heavy borders. Instead, we utilize **Intentional Asymmetry** and **Tonal Depth**. Elements should feel layered rather than placed, using overlapping panels and shifts in luminosity to guide the eye. This is not just a tool; it is a professional workspace for digital alchemists.

---

## 2. Colors & Surface Philosophy
The palette is rooted in deep obsidian tones, punctuated by high-energy, "electric" accents that represent the spark of artificial intelligence.

### The "No-Line" Rule
**Explicit Instruction:** You are prohibited from using 1px solid borders to define sections. Traditional dividers are the mark of a template. Boundaries must be defined through:
*   **Background Color Shifts:** Use `surface-container-low` for secondary sidebars against the `surface` background.
*   **Subtle Tonal Transitions:** Contrast the `background` (#0e0e12) against `surface-container` (#19191e) to create structural logic without visual noise.

### Surface Hierarchy & Nesting
Treat the UI as physical layers of frosted glass.
*   **Lowest Layer:** `surface-container-lowest` (#000000) for the main canvas or "void" where art is created.
*   **Base Layer:** `background` (#0e0e12) for the general application frame.
*   **Elevated Panels:** `surface-container-highest` (#25252b) for active tool windows or floating property inspectors.

### The "Glass & Gradient" Rule
To achieve the "High-Tech Studio" vibe, floating tool panels must utilize **Glassmorphism**. Combine `surface-variant` with a 40–60% opacity and a `backdrop-blur` of 20px–40px.

**Signature Texture:** Primary actions should never be flat. Use a subtle linear gradient from `primary` (#cf96ff) to `primary-dim` (#a533ff) at a 135-degree angle. This adds "soul" and a sense of energy to the interaction.

---

## 3. Typography
We utilize a dual-font strategy to balance technical precision with editorial authority.

*   **The Voice (Display & Headline):** Use **Space Grotesk**. This typeface brings a high-tech, geometric soul to the product. Use `display-lg` for impactful empty states or hero moments. Its wide apertures and technical rhythm convey innovation.
*   **The Utility (Title, Body, Label):** Use **Inter**. Inter is the workhorse. It provides maximum legibility for complex AI prompt engineering and tool settings.
*   **Editorial Scaling:** Don't be afraid of extreme scale. A `headline-lg` title sitting near `label-sm` metadata creates a sophisticated, modern contrast that feels designed, not just "populated."

---

## 4. Elevation & Depth
Depth is achieved through **Tonal Layering**, not structural lines.

*   **The Layering Principle:** Stack containers to create hierarchy. A `surface-container-low` sidebar containing `surface-container-high` cards creates a natural "lift" that feels premium and tactile.
*   **Ambient Shadows:** For floating elements (Modals, Context Menus), use ultra-diffused shadows.
    *   *Shadow Recipe:* Blur: 40px–60px | Opacity: 6% | Color: A tinted version of `primary` or `secondary` to simulate the glow of the screen reflecting off a studio wall.
*   **The "Ghost Border" Fallback:** If accessibility requires an edge, use a "Ghost Border." Apply `outline-variant` (#48474c) at **15% opacity**. This provides a hint of a boundary without breaking the seamless, dark-mode immersion.

---

## 5. Components

### Buttons
*   **Primary:** Gradient of `primary` to `primary-container`. Use `xl` (0.75rem) roundedness. No border. Text is `on_primary_fixed` (Black) for maximum punch.
*   **Secondary:** `surface-container-highest` background with a `secondary` (#00e3fd) "Ghost Border."
*   **States:** On hover, primary buttons should have a soft `surface_tint` outer glow (8px blur).

### Card-Based Layouts (The "Art Card")
*   **Structure:** No borders. Background: `surface-container`.
*   **Interaction:** On hover, the card should transition to `surface-container-high` and the `secondary` accent should appear as a 2px top-accent line or a subtle inner glow.

### Tool panels (Glassmorphism)
*   **Style:** `surface-variant` at 50% opacity.
*   **Blur:** 24px Backdrop Blur.
*   **Edge:** A "Ghost Border" using `outline-variant` at 20% opacity on the top and left sides only to simulate a light source.

### Input Fields (Prompt Bars)
*   **Style:** `surface-container-lowest` (pure black) to create a "well" effect.
*   **Focus State:** The `outline` transitions to a `secondary` (#00e3fd) glow.
*   **Typography:** Use `body-md` for user input, ensuring high contrast against the dark background.

### AI Specific: The "Generation Chip"
*   Use `tertiary` (#ff60cd) for status indicators (e.g., "AI Processing"). This vibrant pink creates a distinct visual lane for machine-learning activities, separate from standard UI actions.

---

## 6. Do's and Don'ts

### Do
*   **DO** use whitespace (spacing scale) as a separator. If you think you need a line, try adding 16px of space instead.
*   **DO** overlap elements. A tool panel that slightly overlaps the edge of the art canvas creates a sense of depth and professional "studio" clutter.
*   **DO** use `secondary_dim` for icons to keep them from being too distracting until they are interacted with.

### Don't
*   **DON'T** use pure white (#ffffff) for text. Always use `on_surface` (#fcf8fe) to prevent eye strain in dark mode.
*   **DON'T** use the `DEFAULT` (0.25rem) roundedness for large containers. Use `xl` (0.75rem) for panels and `full` for chips to maintain the sleek, modern aesthetic.
*   **DON'T** use high-contrast shadows. If the shadow is clearly visible as a dark smudge, it is too heavy. It should feel like an "aura," not a shadow.
