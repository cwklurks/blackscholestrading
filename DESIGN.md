# Design System — Black-Scholes Trader (BST)

## Product Context
- **What this is:** A multi-model options pricing workstation with sensitivity analysis, volatility surfaces, strategy building, and backtesting
- **Who it's for:** Quantitative traders, finance students, options practitioners
- **Space/industry:** Financial derivatives, options pricing tools. Peers: Bloomberg Terminal, TradingView, Thinkorswim, QuantConnect, Deribit
- **Project type:** Data-dense web app / trading workstation

## Aesthetic Direction
- **Direction:** Industrial/Utilitarian with Luxury warmth
- **Decoration level:** Intentional — subtle warm tinting on all surfaces, no gradients, no blobs, no decorative borders. The warmth comes from the neutrals themselves.
- **Mood:** Bloomberg's density and precision, but with the warmth and authority of a copper-accented private office. Not cold, not flashy, not academic.
- **Reference sites:** Bloomberg Terminal (density, monospace discipline), TradingView (workspace-first layout), QuantConnect (amber/orange accent precedent)

## Typography
- **Display/Hero:** Geist Sans — clean geometric sans, loaded via next/font
- **Body:** Geist Sans — same family for cohesion, never more than 2 typefaces
- **UI/Labels:** Geist Sans
- **Data/Tables:** Geist Mono — ALL numerical data uses monospace. Inputs, outputs, Greeks, prices, metrics, table cells. tabular-nums always enabled.
- **Code:** Geist Mono
- **Loading:** Via `next/font/google` (already configured in layout.tsx)
- **Scale:**
  - Page title: 24px / weight 600 / letter-spacing -0.01em
  - Section label: 14px / weight 500
  - Body: 14px / weight 400
  - Data value: 14px mono / weight 500 / tabular-nums
  - Hero number (computed price): 36px mono / weight 700 / tabular-nums
  - Greek label: 12px / weight 500 / uppercase / letter-spacing 0.06em
  - Input value: 13px mono / weight 400 / tabular-nums
  - Muted/caption: 12px / weight 400

## Color

### Approach
Restrained. One accent (warm amber) + warm grays. Semantic colors for Greeks, P&L, and data states. Never pure black or white. All neutrals have a barely-visible warm brown/amber undertone.

### Dark Mode (Default)
| Token | Hex | HSL | Usage |
|-------|-----|-----|-------|
| --background | #141210 | 30 8% 6% | Page background |
| --foreground | #E8E4DF | 35 10% 90% | Primary text |
| --muted | #8A8580 | 30 6% 50% | Secondary text, labels |
| --muted-foreground | #8A8580 | 30 6% 50% | Same as muted |
| --surface | #1E1C1A | 30 6% 10% | Cards, panels, rail backgrounds |
| --card | #1E1C1A | 30 6% 10% | Same as surface |
| --border | #2A2725 | 30 4% 16% | Borders, dividers |
| --input | #252220 | 30 5% 13% | Input field backgrounds |
| --accent | #D4A017 | 42 80% 48% | Primary accent: buttons, active nav, links, focus rings |
| --accent-foreground | #141210 | 30 8% 6% | Text on accent backgrounds |
| --primary | #D4A017 | 42 80% 48% | Same as accent |
| --primary-foreground | #141210 | 30 8% 6% | Text on primary backgrounds |
| --ring | #D4A017 | 42 80% 48% | Focus ring color |
| --positive | #4CAF7D | 152 40% 49% | Profit, gains, positive values |
| --negative | #E05252 | 0 70% 60% | Loss, negative values, errors |
| --destructive | #E05252 | 0 70% 60% | Same as negative |

### Light Mode
| Token | Hex | HSL | Usage |
|-------|-----|-----|-------|
| --background | #FAF9F6 | 40 6% 97% | Page background |
| --foreground | #1A1A1A | 0 0% 10% | Primary text |
| --muted | #6B6B6B | 0 0% 36% | Secondary text |
| --surface | #F0EEEB | 35 5% 93% | Cards, panels |
| --border | #E0DCDA | 30 4% 82% | Borders |
| --input | #E8E5E2 | 30 4% 90% | Input backgrounds |
| --accent | #B8860B | 38 88% 38% | Accent (darker for contrast on light bg) |
| --positive | #2E8B57 | 152 50% 36% | Profit |
| --negative | #C0392B | 6 63% 46% | Loss |

### Semantic: Greek Colors
Each Greek gets a named token. These colors are used everywhere that Greek appears: sensitivity charts, result rows, tooltips, legends.

| Token | Hex | Usage |
|-------|-----|-------|
| --color-delta | #5B8DEF | Delta — cool blue, contrasts amber |
| --color-gamma | #45B899 | Gamma — teal-green |
| --color-vega | #D4A017 | Vega — the accent color (sensitivity to vol) |
| --color-theta | #E05252 | Theta — red (time decay, always costs) |
| --color-rho | #9B8EC4 | Rho — muted purple |

### Chart Colors
Charts derive their theme from the app token system, NOT from Plotly's stock themes.
- Grid lines: var(--border)
- Axis text: var(--muted)
- Chart background: transparent (container provides surface)
- Data series use semantic colors (Greek tokens, positive/negative)
- Chart font: Geist Sans for labels, Geist Mono for axis values

### Key Constraint
One accent color only (warm amber). Warm grays throughout. Never pure black (#000000) or pure white (#FFFFFF).

## Spacing
- **Base unit:** 4px
- **Density:** Comfortable — data-dense but not cramped
- **Scale:** 4 (2xs) | 8 (xs) | 12 (sm) | 16 (md) | 20 (lg) | 24 (xl) | 32 (2xl) | 48 (3xl) | 64 (4xl)

## Layout
- **Approach:** Three-zone workstation — grid-disciplined
  - Zone 1: Navigation rail (fixed left sidebar, 56px icon-only / 240px expanded at lg)
  - Zone 2: Primary workspace (dominant, flexible width)
  - Zone 3: Parameter rail (220-280px, flush panel with surface background, no card border)
- **Grid:** Content area max-w-7xl centered. Parameter rails use fixed widths.
- **Max content width:** 1280px (max-w-7xl)
- **Cards:** Only where the card IS the interaction (e.g., template picker, leg builder). Never as page scaffolding.
- **Parameter rails:** Flush panels with --surface background and border-right, NOT bordered cards.
- **Result areas:** Direct on --background, with border-bottom separating sections. No wrapping card.
- **Border radius:** 0.25rem (4px) for all elements. 9999px for pills only. No bubbly radii.

## Motion
- **Approach:** Minimal-functional — only transitions that aid comprehension
- **Easing:** enter(ease-out) exit(ease-in) move(ease-in-out)
- **Duration:** micro(50-100ms) short(150ms) medium(250ms)
- **Where motion appears:**
  - Computed results: fade-in 150ms ease-out
  - Tab switches: 100ms ease-in-out
  - Loading skeleton: pulse animation (existing)
  - Focus ring: 150ms transition on border-color
  - Button hover: 100ms background transition
- **Where motion does NOT appear:**
  - No scroll-driven effects
  - No entrance animations on page load
  - No parallax, no decorative motion

## Number Formatting
Context-appropriate precision for all numerical displays:

| Value | Format | Example |
|-------|--------|---------|
| Option Price | 2 decimals + $ | $7.97 |
| Delta | 3 decimals | 0.523 |
| Gamma | 3 decimals | 0.019 |
| Theta | 3 decimals + /d suffix | -0.013/d |
| Vega | 2 decimals | 18.23 |
| Rho | 2 decimals | 12.45 |
| Spot/Strike | 2 decimals | 187.32 |
| Volatility | 4 decimals or % | 0.2840 or 28.4% |
| P&L | 2 decimals + sign + color | +$142.50 (green) / -$87.30 (red) |

## Empty States
Every empty state is a guided start, not a dashed box with generic text.

| Page | Empty State Content |
|------|-------------------|
| Workspace/Pricing | Pre-filled ATM example with ghost result preview. "Price an AAPL call to get started" |
| Volatility | "Load a vol surface — try AAPL or SPY" with surface preview silhouette |
| Strategies | "Build a strategy — start with a call spread" with example template highlighted |
| Backtest | "Backtest a strategy — select a template above" |
| Market | "Enter a ticker to load market data" with popular tickers as chips |

## Interaction States

| State | Visual Treatment |
|-------|-----------------|
| Loading (fast, <500ms) | Spinner on button |
| Loading (slow, MC paths) | Progress indicator "Computing... 14/20 paths" |
| Loading (data fetch) | Skeleton pulse with warm muted color |
| Error | --negative background at 10% opacity, --negative text, specific error message + recovery action |
| Success | Results appear with 150ms fade-in. Computation model + timing shown ("Black-Scholes in 12ms") |
| Empty | Guided start (see table above) |

## Responsive
- **Desktop (lg+):** Full sidebar (240px) + parameter rail + workspace
- **Tablet (md):** Icon sidebar (56px) + stacked rail/workspace
- **Mobile (sm):** Bottom tab bar (hide sidebar), stacked layout, h-10 minimum touch targets (40px)

## Accessibility
- Input heights: h-10 (40px) minimum, h-11 (44px) on mobile
- Keyboard: Enter triggers computation from any parameter field
- ARIA: sidebar gets `aria-label="Main navigation"`, main area gets `role="main"`, skip-to-content link
- Color contrast: all text/background combinations meet WCAG AA

## Product Identity
- **Brand mark:** "BST" in Geist Mono, amber accent color, in sidebar header
- **Full name:** "Black-Scholes Trader" in page title/metadata
- **Page subtitles:** Show live context ("AAPL - $187.32 - IV 28.4%") instead of static descriptions

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-28 | Initial design system created | Created by /design-consultation based on /plan-design-review audit (2/10 → 7/10) and competitive research |
| 2026-03-28 | Warm amber accent (#D4A017) | Matches portfolio/CodeSprint brand. Every competitor uses blue/green — amber is genuinely differentiated |
| 2026-03-28 | Warm grays, never pure B/W | Portfolio consistency. Surfaces have barely-visible warm undertone |
| 2026-03-28 | 0.25rem border radius | Sharper than shadcn default (0.5rem). Reads as precision/institutional |
| 2026-03-28 | Semantic Greek color tokens | Delta always blue, Gamma always teal, everywhere. No more hardcoded hex |
| 2026-03-28 | Three-zone layout | Workstation, not dashboard template. Cards only where card IS the interaction |
| 2026-03-28 | Geist Mono for ALL numbers | Inputs, outputs, tables, metrics. tabular-nums always. Professional precision |
| 2026-03-28 | Context-appropriate number formatting | Delta 3dp, Price 2dp + $, Theta + /d suffix. Communicates domain expertise |
| 2026-03-28 | Auto-price on first load | User lands on working example, not blank form. Orientation before customization |
| 2026-03-28 | Shared ticker context across pages | React context so entering AAPL on Market carries to Volatility/Pricing |
| 2026-03-28 | Bottom tab bar on mobile | Hide sidebar below sm, show bottom tabs with icons + labels |
