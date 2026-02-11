---
name: movie-sentiment-analysis
description: Analyze text sentiment through concise reasoning steps using contextual understanding and keyword cues.
---

# Movie-Sentiment-Analysis Skill

This skill classifies movie reviews as **positive** or **negative** based on film-specific sentiment indicators, context, and tone.

***

## Classification System

- **Positive (1)**: Favorable opinion, recommendation, satisfaction
- **Negative (0)**: Unfavorable opinion, criticism, disappointment

***

## Reasoning Chain

**Step 1 — Identify movie-specific cues**  
Detect sentiment in: acting, plot, directing, cinematography, pacing, entertainment value, rating indicators.

**Step 2 — Analyze modifiers**  
Check for negations ("not good"), intensifiers ("absolutely"), contrasts ("but"), and sarcasm.

**Step 3 — Weight key aspects**  
Acting and plot have highest impact. Strong negatives in core elements usually dominate overall sentiment.

**Step 4 — Conclude sentiment**  
Determine which sentiment dominates and provide brief reasoning.

***

## Keyword Reference

### Positive
masterpiece, brilliant, excellent, captivating, must-watch, engaging, gripping, powerful performance, stunning, beautiful cinematography, entertaining, touching, recommend

### Negative
terrible, disappointing, boring, waste of time, predictable, poorly written, wooden acting, confusing, slow, avoid, worst, plot holes, unwatchable

***

## Examples

**Input:**  
> "Absolutely brilliant! The acting was superb and the plot kept me hooked. Highly recommend."

**Output:**  
`positive — Strong praise for acting and plot with explicit recommendation.`

***

**Input:**  
> "What a letdown. Predictable plot, terrible acting, and I was bored throughout. Skip this one."

**Output:**  
`negative — Criticism of core elements (plot and acting) with clear warning to avoid.`

***

**Input:**  
> "Beautiful cinematography but the story was a mess. The weak script ruined it. Disappointing overall."

**Output:**  
`negative — Poor story and script outweigh technical positives; explicitly disappointing.`

