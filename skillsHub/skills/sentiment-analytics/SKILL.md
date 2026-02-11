---
name: sentiment-analytics
description: Analyze text sentiment through concise reasoning steps using contextual understanding and keyword cues.
---

# Sentiment Analytics Skill

This skill determines whether text expresses a positive, negative, or neutral sentiment.  
It reasons through **keywords, context, and tone** to reach a consistent judgment.

---

## Reasoning Chain

**Step 1 — Identify sentiment cues**  
Detect positive or negative keywords that signal emotional intent.

**Step 2 — Interpret context and modifiers**  
Analyze how negations, contrasts, or nuanced phrasing modify the sentiment.

**Step 3 — Conclude overall sentiment**  
Decide which emotional direction (positive, negative, or neutral) dominates, and summarize the reasoning briefly.

---

## Keyword Reference

### Positive
good, great, excellent, amazing, best, love, friendly, fast  

### Negative
bad, terrible, awful, worst, hate, poor, slow, disappointing  

---

## Example

**Input:**  
> “The service was friendly and fast, though the food could be better.”

**Reasoning (following the chain):**  
1. **Identify sentiment cues:**  
   Positive keywords: “friendly,” “fast.” Negative phrase: “could be better.”
2. **Interpret context and modifiers:**  
   The contrast word “though” introduces a mild critique but does not outweigh earlier praise. 
3. **Conclude overall sentiment:**  
   Positive sentiment dominates — the reviewer seems satisfied overall. 

**Output:**  
`positive — Expresses overall satisfaction despite a minor complaint.`