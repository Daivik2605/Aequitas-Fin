# Parser probe - 2026-09-15

Source: `news_release.pdf` pages [12, 14]  |  needles: 8  |  proximity window: 300 chars

**Association** is the metric that matters: the value found *and* its row
label within the window. Presence alone means the number survived but may
have lost what it refers to.

| config | time | elements | tables | present | associated |
| --- | --- | --- | --- | --- | --- |
| fast | 2s | 206 | 0 | 8/8 | **4/8** |
| hi_res | 19s | 142 | 2 | 8/8 | **7/8** |
| hi_res+tables | 30s | 142 | 2 | 8/8 | **7/8** |
| hi_res+tables (markdown) | - | 142 | 2 | 7/8 | **6/8** |

### fast
- MISS: Issue of preferred shares -> 500 (value found, label far away)
- MISS: Redemption of preferred shares -> (650) (value found, label far away)
- MISS: Treasury shares -> (3) (value found, label far away)
- MISS: Corporate and Other -> 285 (value found, label far away)

### hi_res
- MISS: Corporate and Other -> 285 (value found, label far away)

### hi_res+tables
- MISS: Corporate and Other -> 285 (value found, label far away)

### hi_res+tables (markdown)
- MISS: Balance at beginning of period -> 4,949 (value absent)
- MISS: CIBC -> 1,882 (value found, label far away)
