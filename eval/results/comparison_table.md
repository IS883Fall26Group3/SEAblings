Labeling step, gemini-flash-lite-latest, temperature 0, 5 runs per version, 16 test pairs (97 skill requirements, team answer key; years rows left out). A call labels one CV against its jobs. Cost at $0.30 / $2.50 per 1M input / output tokens.

### Chain of thought vs none

| Version | Accuracy: mean (min-max), SD | Balanced accuracy | Met / partial / missing right | Quotes found in CV | Input tok / call | Output tok / call | Cost / 1,000 calls |
|---|---|---|---|---|---|---|---|
| App prompt (reasoning, 3 examples) | 68.5% (65%-71%), SD 2.5 | 66.4% | 90% / 23% / 86% | 100% | 2,796 | 727 | $2.66 |
| No written reasoning | 73.2% (71%-75%), SD 1.6 | 71.3% | 93% / 32% / 89% | 100% | 2,744 | 535 | $2.16 |

16 requirements where No written reasoning usually disagrees with the app prompt (app prompt right on 5, No written reasoning right on 10):
- P05 Structured Problem Solving: team partial, app prompt met, No written reasoning partial
- P09 Data Engineering: team met, app prompt partial, No written reasoning met
- P09 Apache Spark: team partial, app prompt partial, No written reasoning missing
- P13 Management Consulting: team missing, app prompt missing, No written reasoning partial
- P02 FP&A: team partial, app prompt partial, No written reasoning missing
- P02 Variance Analysis: team partial, app prompt met, No written reasoning partial
- P06 FP&A: team partial, app prompt partial, No written reasoning met
- P06 Variance Analysis: team partial, app prompt missing, No written reasoning met
- P06 P&L: team met, app prompt partial, No written reasoning met
- P10 Product Management: team partial, app prompt missing, No written reasoning partial
- P10 Pricing Models: team partial, app prompt met, No written reasoning partial
- P14 Business Analysis: team partial, app prompt met, No written reasoning partial
- P14 Testing: team missing, app prompt partial, No written reasoning missing
- P03 Performance Analysis: team partial, app prompt partial, No written reasoning missing
- P11 Biotech Consulting: team partial, app prompt met, No written reasoning partial

### Zero-shot vs one-shot vs few-shot

| Version | Accuracy: mean (min-max), SD | Balanced accuracy | Met / partial / missing right | Quotes found in CV | Input tok / call | Output tok / call | Cost / 1,000 calls |
|---|---|---|---|---|---|---|---|
| App prompt (reasoning, 3 examples) | 68.5% (65%-71%), SD 2.5 | 66.4% | 90% / 23% / 86% | 100% | 2,796 | 727 | $2.66 |
| One-shot (1 example) | 70.5% (66%-73%), SD 2.8 | 68.7% | 89% / 31% / 86% | 99% | 2,707 | 745 | $2.67 |
| Zero-shot (no examples) | 69.3% (68%-71%), SD 1.1 | 66.8% | 91% / 18% / 91% | 99% | 2,634 | 693 | $2.52 |

6 requirements where One-shot (1 example) usually disagrees with the app prompt (app prompt right on 1, One-shot (1 example) right on 5):
- P13 Management Consulting: team missing, app prompt missing, One-shot (1 example) partial
- P13 Life Sciences: team partial, app prompt met, One-shot (1 example) partial
- P02 Variance Analysis: team partial, app prompt met, One-shot (1 example) partial
- P10 Roadmap Strategy: team missing, app prompt partial, One-shot (1 example) missing
- P11 Biotech Consulting: team partial, app prompt met, One-shot (1 example) partial
- P12 Testing: team partial, app prompt met, One-shot (1 example) partial

9 requirements where Zero-shot (no examples) usually disagrees with the app prompt (app prompt right on 4, Zero-shot (no examples) right on 4):
- P09 Data Engineering: team met, app prompt partial, Zero-shot (no examples) missing
- P13 Management Consulting: team missing, app prompt missing, Zero-shot (no examples) partial
- P02 FP&A: team partial, app prompt partial, Zero-shot (no examples) missing
- P06 FP&A: team partial, app prompt partial, Zero-shot (no examples) missing
- P06 Variance Analysis: team partial, app prompt missing, Zero-shot (no examples) partial
- P10 Roadmap Strategy: team missing, app prompt partial, Zero-shot (no examples) missing
- P14 Business Analysis: team partial, app prompt met, Zero-shot (no examples) partial
- P03 Performance Analysis: team partial, app prompt partial, Zero-shot (no examples) met
- P12 Agile Methodologies: team missing, app prompt partial, Zero-shot (no examples) missing
