| boxes                          | method     |   IoU |   Dice |
|:-------------------------------|:-----------|------:|-------:|
| clean                          | edge_canny | 0.302 |  0.424 |
| clean                          | otsu       | 0.416 |  0.569 |
| clean                          | watershed  | 0.641 |  0.775 |
| clean                          | grabcut    | 0.594 |  0.717 |
| noisy sigma=30                 | edge_canny | 0.263 |  0.378 |
| noisy sigma=30                 | otsu       | 0.361 |  0.497 |
| noisy sigma=30                 | watershed  | 0.541 |  0.680 |
| noisy sigma=30                 | grabcut    | 0.453 |  0.605 |
| restored (balanced: bilateral) | edge_canny | 0.240 |  0.352 |
| restored (balanced: bilateral) | otsu       | 0.370 |  0.510 |
| restored (balanced: bilateral) | watershed  | 0.560 |  0.694 |
| restored (balanced: bilateral) | grabcut    | 0.438 |  0.587 |
| restored (quality: NLM)        | edge_canny | 0.203 |  0.306 |
| restored (quality: NLM)        | otsu       | 0.376 |  0.521 |
| restored (quality: NLM)        | watershed  | 0.513 |  0.662 |
| restored (quality: NLM)        | grabcut    | 0.469 |  0.616 |