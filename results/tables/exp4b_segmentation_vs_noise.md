| boxes                          | method     |   IoU |   Dice |
|:-------------------------------|:-----------|------:|-------:|
| clean                          | edge_canny | 0.299 |  0.421 |
| clean                          | otsu       | 0.416 |  0.569 |
| clean                          | watershed  | 0.642 |  0.776 |
| clean                          | grabcut    | 0.592 |  0.715 |
| noisy sigma=30                 | edge_canny | 0.263 |  0.377 |
| noisy sigma=30                 | otsu       | 0.363 |  0.498 |
| noisy sigma=30                 | watershed  | 0.542 |  0.681 |
| noisy sigma=30                 | grabcut    | 0.453 |  0.605 |
| restored (balanced: bilateral) | edge_canny | 0.241 |  0.352 |
| restored (balanced: bilateral) | otsu       | 0.371 |  0.512 |
| restored (balanced: bilateral) | watershed  | 0.560 |  0.694 |
| restored (balanced: bilateral) | grabcut    | 0.438 |  0.587 |
| restored (quality: NLM)        | edge_canny | 0.203 |  0.307 |
| restored (quality: NLM)        | otsu       | 0.377 |  0.522 |
| restored (quality: NLM)        | watershed  | 0.512 |  0.661 |
| restored (quality: NLM)        | grabcut    | 0.467 |  0.614 |