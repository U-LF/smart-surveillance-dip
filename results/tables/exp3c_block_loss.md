| case           | method                    |   PSNR |   SSIM |     ms |
|:---------------|:--------------------------|-------:|-------:|-------:|
| 1% blocks lost | (degraded)                | 26.938 |  0.986 |  0.000 |
| 1% blocks lost | Telea in-painting         | 40.383 |  0.996 |  4.933 |
| 1% blocks lost | Navier-Stokes in-painting | 40.514 |  0.996 |  4.973 |
| 1% blocks lost | median 5x5 (naive)        | 23.020 |  0.743 |  1.261 |
| 3% blocks lost | (degraded)                | 21.603 |  0.955 |  0.000 |
| 3% blocks lost | Telea in-painting         | 34.999 |  0.987 | 10.253 |
| 3% blocks lost | Navier-Stokes in-painting | 34.954 |  0.987 | 11.538 |
| 3% blocks lost | median 5x5 (naive)        | 20.078 |  0.719 |  1.192 |
| 8% blocks lost | (degraded)                | 17.359 |  0.880 |  0.000 |
| 8% blocks lost | Telea in-painting         | 30.388 |  0.963 | 24.517 |
| 8% blocks lost | Navier-Stokes in-painting | 30.334 |  0.963 | 29.043 |
| 8% blocks lost | median 5x5 (naive)        | 16.791 |  0.663 |  1.266 |