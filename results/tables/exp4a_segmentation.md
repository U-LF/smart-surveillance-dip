| boxes              | method         | family   |   IoU |   IoU_std |   Dice |       ms |
|:-------------------|:---------------|:---------|------:|----------:|-------:|---------:|
| ground-truth boxes | box            | baseline | 0.497 |     0.098 |  0.658 |    0.090 |
| ground-truth boxes | edge_canny     | edge     | 0.286 |     0.196 |  0.409 |   13.566 |
| ground-truth boxes | edge_sobel     | edge     | 0.201 |     0.163 |  0.306 |   15.552 |
| ground-truth boxes | otsu           | region   | 0.426 |     0.163 |  0.579 |   21.937 |
| ground-truth boxes | region_growing | region   | 0.544 |     0.139 |  0.693 | 3428.311 |
| ground-truth boxes | watershed      | region   | 0.692 |     0.100 |  0.814 |   12.683 |
| ground-truth boxes | grabcut        | region   | 0.642 |     0.173 |  0.766 | 1446.121 |
| YOLO boxes         | box            | baseline | 0.472 |     0.103 |  0.634 |    0.177 |
| YOLO boxes         | edge_canny     | edge     | 0.275 |     0.195 |  0.395 |   14.185 |
| YOLO boxes         | edge_sobel     | edge     | 0.179 |     0.156 |  0.276 |   16.506 |
| YOLO boxes         | otsu           | region   | 0.405 |     0.166 |  0.556 |   20.671 |
| YOLO boxes         | region_growing | region   | 0.524 |     0.153 |  0.673 | 3273.564 |
| YOLO boxes         | watershed      | region   | 0.633 |     0.127 |  0.766 |   12.887 |
| YOLO boxes         | grabcut        | region   | 0.590 |     0.184 |  0.723 | 1331.271 |