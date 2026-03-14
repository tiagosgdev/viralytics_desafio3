
## Test 1

10k img | yolov8l | batch=16 > 1.311 hours

                   all        970       1614      0.731      0.726      0.784      0.655
      short_sleeve_top        228        231      0.823      0.806      0.873      0.747
       long_sleeve_top        138        138      0.728      0.739      0.762       0.65
  short_sleeve_outwear         78         79      0.677      0.734      0.789      0.667
   long_sleeve_outwear        105        105      0.699      0.667      0.759      0.634
                  vest         93         94      0.708      0.734      0.823      0.662
                 sling         75         75      0.814       0.76      0.835      0.684
                shorts        155        156      0.836      0.785       0.87      0.677
              trousers        247        249       0.92      0.832      0.934      0.724
                 skirt        170        170      0.766      0.692      0.797      0.669
    short_sleeve_dress         75         77      0.529      0.662      0.616      0.531
     long_sleeve_dress         74         74      0.676       0.62      0.646      0.567
            vest_dress         90         90      0.622      0.644      0.697      0.601
           sling_dress         75         76        0.7      0.767      0.794      0.707


── Per-class mAP@50 ──────────────────────────────

            Category  mAP@50
            trousers  0.9149
    short_sleeve_top  0.8598
              shorts  0.8556
               sling  0.8263
                vest  0.8218
         sling_dress  0.7906
               skirt  0.7707
short_sleeve_outwear  0.7578
     long_sleeve_top  0.7513
 long_sleeve_outwear  0.7434
          vest_dress  0.6786
   long_sleeve_dress  0.6152
  short_sleeve_dress  0.5886

   Overall mAP@50   : 0.7673
   Overall mAP@50:95: 0.6637
   Precision        : 0.7229
   Recall           : 0.7302



## Test 2

10k img | yolov8l | batch=26 > 1.316 hours

                   all        970       1614      0.723      0.712      0.777      0.652
      short_sleeve_top        228        231       0.79      0.779      0.857      0.729
       long_sleeve_top        138        138      0.674      0.718       0.76      0.657
  short_sleeve_outwear         78         79      0.726      0.684      0.785      0.696
   long_sleeve_outwear        105        105      0.715      0.691       0.77      0.637
                  vest         93         94      0.742      0.787      0.852      0.697
                 sling         75         75       0.78        0.8      0.865      0.699
                shorts        155        156       0.81      0.821      0.864      0.694
              trousers        247        249      0.919      0.839      0.941      0.729
                 skirt        170        170      0.761      0.706      0.793      0.651
    short_sleeve_dress         75         77      0.545      0.571       0.54      0.455
     long_sleeve_dress         74         74      0.586      0.568       0.64      0.566
            vest_dress         90         90      0.711      0.629       0.71      0.645
           sling_dress         75         76      0.645      0.668      0.723      0.623
Speed: 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.3ms postprocess per image


── Per-class mAP@50 ──────────────────────────────

Category  mAP@50
            trousers  0.9252
              shorts  0.8695
               sling  0.8550
                vest  0.8524
    short_sleeve_top  0.8436
               skirt  0.7765
short_sleeve_outwear  0.7562
     long_sleeve_top  0.7421
 long_sleeve_outwear  0.7408
         sling_dress  0.7112
          vest_dress  0.7094
   long_sleeve_dress  0.5901
  short_sleeve_dress  0.5069

   Overall mAP@50   : 0.7599
   Overall mAP@50:95: 0.6593
   Precision        : 0.7236
   Recall           : 0.7117



---

## Results:

Test 2 (batch=26) performed slightly worse than Test 1 (batch=16) across the board:

  ┌───────────┬───────────────────┬───────────────────┐
  │  Metric   │ Test 1 (batch=16) │ Test 2 (batch=26) │
  ├───────────┼───────────────────┼───────────────────┤
  │ mAP@50    │ 0.7673            │ 0.7599            │
  ├───────────┼───────────────────┼───────────────────┤
  │ mAP@50:95 │ 0.6637            │ 0.6593            │
  ├───────────┼───────────────────┼───────────────────┤
  │ Precision │ 0.7229            │ 0.7236            │
  ├───────────┼───────────────────┼───────────────────┤
  │ Recall    │ 0.7302            │ 0.7117            │
  └───────────┴───────────────────┴───────────────────┘

  Nearly identical training time (~1.31h). Batch 16 is the better config — slightly better mAP and recall. The bigger batch
  didn't help here.

  Weakest classes in both tests: short_sleeve_dress (~0.54-0.59 mAP) and long_sleeve_dress (~0.62-0.64). These could benefit
  from more training data or targeted augmentation.
