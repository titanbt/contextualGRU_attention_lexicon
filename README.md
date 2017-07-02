# Sentence modeling with Deep Neural Architecture using Lexicon and Attention Mechanism for Sentiment Analysis

## STS Corpus
1. Bi-CGRNN + CharAVs + LexW2Vs + GoogleW2Vs

  ```#sh run_attention_lex_contextualGRU_s140_sr_w2v.sh```
  
    - Trained model: https://drive.google.com/file/d/0B5dxQhIZFZ-1VlhnVGE1by1STGs/view?usp=sharing 
    
    (copy the trained model into weights/attention_context/s140 folder)
   
    - DependencyW2Vs: https://drive.google.com/file/d/0B5dxQhIZFZ-1eVo5Z0NRelBDV1E/view?usp=sharing
   
    - GoogleW2Vs: https://drive.google.com/file/d/0B5dxQhIZFZ-1X1FsZjBpNVcxYzQ/view?usp=sharing

2. Bi-CGRNN + CharAVs + LexW2Vs + GloveW2Vs

  ```#sh run_attention_lex_contextualGRU_s140_sr_glove.sh```

3. Bi-CGRNN + GoogleW2Vs

  ```#sh run_contextualGRU_s140_sr_w2v.sh```

4. Bi-GRNN + CharAVs + GoogleW2Vs

  ```#sh run_attention_GRU_s140_sr_w2v.sh```

5. Bi-GRNN + LexW2Vs + GoogleW2Vs

  ```#sh run_lexicon_GRU_s140_sr_w2v.sh```

## HCR Corpus
1. Bi-CGRNN + CharAVs + LexW2Vs + GoogleW2Vs

  ```#sh run_attention_lex_contextualGRU_hcr_sr_w2v.sh```

2. Bi-CGRNN + CharAVs + LexW2Vs + GloveW2Vs

  ```#sh run_attention_lex_contextualGRU_hcr_sr_glove.sh```

3. Bi-CGRNN + GoogleW2Vs

  ```#sh run_contextualGRU_hcr_sr_w2v.sh```

4. Bi-GRNN + CharAVs + GoogleW2Vs

  ```#sh run_attention_GRU_hcr_sr_w2v.sh```

5. Bi-GRNN + LexW2Vs + GoogleW2Vs

  ```#sh run_lexicon_GRU_hcr_sr_w2v.sh```
