!mkdir -p ckpt/intent
!mkdir -p ckpt/slot
!wget https://www.dropbox.com/s/cjwinsy1gu6u2ya/best_intent_classification.model?dl=1 -O ckpt/intent/best.model
!wget https://www.dropbox.com/s/58hw81e9rfgprc7/best_slot_tagging.model?dl=1 -O ckpt/slot/best.model

!mkdir -p cache/intent
!mkdir -p cache/slot
!wget https://www.dropbox.com/s/uu0tzjn9oz8ezmc/intent_embeddings.pt?dl=1 -O cache/intent/embeddings.pt
!wget https://www.dropbox.com/s/1i99hiepihhtnib/intent_vocab.pkl?dl=1 -O cache/intent/vocab.pkl
!wget https://www.dropbox.com/s/3c4wvtfxyf4j0u7/intent2idx.json?dl=1 -O cache/intent/intent2idx.json
!wget https://www.dropbox.com/s/hybqxktxi9z5bls/slog_embeddings.pt?dl=1 -O cache/slot/embeddings.pt
!wget https://www.dropbox.com/s/uxcom850d6xla1o/slog_vocab.pkl?dl=1 -O cache/slot/vocab.pkl
!wget https://www.dropbox.com/s/wl8jc03hqhbzcm1/tag2idx.json?dl=1 -O cache/slot/tag2idx.json