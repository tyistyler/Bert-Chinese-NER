## Version
    python3
    pytorch>=1.2
## Installation
    pip install transformers,pytorch-crf,seqeval
## How to train the model
    python main.py --task weibo --model_type bert --model_dir weibo_model --do_train --do_eval (--use_crf)
## How to predict the new data
    python main.py --task weibo --model_type bert --model_dir weibo_model --do_pred (--use_crf)
## Some question about dataset
    the Weibo dataset we use has some questions, you can download it in [here](https://github.com/hltcoe/golden-horse/tree/master/data)
[here](https://github.com/hltcoe/golden-horse/tree/master/data)
