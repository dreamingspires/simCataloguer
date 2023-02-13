import gpt_2_simple as gpt2
import sys
from pathlib import Path, PurePath
import tensorflow as tf

class Writer:
    def __init__(self, download_model: bool = False, model_dir: str = 'current_models', model_name='124M') -> None:
        if PurePath(model_dir).is_absolute():
            model_path = Path(model_dir)
        else:
            file_path = Path.cwd() / sys.argv[0]
            model_path = file_path.parent / Path(model_dir)
        if download_model:
            gpt2.download_gpt2(model_dir = str(model_path), model_name=model_name)
        file_name = "obama_sotu.txt"

        tf.compat.v1.reset_default_graph()

        sess = gpt2.start_tf_sess()

        gpt2.finetune(sess,
            dataset=file_name,
            model_name='124M',
            steps=100,
            restore_from='fresh',
            run_name='simObama',
            print_every=10,
            sample_every=20,
            save_every=100
        )
