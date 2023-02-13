import gpt_2_simple as gpt2
import sys
from pathlib import Path, PurePath
import tensorflow as tf

def get_rel_path(directory: str, default_dir: Path) -> Path:
    if PurePath(directory).is_absolute():
        return Path(directory)
    else:
        return default_dir.parent / Path(directory)

class Writer:
    def __init__(
        self, 
        model_dir: str = 'current_models', 
        model_name='124M'
    ) -> None:
        self.default_dir = Path.cwd() / sys.argv[0]
        self.model_name = model_name
        self.model_path = get_rel_path(model_dir, self.default_dir)
        has_downloaded_path = self.model_path/'model_has_downloaded.lock'
        if not has_downloaded_path.exists():
            gpt2.download_gpt2(model_dir = str(self.model_path), model_name=model_name)
            with open(has_downloaded_path, 'w+'):
                pass

        self.sess = None
        

    def train_model(
            self, 
            checkpoint_dir: str = 'checkpoint',  
            file_name = "obama_sotu.txt", 
            run_name: str = 'simObama'
        ):
        checkpoint_path = get_rel_path(checkpoint_dir, self.default_dir)

        train_path = get_rel_path(file_name, self.default_dir)
        tf.compat.v1.reset_default_graph()

        self.sess = gpt2.start_tf_sess()


        gpt2.finetune(self.sess,
            dataset=str(train_path),
            model_name=self.model_name,
            steps=100,
            restore_from='fresh',
            run_name=run_name,
            print_every=10,
            sample_every=20,
            save_every=100,
            checkpoint_dir = str(checkpoint_path),
            model_dir = str(self.model_path)
        )

    def generate(self, run_name: str = 'simObama'):
        return gpt2.generate(self.sess, run_name=run_name)



