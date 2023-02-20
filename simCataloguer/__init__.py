import gpt_2_simple as gpt2
import sys
from pathlib import Path, PurePath
import tensorflow as tf
import pixray_module

def get_rel_path(directory: str | Path, default_dir: Path) -> Path:
    if isinstance(directory, Path):
        return directory
    if PurePath(directory).is_absolute():
        return Path(directory)
    else:
        return default_dir.parent / Path(directory)

class Writer:
    def __init__(
        self, 
        model_dir: str | Path = 'current_models', 
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
        self.model_was_trained: bool = False

    def train_model(
            self, 
            file_name: str | Path, 
            run_name: str,
            checkpoint_dir: str | Path = 'checkpoint',  
        ):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path("outputs/hairout")
            checkpoint_path = get_rel_path(checkpoint_dir, self.default_dir)
        else:
            checkpoint_path = checkpoint_dir
        
        train_path = get_rel_path(file_name, self.default_dir)
        
        tf.compat.v1.reset_default_graph()
        if self.sess is None:
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
        self.model_was_trained = True

    def generate(
        self, 
        run_name: str, 
        length: int = 500, 
        temperature: float = 0.8, 
        n_samples: int = 1, 
        batch_size: int = 1
    ) -> list[str]:
        # Choose or refuse one text at a time - 300 to 500 words

        if self.sess is None:
            self.sess = gpt2.start_tf_sess()
        return gpt2.generate(
            self.sess, 
            run_name=run_name, 
            length=length, 
            temperature=temperature,
            nsamples=n_samples, 
            batch_size=batch_size,
            return_as_list = True
        )


    def make_pixray(
        self,
        prompt: str,
        output: str | Path,
        quality: str = "better",
        num_cuts: int = 10
    ):
        if not isinstance(output, Path):
            output = Path("outputs/hairout")
        pixray_module.run(
            prompt[0:70],
            "vdiff",
            quality=quality,
            custom_loss="edge, symmetry",
            edge_color="grey",
            num_cuts=num_cuts,
            outdir=str(output.absolute()),
            make_video = False,
            save_intermediates = False
        )

    def __call__(
        self,
        input_file: str | Path, 
        run_name: str, 
        output: str | Path,
        checkpoint_dir: str | Path = 'checkpoint',
        text_length: int = 500, 
        text_temperature: float = 0.8, 
        pixray_quality: str = "better",
        pixray_num_cuts: int = 10
    ):
        if not self.model_was_trained:
            self.train_model(
                file_name=input_file, 
                run_name=run_name, 
                checkpoint_dir=checkpoint_dir
            )
        generated_texts = self.generate(
            run_name=run_name, length=text_length, temperature=text_temperature
        )
        self.make_pixray(generated_texts[0], output, pixray_quality, pixray_num_cuts)

    # def train_model(
    #     self, 
    #     file_name: str, 
    #     run_name: str,
    #     checkpoint_dir: str = 'checkpoint',
    #     always_retrain: bool = False
    # ):
    #     if always_retrain or not (Path(checkpoint_dir) / run_name).exists():
    #         self._train_model(file_name, run_name, checkpoint_dir)

